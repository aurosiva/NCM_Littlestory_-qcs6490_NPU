"""Minimal Next-Concept Modeling (NCM).

Stage 1  MultiscaleTokenizer: text -> K levels of discrete codes (coarse -> fine)
         z = E(x);  for k: u_k = Down_k(z - zhat^(k-1)); r_k = nearest code;
         zhat^(k) = zhat^(k-1) + R_k(Up(e_k[r_k]));  decoder: zhat -> tokens
Stage 2  NCMGenerator: predicts all codes of level k in parallel from
         Down_k(zhat^(k-1)). Block-diagonal attention, so at inference each
         level is one small forward pass of l_k (+1 prefix) positions.

All Down/Up resizes are fixed matrices, so the same maths runs in plain numpy
on the board (see edge/run_ncm.py).
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GenCfg, TokCfg


def down_matrix(L, l):
    """[l, L] area (average) pooling from length L to l."""
    return F.adaptive_avg_pool1d(torch.eye(L).unsqueeze(0), l)[0].T.contiguous()


def up_matrix(L, l):
    """[L, l] linear interpolation from length l to L."""
    if l == L:
        return torch.eye(L)
    return F.interpolate(torch.eye(l).unsqueeze(0), size=L, mode="linear", align_corners=False)[0].T.contiguous()


def cascade_strides(total, base=4):
    """Split a stride into factors <= base, e.g. 128 -> [4, 4, 4, 2]."""
    out = []
    while total > 1:
        f = min(base, total)
        while total % f:
            f -= 1
        out.append(f)
        total //= f
    return out


def learned_down(d, total):
    """Strided convs, initialised to average pooling (official code: learned_l1_cascade)."""
    mods = []
    for s in cascade_strides(total):
        c = nn.Conv1d(d, d, s, stride=s)
        with torch.no_grad():
            c.weight.zero_(), c.bias.zero_()
            c.weight[torch.arange(d), torch.arange(d), :] = 1.0 / s
        mods.append(c)
    return nn.Sequential(*mods)


def learned_up(d, total):
    """Transposed strided convs, initialised to repeat (nearest) upsampling."""
    mods = []
    for s in reversed(cascade_strides(total)):
        c = nn.ConvTranspose1d(d, d, s, stride=s)
        with torch.no_grad():
            c.weight.zero_(), c.bias.zero_()
            c.weight[torch.arange(d), torch.arange(d), :] = 1.0
        mods.append(c)
    return nn.Sequential(*mods)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def rotate(x, cos, sin):
    """RoPE on [B,H,T,hd] with cos/sin [T,hd] (half-split layout, as in the official code)."""
    h = x.shape[-1] // 2
    return x * cos.to(x.dtype) + torch.cat([-x[..., h:], x[..., :h]], -1) * sin.to(x.dtype)


class Block(nn.Module):
    def __init__(self, dim, heads, dropout, qk_norm=False):
        super().__init__()
        self.heads, self.drop = heads, dropout
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))

    def forward(self, x, mask=None, rope=None):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = rotate(q, *rope), rotate(k, *rope)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.0)
        x = x + F.dropout(self.proj(a.transpose(1, 2).reshape(B, T, C)), self.drop, self.training)
        return x + F.dropout(self.mlp(self.n2(x)), self.drop, self.training)


# ----------------------------------------------------------------------------
# Stage 1: multiscale residual VQ tokenizer
# ----------------------------------------------------------------------------
class MultiscaleTokenizer(nn.Module):
    def __init__(self, cfg: TokCfg):
        super().__init__()
        self.cfg = cfg
        K, V, d, L = cfg.K, cfg.codebook, cfg.d, cfg.L
        self.tok_emb = nn.Embedding(cfg.vocab, d)
        self.enc_pos = nn.Parameter(torch.randn(L, d) * 0.02)
        self.enc = nn.ModuleList([Block(d, cfg.heads, cfg.dropout) for _ in range(cfg.enc_layers)])
        # final non-affine LayerNorm fixes the latent scale; without it the latent
        # grows during training, codebooks lag behind and the commitment loss explodes
        self.enc_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.LayerNorm(d, elementwise_affine=False))

        for k, l in enumerate(cfg.levels):
            self.register_buffer(f"D{k}", down_matrix(L, l), persistent=False)
            self.register_buffer(f"U{k}", up_matrix(L, l), persistent=False)
        # EMA codebooks, one per level
        self.register_buffer("codebook", torch.zeros(K, V, d))
        self.register_buffer("ema_count", torch.ones(K, V))
        self.register_buffer("ema_sum", torch.zeros(K, V, d))
        self.register_buffer("initted", torch.zeros(K, dtype=torch.bool))
        # R_k: per-level refinement after upsampling (starts as identity)
        self.phi = nn.ModuleList([nn.Conv1d(d, d, 3, padding=1) for _ in range(K)])
        for p in self.phi:
            nn.init.zeros_(p.weight), nn.init.zeros_(p.bias)
        if cfg.l1_learned:  # level 1 only: learned resize between L and l_1
            assert L % cfg.levels[0] == 0
            self.down_l1 = learned_down(d, L // cfg.levels[0])
            self.up_l1 = learned_up(d, L // cfg.levels[0])

        self.dec_pos = nn.Parameter(torch.randn(L, d) * 0.02)
        self.dec = nn.ModuleList([Block(d, cfg.heads, cfg.dropout) for _ in range(cfg.dec_layers)])
        self.dec_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab)

    def D(self, k):
        return getattr(self, f"D{k}")

    def U(self, k):
        return getattr(self, f"U{k}")

    # -- pieces ---------------------------------------------------------------
    def encode_latent(self, x):
        h = self.tok_emb(x) + self.enc_pos
        for b in self.enc:
            h = b(h)
        return self.enc_out(h)

    def qdown(self, k, x):
        """Down_k used for quantization: [B,L,d] -> [B,l_k,d]."""
        if k == 0 and self.cfg.l1_learned:
            return self.down_l1(x.transpose(1, 2)).transpose(1, 2)
        return self.D(k) @ x

    def up(self, k, q):
        if k == 0 and self.cfg.l1_learned:
            return self.up_l1(q.transpose(1, 2)).transpose(1, 2)
        return self.U(k) @ q

    def contrib(self, k, q):
        """R_k(Up(q)): code vectors [B,l_k,d] -> contribution to zhat [B,L,d]."""
        up = self.up(k, q)
        return up + self.phi[k](up.transpose(1, 2)).transpose(1, 2)

    def decode(self, zhat):
        h = zhat + self.dec_pos
        for b in self.dec:
            h = b(h)
        return self.head(self.dec_norm(h))

    def nearest(self, k, u):
        flat = u.reshape(-1, u.shape[-1]).float()
        cb = self.codebook[k]
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ cb.T + cb.pow(2).sum(1)[None]
        return dist.argmin(1).view(u.shape[:-1])

    @torch.no_grad()
    def _ema_update(self, k, u, idx):
        c = self.cfg
        flat, idx = u.reshape(-1, c.d).float(), idx.flatten()
        counts = torch.bincount(idx, minlength=c.codebook).float()
        sums = torch.zeros(c.codebook, c.d, device=u.device).index_add_(0, idx, flat)
        self.ema_count[k].mul_(c.ema_decay).add_(counts, alpha=1 - c.ema_decay)
        self.ema_sum[k].mul_(c.ema_decay).add_(sums, alpha=1 - c.ema_decay)
        n = self.ema_count[k].sum()
        cs = (self.ema_count[k] + 1e-5) / (n + c.codebook * 1e-5) * n
        self.codebook[k].copy_(self.ema_sum[k] / cs[:, None])
        # restart dead codes from random encoder outputs of this batch
        dead = self.ema_count[k] < 1e-2
        nd = int(dead.sum())
        if nd:
            pick = flat[torch.randint(0, flat.shape[0], (nd,), device=u.device)]
            self.codebook[k][dead] = pick
            self.ema_sum[k][dead] = pick
            self.ema_count[k][dead] = 1.0

    # -- training ---------------------------------------------------------------
    def forward(self, x):
        """Returns loss dict. Nested dropout: each sample is decoded from a
        randomly truncated hierarchy so coarse levels carry meaning."""
        c = self.cfg
        B = x.shape[0]
        z = self.encode_latent(x)
        if self.training and c.nested_dropout > 0:
            cut = torch.randint(0, c.K, (B,), device=x.device)
            cut = torch.where(torch.rand(B, device=x.device) < c.nested_dropout, cut, c.K - 1)
        else:
            cut = torch.full((B,), c.K - 1, device=x.device)

        # VAR-style quantization: residuals and code choice use the detached latent,
        # so there is no gradient chain through all K levels (that chain diverged).
        zd = z.detach()
        zhat = torch.zeros_like(zd)
        zsel = torch.zeros_like(zd)
        commit = latent = 0.0
        for k in range(c.K):
            u = self.qdown(k, zd - zhat.detach()).float()
            ud = u.detach()  # codebook search/EMA must stay outside the autograd graph
            if self.training and not self.initted[k]:
                flat = ud.reshape(-1, c.d)
                init = flat[torch.randint(0, flat.shape[0], (c.codebook,), device=x.device)]
                init = init + 0.01 * torch.randn_like(init)
                self.codebook[k].copy_(init), self.ema_sum[k].copy_(init), self.initted[k].fill_(True)
            idx = self.nearest(k, ud)
            if self.training:
                self._ema_update(k, ud, idx)
            q = self.codebook[k][idx]
            if k == 0 and c.l1_learned:  # straight-through at level 1 only, trains the learned down
                q = u + (q - u).detach()
            h = self.contrib(k, q)  # gradient reaches R_k (and the learned level-1 resize)
            zhat = zhat + h
            zsel = zsel + h * (k <= cut).view(B, 1, 1).to(h.dtype)
            if c.per_level_vq_loss:
                commit = commit + F.mse_loss(z.float(), zhat.detach().float()) / c.K
                latent = latent + F.mse_loss(zhat.float(), zd.float()) / c.K

        if not c.per_level_vq_loss:
            commit = F.mse_loss(z.float(), zhat.detach().float())   # pulls encoder towards its codes
            latent = F.mse_loss(zhat.float(), zd.float())           # trains R_k to rebuild z
        logits = self.decode(z + (zsel - z).detach())           # straight-through to the encoder
        rec = F.cross_entropy(logits.float().reshape(-1, c.vocab), x.reshape(-1))
        loss = rec + c.commit_beta * commit + c.latent_w * latent
        return {"loss": loss, "rec": rec, "commit": commit, "latent": latent}

    # -- helpers for the generator (frozen tokenizer) ---------------------------
    @torch.no_grad()
    def encode_codes(self, x):
        z = self.encode_latent(x)
        zhat, codes = torch.zeros_like(z), []
        for k in range(self.cfg.K):
            idx = self.nearest(k, self.qdown(k, z - zhat))
            codes.append(idx)
            zhat = zhat + self.contrib(k, self.codebook[k][idx].to(z.dtype))
        return codes

    @torch.no_grad()
    def level_inputs(self, codes):
        """Generator inputs: Down_k(zhat^(k-1)) for every level k."""
        B, L, d = codes[0].shape[0], self.cfg.L, self.cfg.d
        zhat = torch.zeros(B, L, d, device=codes[0].device)
        ins = []
        for k, idx in enumerate(codes):
            ins.append(self.D(k) @ zhat)
            zhat = zhat + self.contrib(k, self.codebook[k][idx])
        return ins, zhat


# ----------------------------------------------------------------------------
# Stage 2: NCM generator
# ----------------------------------------------------------------------------
class NCMGenerator(nn.Module):
    def __init__(self, gcfg: GenCfg, tcfg: TokCfg, codebook=None):
        super().__init__()
        self.levels = tcfg.levels
        self.tied = gcfg.tied_heads
        K, W, V, S = tcfg.K, gcfg.width, tcfg.codebook, sum(tcfg.levels)
        self.in_proj = nn.Linear(tcfg.d, W)
        self.prefix = nn.Parameter(torch.randn(1, 1, W) * 0.02)  # P=1 start token (unconditional)
        self.use_rope = gcfg.rope
        if not self.use_rope:
            self.pos = nn.Parameter(torch.randn(S, W) * 0.02)      # position within the whole hierarchy
        self.lvl = nn.Parameter(torch.randn(K, W) * 0.02)          # level embedding
        self.blocks = nn.ModuleList([Block(W, gcfg.heads, gcfg.dropout, gcfg.qk_norm) for _ in range(gcfg.layers)])
        self.norm = nn.LayerNorm(W)
        if self.tied:
            # official-code style: project to the latent space and score every code by
            # -||z - e||^2 (the same rule the tokenizer uses to pick codes), with a learned temperature
            self.head_proj = nn.Parameter(torch.randn(K, W, tcfg.d) * W ** -0.5)
            self.log_inv_T = nn.Parameter(torch.zeros(K))
            cb = codebook if codebook is not None else torch.zeros(K, V, tcfg.d)
            self.register_buffer("cb", cb.detach().clone().float())   # frozen tokenizer codebooks
            self.register_buffer("cb_sq", (self.cb ** 2).sum(-1))
            self.register_buffer("d_scale", self.cb_sq.mean(-1).clamp(min=1e-12))
        else:
            self.head_w = nn.Parameter(torch.randn(K, W, V) * 0.02)    # per-level code classifier
            self.head_b = nn.Parameter(torch.zeros(K, V))
        offs = [0]
        for l in self.levels:
            offs.append(offs[-1] + l)
        self.offs = offs
        # block-diagonal + prefix mask (True = may attend)
        m = torch.zeros(S + 1, S + 1, dtype=torch.bool)
        m[:, 0] = True
        for k, l in enumerate(self.levels):
            a, b = offs[k] + 1, offs[k + 1] + 1
            m[a:b, a:b] = True
        self.register_buffer("mask", m, persistent=False)
        if self.use_rope:
            hd = W // gcfg.heads
            self.register_buffer("inv_freq", 1.0 / gcfg.rope_base ** (torch.arange(hd // 2).float() / (hd // 2)),
                                 persistent=False)
            # prefix at 0, then positions restart at 0 inside every level
            pos = torch.cat([torch.zeros(1)] + [torch.arange(l).float() for l in self.levels])
            cos, sin = self._rope(pos)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def _rope(self, pos):
        f = pos.float()[:, None] * self.inv_freq[None]
        return torch.cat([f.cos(), f.cos()], -1), torch.cat([f.sin(), f.sin()], -1)

    def head(self, x, level):
        """x [B,n,W] -> code logits [B,n,V]. `level` is an int or a LongTensor of shape [1]."""
        if not torch.is_tensor(level):
            level = torch.tensor([level], device=x.device)
        if self.tied:
            z = x @ self.head_proj[level]                                    # [B,n,d]
            neg_d2 = 2 * z @ self.cb[level].transpose(-1, -2) - self.cb_sq[level].unsqueeze(1)
            return neg_d2 / self.d_scale[level].view(-1, 1, 1) * self.log_inv_T[level].exp().view(-1, 1, 1)
        return x @ self.head_w[level] + self.head_b[level].unsqueeze(1)

    def forward(self, zins):
        """Training: all levels in one pass. zins[k]: [B,l_k,d] -> list of logits [B,l_k,V]."""
        B = zins[0].shape[0]
        x = torch.cat([self.in_proj(z) + self.lvl[k] for k, z in enumerate(zins)], 1)
        if not self.use_rope:
            x = x + self.pos
        x = torch.cat([self.prefix.expand(B, -1, -1), x], 1)
        rope = (self.rope_cos, self.rope_sin) if self.use_rope else None
        for b in self.blocks:
            x = b(x, self.mask, rope)
        x = self.norm(x)[:, 1:]
        return [self.head(x[:, self.offs[k]:self.offs[k + 1]], k) for k in range(len(self.levels))]

    def step(self, zin, pos_ids, level, mask):
        """One level (also the exported ONNX graph). zin [B,n,d]; pos_ids [n] = offs[k]..offs[k]+n-1;
        level LongTensor [1]; mask bool [n+1,n+1] (prefix row sees only itself)."""
        x = self.in_proj(zin) + self.lvl[level]
        rope = None
        if self.use_rope:
            local = (pos_ids - pos_ids[0]).float()
            rope = self._rope(torch.cat([local.new_zeros(1), local]))
        else:
            x = x + self.pos[pos_ids]
        x = torch.cat([self.prefix.expand(x.shape[0], -1, -1), x], 1)
        m = torch.where(mask, 0.0, -1e9).to(x.dtype)
        for b in self.blocks:
            x = b(x, m, rope)
        return self.head(self.norm(x)[:, 1:], level)

    def forward_level(self, zin, k):
        """Inference: one level only (identical to the full pass thanks to the mask)."""
        n, dev = zin.shape[1], zin.device
        mask = torch.ones(n + 1, n + 1, dtype=torch.bool, device=dev)
        mask[0, 1:] = False
        return self.step(zin, torch.arange(self.offs[k], self.offs[k] + n, device=dev),
                         torch.tensor([k], device=dev), mask)


def sample_codes(logits, temperature=1.0, top_k=0):
    logits = logits.float() / max(temperature, 1e-5)
    if top_k:
        v, _ = logits.topk(top_k, -1)
        logits = logits.masked_fill(logits < v[..., -1:], float("-inf"))
    p = logits.softmax(-1)
    return torch.multinomial(p.reshape(-1, p.shape[-1]), 1).view(p.shape[:-1])


@torch.no_grad()
def generate(tok, gen, n=4, temperature=1.0, top_k=0):
    """Algorithm 2: K sequential steps, all codes of a level sampled in parallel."""
    c = tok.cfg
    dev = tok.codebook.device
    zhat = torch.zeros(n, c.L, c.d, device=dev)
    for k in range(c.K):
        logits = gen.forward_level(tok.D(k) @ zhat, k)
        idx = sample_codes(logits, temperature, top_k)
        zhat = zhat + tok.contrib(k, tok.codebook[k][idx])
    return tok.decode(zhat).argmax(-1)


def corrupt(codes, V, hi, lo):
    """Replace input codes with random ones at a rate decreasing linearly over levels."""
    K = len(codes)
    out = []
    for k, c in enumerate(codes):
        p = hi + (lo - hi) * k / max(K - 1, 1)
        m = torch.rand(c.shape, device=c.device) < p
        out.append(torch.where(m, torch.randint_like(c, V), c))
    return out


def level_weights(levels, alpha):
    """w_k = l_k^-alpha / sum_j l_j^(1-alpha)  (per-code weight; sums to 1 over all codes)."""
    den = sum(l ** (1 - alpha) for l in levels)
    return [l ** (-alpha) / den for l in levels]


def set_seed(s):
    random.seed(s), torch.manual_seed(s)
