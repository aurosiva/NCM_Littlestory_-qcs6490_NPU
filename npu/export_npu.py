"""Export NCM (v2) for the Hexagon NPU (QNN HTP, QCS6490 / v68).

NPU graphs need fixed shapes, so:
  gen.onnx   zin [128,d] (level input zero-padded to 128), lvl [1,W] (level embedding),
             mask [129,129] (additive; hides padding)  ->  h [128,W] (final-norm hidden states)
  dec.onnx   zhat [128,d]  ->  logits [128,vocab]
All inputs/outputs are rank 2 so the converter never reorders axes.

With --batch B, both graphs take B stories stacked on the first axis (one NPU call per level for
all B), and the decoder also does the argmax on the NPU, returning token ids instead of the large
logits table:
  gen_bB.onnx   zin [B*128,d], lvl [1,W], mask [129,129]  ->  h [B*128,W]
  dec_bB.onnx   zhat [B*128,d]  ->  tokens [B*128]
The codebook-distance head (h -> code logits) stays on the CPU in fp32: it is a difference of
large numbers that does not survive 8/16-bit rounding (see README, INT8 note).

Also writes calibration inputs recorded from real generation, and cpu_head.npz.

    .venv/bin/python npu/export_npu.py --ckpt ckpt_v2 --out npu/build
    .venv/bin/python npu/export_npu.py --ckpt ckpt_v2 --out npu/build --batch 4
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GenCfg  # noqa: E402
from model import NCMGenerator, rotate, sample_codes  # noqa: E402
from train_generator import load_tokenizer  # noqa: E402

NEG = -60.0  # additive mask value; must stay small so quantized attention scores keep resolution


def block_fwd(b, x, mask=None, rope=None):
    """Same maths as model.Block.forward (eval), written with plain ops for the converter."""
    B, T, C = x.shape
    H = b.heads
    q, k, v = b.qkv(b.n1(x)).reshape(B, T, 3, H, C // H).permute(2, 0, 3, 1, 4)
    if b.qk_norm:
        q, k = b.q_norm(q), b.k_norm(k)
    if rope is not None:
        q, k = rotate(q, *rope), rotate(k, *rope)
    s = q @ k.transpose(-1, -2) * (1.0 / math.sqrt(C // H))
    if mask is not None:
        s = s + mask
    a = s.softmax(-1) @ v
    x = x + b.proj(a.transpose(1, 2).reshape(B, T, C))
    return x + b.mlp(b.n2(x))


class GenNPU(nn.Module):
    def __init__(self, g, L, batch=None):
        super().__init__()
        self.g, self.batch, self.L = g, batch, L
        cos, sin = g._rope(torch.cat([torch.zeros(1), torch.arange(L).float()]))  # prefix at 0, then local 0..L-1
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, zin, lvl, mask):
        g = self.g
        B = self.batch or 1
        x = g.in_proj(zin.reshape(B, self.L, -1)) + lvl[None]
        x = torch.cat([g.prefix.expand(B, -1, -1), x], 1)
        for b in g.blocks:
            x = block_fwd(b, x, mask[None, None], (self.cos, self.sin))
        h = g.norm(x)[:, 1:]
        return h.reshape(B * self.L, -1) if self.batch else h[0]


class DecNPU(nn.Module):
    def __init__(self, tok, batch=None):
        super().__init__()
        self.t, self.batch = tok, batch

    def forward(self, zhat):
        t = self.t
        B = self.batch or 1
        h = zhat.reshape(B, t.cfg.L, -1) + t.dec_pos
        for b in t.dec:
            h = block_fwd(b, h)
        logits = t.head(t.dec_norm(h))
        if self.batch:  # token ids only: [B*L] instead of [B*L, vocab]
            return logits.argmax(-1).reshape(-1).to(torch.int32)
        return logits[0]


def npu_mask(n, L):
    """Prefix sees itself; the n real codes see prefix + each other; padding is hidden."""
    m = np.full((L + 1, L + 1), NEG, np.float32)
    m[0, 0] = 0
    m[1:, : n + 1] = 0
    return m


def head_np(w, h, k):
    """CPU head, numpy fp32: same as NCMGenerator.head (tied)."""
    z = h @ w["head_proj"][k]
    return (2 * z @ w["cb"][k].T - w["cb_sq"][k]) / w["d_scale"][k] * np.exp(w["log_inv_T"][k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_v2")
    ap.add_argument("--out", default="npu/build")
    ap.add_argument("--calib_stories", type=int, default=8, help="stories per level used for calibration")
    ap.add_argument("--batch", type=int, default=None, help="export batched graphs gen_bB/dec_bB (decoder returns tokens)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    tok = load_tokenizer(f"{args.ckpt}/tokenizer.pt", "cpu")
    ck = torch.load(f"{args.ckpt}/generator.pt", map_location="cpu")
    gen = NCMGenerator(GenCfg(**ck["cfg"]), tok.cfg)
    gen.load_state_dict(ck["model"])
    gen.eval()
    assert gen.tied and gen.use_rope, "NPU export is written for the v2 preset"
    L, d, W, K = tok.cfg.L, tok.cfg.d, gen.in_proj.out_features, tok.cfg.K
    levels = tok.cfg.levels
    gnpu, dnpu = GenNPU(gen, L).eval(), DecNPU(tok).eval()

    # ---- record real generation states (torch fp32), check padded graph == original step
    torch.manual_seed(0)
    B = 32
    zhat = torch.zeros(B, L, d)
    calib_gen, max_score, worst = [], 0.0, 0.0
    w = {"lvl": gen.lvl.numpy(), "head_proj": gen.head_proj.numpy(), "cb": gen.cb.numpy(),
         "cb_sq": gen.cb_sq.numpy(), "d_scale": gen.d_scale.numpy(), "log_inv_T": gen.log_inv_T.numpy()}
    for k, n in enumerate(levels):
        zin = tok.D(k) @ zhat
        ref = gen.forward_level(zin, k)
        pad = torch.zeros(B, L, d)
        pad[:, :n] = zin
        m = torch.from_numpy(npu_mask(n, L))
        for i in range(B):
            h = gnpu(pad[i], gen.lvl[k][None], m)
            got = head_np(w, h.numpy()[:n], k)
            worst = max(worst, float(np.abs(got - ref[i].numpy()).max()))
            if i < args.calib_stories:
                calib_gen.append((pad[i].numpy(), gen.lvl[k][None].numpy(), m.numpy()))
        # attention score range (to check NEG is far enough below)
        x = torch.cat([gen.prefix.expand(1, -1, -1), gen.in_proj(pad[:1]) + gen.lvl[k]], 1)
        for b in gen.blocks:
            q, kk, _ = b.qkv(b.n1(x)).reshape(1, L + 1, 3, b.heads, W // b.heads).permute(2, 0, 3, 1, 4)
            q, kk = b.q_norm(q), b.k_norm(kk)
            max_score = max(max_score, float((q @ kk.transpose(-1, -2)).abs().max() / math.sqrt(W // b.heads)))
            x = block_fwd(b, x, m[None, None], (gnpu.cos, gnpu.sin))
        idx = sample_codes(ref, 1.0, 50)
        zhat = zhat + tok.contrib(k, tok.codebook[k][idx])
    dref = tok.decode(zhat)
    dworst = max(float((dnpu(zhat[i]) - dref[i]).abs().max()) for i in range(4))
    print(f"padded NPU graph vs original: gen logits max diff {worst:.2e}, decoder {dworst:.2e}")
    print(f"max |attention score| {max_score:.1f} (mask value {NEG})")
    assert worst < 1e-2 and dworst < 1e-2
    assert max_score < -NEG / 2, "raise |NEG|"

    # ---- batched variant: same calibration data, B stories of one level per sample
    sfx, Bt = "", args.batch
    if Bt:
        assert args.calib_stories % Bt == 0 and B % Bt == 0, "calib_stories and 32 must be multiples of --batch"
        sfx = f"_b{Bt}"
        gnpu, dnpu = GenNPU(gen, L, Bt).eval(), DecNPU(tok, Bt).eval()
        grouped = []
        for s0 in range(0, len(calib_gen), Bt):  # calib_gen is level-major, calib_stories per level
            grp = calib_gen[s0:s0 + Bt]
            grouped.append((np.concatenate([a for a, _, _ in grp]), grp[0][1], grp[0][2]))
        # batched graph must equal the single-story graph, story by story
        a, lv, mk = (torch.from_numpy(x) for x in grouped[-1])
        one = torch.cat([GenNPU(gen, L)(a[i * L:(i + 1) * L], lv, mk) for i in range(Bt)])
        bdiff = float((gnpu(a, lv, mk) - one).abs().max())
        ztok = zhat[:Bt].reshape(Bt * L, d)
        tdiff = float((dnpu(ztok) != tok.decode(zhat[:Bt]).argmax(-1).reshape(-1)).float().mean())
        print(f"batch {Bt}: gen max diff vs single {bdiff:.2e}, decoder token mismatch {tdiff:.4f}")
        assert bdiff < 1e-3 and tdiff < 1e-3
        calib_gen = grouped
        dec_in = [zhat[j:j + Bt].reshape(Bt * L, d) for j in range(0, B, Bt)]
    else:
        dec_in = [zhat[j] for j in range(B)]

    # ---- ONNX (legacy exporter, fully static)
    ex = dict(export_params=True, opset_version=17, dynamo=False, do_constant_folding=True)
    zi, lv, mk = (torch.from_numpy(a) for a in calib_gen[0])
    torch.onnx.export(gnpu, (zi, lv, mk), f"{args.out}/gen{sfx}.onnx", input_names=["zin", "lvl", "mask"],
                      output_names=["h"], **ex)
    torch.onnx.export(dnpu, (dec_in[0],), f"{args.out}/dec{sfx}.onnx", input_names=["zhat"],
                      output_names=["tokens" if Bt else "logits"], **ex)

    # ---- calibration raws (float32, C order) + input lists for qairt-quantizer
    cdir = os.path.abspath(f"{args.out}/calib{sfx}")
    os.makedirs(cdir, exist_ok=True)
    with open(f"{args.out}/calib_gen{sfx}.txt", "w") as f:
        for j, (a, b_, c) in enumerate(calib_gen):
            for nm, arr in (("zin", a), ("lvl", b_), ("mask", c)):
                arr.astype(np.float32).tofile(f"{cdir}/g{j}_{nm}.raw")
            f.write(f"zin:={cdir}/g{j}_zin.raw lvl:={cdir}/g{j}_lvl.raw mask:={cdir}/g{j}_mask.raw\n")
    with open(f"{args.out}/calib_dec{sfx}.txt", "w") as f:
        for j, z in enumerate(dec_in):
            z.numpy().astype(np.float32).tofile(f"{cdir}/d{j}_zhat.raw")
            f.write(f"zhat:={cdir}/d{j}_zhat.raw\n")

    np.savez(f"{args.out}/cpu_head.npz", **w)
    print(f"wrote {args.out}: gen{sfx}.onnx, dec{sfx}.onnx, {len(calib_gen)} gen + {len(dec_in)} dec "
          "calibration inputs, cpu_head.npz")


if __name__ == "__main__":
    main()
