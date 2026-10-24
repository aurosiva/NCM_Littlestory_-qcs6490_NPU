"""Export NCM (v2) for the Hexagon NPU (QNN HTP, QCS6490 / v68).

NPU graphs need fixed shapes, so:
  gen.onnx   zin [128,d] (level input zero-padded to 128), lvl [1,W] (level embedding),
             mask [129,129] (additive; hides padding)  ->  h [128,W] (final-norm hidden states)
  dec.onnx   zhat [128,d]  ->  logits [128,vocab]
All inputs/outputs are rank 2 so the converter never reorders axes.
The codebook-distance head (h -> code logits) stays on the CPU in fp32: it is a difference of
large numbers that does not survive 8/16-bit rounding (see README, INT8 note).

Also writes calibration inputs recorded from real generation, and cpu_head.npz.

    .venv/bin/python npu/export_npu.py --ckpt ckpt_v2 --out npu/build
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
    def __init__(self, g, L):
        super().__init__()
        self.g = g
        cos, sin = g._rope(torch.cat([torch.zeros(1), torch.arange(L).float()]))  # prefix at 0, then local 0..L-1
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, zin, lvl, mask):
        g = self.g
        x = g.in_proj(zin[None]) + lvl[None]
        x = torch.cat([g.prefix, x], 1)
        for b in g.blocks:
            x = block_fwd(b, x, mask[None, None], (self.cos, self.sin))
        return g.norm(x)[0, 1:]


class DecNPU(nn.Module):
    def __init__(self, tok):
        super().__init__()
        self.t = tok

    def forward(self, zhat):
        t = self.t
        h = zhat[None] + t.dec_pos
        for b in t.dec:
            h = block_fwd(b, h)
        return t.head(t.dec_norm(h))[0]


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

    # ---- ONNX (legacy exporter, fully static)
    ex = dict(export_params=True, opset_version=17, dynamo=False, do_constant_folding=True)
    zi, lv, mk = (torch.from_numpy(a) for a in calib_gen[0])
    torch.onnx.export(gnpu, (zi, lv, mk), f"{args.out}/gen.onnx", input_names=["zin", "lvl", "mask"],
                      output_names=["h"], **ex)
    torch.onnx.export(dnpu, (zhat[0],), f"{args.out}/dec.onnx", input_names=["zhat"], output_names=["logits"], **ex)

    # ---- calibration raws (float32, C order) + input lists for qairt-quantizer
    cdir = os.path.abspath(f"{args.out}/calib")
    os.makedirs(cdir, exist_ok=True)
    with open(f"{args.out}/calib_gen.txt", "w") as f:
        for j, (a, b_, c) in enumerate(calib_gen):
            for nm, arr in (("zin", a), ("lvl", b_), ("mask", c)):
                arr.astype(np.float32).tofile(f"{cdir}/g{j}_{nm}.raw")
            f.write(f"zin:={cdir}/g{j}_zin.raw lvl:={cdir}/g{j}_lvl.raw mask:={cdir}/g{j}_mask.raw\n")
    with open(f"{args.out}/calib_dec.txt", "w") as f:
        for j in range(B):
            zhat[j].numpy().astype(np.float32).tofile(f"{cdir}/d{j}_zhat.raw")
            f.write(f"zhat:={cdir}/d{j}_zhat.raw\n")

    np.savez(f"{args.out}/cpu_head.npz", **w)
    print(f"wrote {args.out}: gen.onnx, dec.onnx, {len(calib_gen)} gen + {B} dec calibration inputs, cpu_head.npz")


if __name__ == "__main__":
    main()
