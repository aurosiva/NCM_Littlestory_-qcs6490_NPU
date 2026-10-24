"""Export the trained NCM for the Radxa Q6A.

Writes edge/:
  ncm_level.onnx   generator, one level per call  (zin, pos_ids, level, mask) -> code logits
  decoder.onnx     tokenizer decoder              zhat [1,L,d] -> token logits
  tok_weights.npz  codebooks, Down/Up matrices, R_k conv weights (applied in numpy)
  bpe/             text BPE files
  *.int8.onnx      dynamically INT8-quantized copies (with --int8)
Then checks that the numpy+ONNX pipeline matches PyTorch.
"""
import argparse
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn

from config import GenCfg, TokCfg
from model import NCMGenerator
from train_generator import load_tokenizer


class LevelStep(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, zin, pos_ids, level, mask):
        return self.g.step(zin, pos_ids, level, mask)


class Decoder(nn.Module):
    def __init__(self, tok):
        super().__init__()
        self.tok = tok

    def forward(self, zhat):
        return self.tok.decode(zhat)


def export(model, args, path, names, outs, dyn):
    kw = dict(input_names=names, output_names=outs, dynamic_axes=dyn, opset_version=17)
    try:
        torch.onnx.export(model, args, path, dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(model, args, path, **kw)
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="edge")
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok = load_tokenizer(f"{args.ckpt}/tokenizer.pt", "cpu")
    tc = tok.cfg
    ck = torch.load(f"{args.ckpt}/generator.pt", map_location="cpu")
    gen = NCMGenerator(GenCfg(**ck["cfg"]), tc)
    gen.load_state_dict(ck["model"])
    gen.eval()

    # generator: one level per call; dynamic length so CPU runs only l_k positions
    n = tc.levels[3]
    mask = torch.ones(n + 1, n + 1, dtype=torch.bool)
    mask[0, 1:] = False
    export(LevelStep(gen).eval(), (torch.randn(1, n, tc.d), torch.arange(n) + gen.offs[3], torch.tensor([3]), mask),
           f"{args.out}/ncm_level.onnx", ["zin", "pos_ids", "level", "mask"], ["logits"],
           {"zin": {0: "B", 1: "n"}, "pos_ids": {0: "n"}, "mask": {0: "n1", 1: "n1"}, "logits": {0: "B", 1: "n"}})
    export(Decoder(tok).eval(), (torch.randn(1, tc.L, tc.d),), f"{args.out}/decoder.onnx", ["zhat"], ["logits"],
           {"zhat": {0: "B"}, "logits": {0: "B"}})

    w = {"codebook": tok.codebook.numpy(), "levels": np.array(tc.levels), "offs": np.array(gen.offs),
         "phi_w": np.stack([p.weight.detach().numpy() for p in tok.phi]),
         "phi_b": np.stack([p.bias.detach().numpy() for p in tok.phi])}
    for k in range(tc.K):
        w[f"D{k}"], w[f"U{k}"] = tok.D(k).numpy(), tok.U(k).numpy()
    if tc.l1_learned:
        w["up0_n"] = np.array(len(tok.up_l1))
        for i, m in enumerate(tok.up_l1):
            w[f"up0_w{i}"], w[f"up0_b{i}"] = m.weight.detach().numpy(), m.bias.detach().numpy()
    np.savez(f"{args.out}/tok_weights.npz", **w)
    shutil.copytree(f"{args.data}/bpe", f"{args.out}/bpe", dirs_exist_ok=True)
    shutil.copy(os.path.join(os.path.dirname(__file__), "edge_run_ncm.py"), f"{args.out}/run_ncm.py")

    if args.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        for name in ("ncm_level", "decoder"):
            # MatMul weights only. By default quantize_dynamic also turns Gather tables (codebook,
            # head projection) into int8; the v2 tied head compares against the codebook and that
            # rounding wipes out every prediction past level 1.
            quantize_dynamic(f"{args.out}/{name}.onnx", f"{args.out}/{name}.int8.onnx",
                             weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
            print("wrote", f"{args.out}/{name}.int8.onnx",
                  f"{os.path.getsize(f'{args.out}/{name}.int8.onnx') / 1e6:.1f} MB")

    # parity check: follow PyTorch's greedy path and compare every ONNX/numpy piece on it
    import onnxruntime as ort
    sys.path.insert(0, args.out)
    from run_ncm import contrib as contrib_np
    g_s = ort.InferenceSession(f"{args.out}/ncm_level.onnx", providers=["CPUExecutionProvider"])
    d_s = ort.InferenceSession(f"{args.out}/decoder.onnx", providers=["CPUExecutionProvider"])
    err = 0.0
    with torch.no_grad():
        zhat = torch.zeros(2, tc.L, tc.d)
        for k, n in enumerate(tc.levels):
            zin = tok.D(k) @ zhat
            ref = gen.forward_level(zin, k)
            m = np.ones((n + 1, n + 1), bool)
            m[0, 1:] = False
            o = g_s.run(None, {"zin": zin.numpy(), "pos_ids": np.arange(gen.offs[k], gen.offs[k] + n),
                               "level": np.array([k]), "mask": m})[0]
            err = max(err, float(np.abs(o - ref.numpy()).max()))
            q = tok.codebook[k][ref.argmax(-1)]
            c = tok.contrib(k, q)
            err = max(err, float(np.abs(contrib_np(np.load(f"{args.out}/tok_weights.npz"), k, q.numpy()) - c.numpy()).max()))
            zhat = zhat + c
        err = max(err, float(np.abs(d_s.run(None, {"zhat": zhat.numpy()})[0] - tok.decode(zhat).numpy()).max()))
    print(f"parity check: max abs diff {err:.2e}", "OK" if err < 1e-3 else "MISMATCH")

if __name__ == "__main__":
    main()
