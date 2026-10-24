"""Stage 1: train the multiscale residual VQ tokenizer. -> ckpt/tokenizer.pt"""
import argparse
import math
import os
import time

import numpy as np
import torch

from config import V2_TOK, TokCfg, TrainCfg
from model import MultiscaleTokenizer, set_seed


def load_bin(path, L):
    return np.memmap(path, dtype=np.uint16, mode="r").reshape(-1, L)


def lr_at(step, total, peak, warmup, floor_ratio):
    if step < warmup:
        return peak * step / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (floor_ratio + (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def evaluate(tok, val, dev, n=2048, chunk=256):
    """Token reconstruction accuracy (non-pad) using the first k levels."""
    tok.eval()
    c = tok.cfg
    check = (1, c.K // 2, c.K - 1)
    hit, tot = {k: 0 for k in check}, 0
    for i in range(0, n, chunk):
        x = torch.from_numpy(val[i:i + chunk].astype(np.int64)).to(dev)
        z = tok.encode_latent(x)
        zhat = torch.zeros_like(z)
        nonpad = x != 0
        tot += nonpad.sum().item()
        for k in range(c.K):
            idx = tok.nearest(k, tok.qdown(k, z - zhat))
            zhat = zhat + tok.contrib(k, tok.codebook[k][idx])
            if k in check:
                hit[k] += ((tok.decode(zhat).argmax(-1) == x) & nonpad).sum().item()
    out = {f"acc@{k + 1}": hit[k] / tot for k in check}
    used = [(tok.ema_count[k] > 0.5).float().mean().item() for k in range(c.K)]
    out["code_use"] = sum(used) / len(used)
    tok.train()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=TrainCfg.tok_steps)
    ap.add_argument("--batch", type=int, default=TrainCfg.tok_batch)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--preset", choices=["base", "v2"], default="base")
    args = ap.parse_args()
    set_seed(0)
    tc, tr = TokCfg(**(V2_TOK if args.preset == "v2" else {})), TrainCfg()
    print("tokenizer config:", tc)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = load_bin(f"{args.data}/train.bin", tc.L), load_bin(f"{args.data}/val.bin", tc.L)

    tok = MultiscaleTokenizer(tc).to(dev)
    print(f"tokenizer params: {sum(p.numel() for p in tok.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(tok.parameters(), lr=tr.tok_lr, betas=(0.9, 0.95), weight_decay=0.01)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    for step in range(args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, tr.tok_lr, 1000, tr.lr_floor_ratio)
        idx = np.random.randint(0, len(train), args.batch)
        x = torch.from_numpy(train[np.sort(idx)].astype(np.int64)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            out = tok(x)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"step {step} loss {out['loss'].item():.3f} rec {out['rec'].item():.3f} "
                  f"commit {out['commit'].item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        if step % 1000 == 0:
            print("  eval", {k: round(v, 4) for k, v in evaluate(tok, val, dev).items()}, flush=True)
            torch.save({"cfg": tc.__dict__, "model": tok.state_dict()}, f"{args.out}/tokenizer.pt")
    print("saved", f"{args.out}/tokenizer.pt")


if __name__ == "__main__":
    main()
