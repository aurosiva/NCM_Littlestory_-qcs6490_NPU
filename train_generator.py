"""Stage 2: train the NCM generator on codes from the frozen tokenizer. -> ckpt/generator.pt"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import V2_GEN, GenCfg, TokCfg, TrainCfg
from model import MultiscaleTokenizer, NCMGenerator, corrupt, generate, level_weights, set_seed
from train_tokenizer import load_bin, lr_at


def load_tokenizer(path, dev):
    ck = torch.load(path, map_location=dev)
    tok = MultiscaleTokenizer(TokCfg(**ck["cfg"])).to(dev)
    tok.load_state_dict(ck["model"])
    return tok.eval().requires_grad_(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=TrainCfg.gen_steps)
    ap.add_argument("--batch", type=int, default=TrainCfg.gen_batch)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--preset", choices=["base", "v2"], default="base")
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per step (effective batch = batch*accum)")
    args = ap.parse_args()
    set_seed(0)
    gc, tr = GenCfg(**(V2_GEN if args.preset == "v2" else {})), TrainCfg()
    print("generator config:", gc)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(f"{args.out}/tokenizer.pt", dev)
    tc = tok.cfg
    train = load_bin(f"{args.data}/train.bin", tc.L)

    gen = NCMGenerator(gc, tc, codebook=tok.codebook).to(dev)
    print(f"generator params: {sum(p.numel() for p in gen.parameters()) / 1e6:.2f}M")
    decay = [p for n, p in gen.named_parameters() if p.dim() >= 2 and n not in ("pos", "lvl", "prefix")]
    no_decay = [p for n, p in gen.named_parameters() if not (p.dim() >= 2 and n not in ("pos", "lvl", "prefix"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tr.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tr.gen_lr, betas=(0.9, 0.95))
    w = level_weights(tc.levels, gc.alpha)
    os.makedirs(args.out, exist_ok=True)

    from tokenizers import ByteLevelBPETokenizer
    bpe = ByteLevelBPETokenizer(f"{args.data}/bpe/vocab.json", f"{args.data}/bpe/merges.txt")

    t0 = time.time()
    for step in range(args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, tr.gen_lr, tr.warmup, tr.lr_floor_ratio)
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            idx = np.sort(np.random.randint(0, len(train), args.batch))
            x = torch.from_numpy(train[idx].astype(np.int64)).to(dev)
            with torch.no_grad():
                codes = tok.encode_codes(x)
                zins, _ = tok.level_inputs(corrupt(codes, tc.codebook, gc.corrupt_hi, gc.corrupt_lo))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = gen(zins)
            losses = [F.cross_entropy(lg.float().reshape(-1, tc.codebook), c.reshape(-1)) for lg, c in zip(logits, codes)]
            loss = sum(wk * l * lk for wk, l, lk in zip(w, losses, tc.levels))
            (loss / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            per = " ".join(f"{l.item():.2f}" for l in losses)
            print(f"step {step} loss {loss.item():.3f} | per-level CE: {per} ({time.time() - t0:.0f}s)", flush=True)
        if step % 2000 == 0:
            gen.eval()
            out = generate(tok, gen, n=2, temperature=1.0, top_k=50)
            for row in out.tolist():
                row = row[: row.index(1)] if 1 in row else row
                print("  >", bpe.decode([t for t in row if t > 1])[:300].replace("\n", " "))
            gen.train()
            torch.save({"cfg": gc.__dict__, "model": gen.state_dict()}, f"{args.out}/generator.pt")
    print("saved", f"{args.out}/generator.pt")


if __name__ == "__main__":
    main()
