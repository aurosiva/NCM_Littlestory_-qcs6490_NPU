"""Compare finished models the way the paper does: generative perplexity under GPT-2 Large
(lower = more natural text) and unigram entropy (lower = more repetitive), plus the share
of immediately repeated words ("the the"). Real validation stories give a reference.

    python compare.py edge edge_v2 --n 256
"""
import argparse
import math
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_run_ncm import contrib, softmax_sample  # same maths as the board runner


def generate(edge, n, seed, int8=True, bs=64, temperature=1.0, top_k=50):
    sfx = ".int8.onnx" if int8 else ".onnx"
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    gen = ort.InferenceSession(f"{edge}/ncm_level{sfx}", so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(f"{edge}/decoder{sfx}", so, providers=["CPUExecutionProvider"])
    w = dict(np.load(f"{edge}/tok_weights.npz"))
    cb, levels, offs = w["codebook"], w["levels"].tolist(), w["offs"].tolist()
    bpe = ByteLevelBPETokenizer(f"{edge}/bpe/vocab.json", f"{edge}/bpe/merges.txt")
    rng = np.random.default_rng(seed)
    texts = []
    for start in range(0, n, bs):
        B = min(bs, n - start)
        zhat = np.zeros((B, levels[-1], cb.shape[2]), np.float32)
        for k, l in enumerate(levels):
            m = np.ones((l + 1, l + 1), bool)
            m[0, 1:] = False
            lg = gen.run(None, {"zin": (w[f"D{k}"] @ zhat).astype(np.float32),
                                "pos_ids": np.arange(offs[k], offs[k] + l), "level": np.array([k]), "mask": m})[0]
            zhat += contrib(w, k, cb[k][softmax_sample(lg, rng, temperature, top_k)]).astype(np.float32)
        for row in dec.run(None, {"zhat": zhat})[0].argmax(-1).tolist():
            row = row[: row.index(1)] if 1 in row else row
            texts.append(bpe.decode([t for t in row if t > 1]).strip())
    return texts


def real_stories(n):
    bpe = ByteLevelBPETokenizer("data/bpe/vocab.json", "data/bpe/merges.txt")
    val = np.memmap("data/val.bin", dtype=np.uint16, mode="r").reshape(-1, 128)
    out = []
    for row in val[-n:].tolist():  # last n: not used by any eval during training
        row = row[: row.index(1)] if 1 in row else row
        out.append(bpe.decode([t for t in row if t > 1]).strip())
    return out


@torch.no_grad()
def score(texts, lm, tk, dev):
    nll, ntok, ents, reps = 0.0, 0, [], []
    for t in texts:
        ids = tk(t, return_tensors="pt").input_ids[:, :1024].to(dev)
        if ids.shape[1] < 2:
            continue
        ids = torch.cat([torch.tensor([[tk.eos_token_id]], device=dev), ids], 1)
        logits = lm(ids).logits[:, :-1].float()
        nll += torch.nn.functional.cross_entropy(logits[0], ids[0, 1:], reduction="sum").item()
        ntok += ids.shape[1] - 1
        _, c = np.unique(ids[0, 1:].cpu().numpy(), return_counts=True)
        p = c / c.sum()
        ents.append(float(-(p * np.log(p)).sum()))
        words = t.lower().split()
        reps.append(sum(a == b for a, b in zip(words, words[1:])) / max(1, len(words) - 1))
    return math.exp(nll / ntok), float(np.mean(ents)), float(np.mean(reps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edges", nargs="+")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--fp32", action="store_true", help="score the fp32 ONNX files instead of int8")
    ap.add_argument("--ref", default="gpt2-large")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tk = AutoTokenizer.from_pretrained(args.ref)
    lm = AutoModelForCausalLM.from_pretrained(args.ref, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()

    rows = [("real TinyStories (reference)", *score(real_stories(args.n), lm, tk, dev))]
    for e in args.edges:
        for s in [int(x) for x in args.seeds.split(",")]:
            texts = generate(e, args.n, s, int8=not args.fp32)
            os.makedirs("logs", exist_ok=True)
            with open(f"logs/samples_{os.path.basename(e.rstrip('/'))}_seed{s}.txt", "w") as f:
                f.write("\n\n---\n\n".join(texts))
            rows.append((f"{e} seed {s}", *score(texts, lm, tk, dev)))
    print(f"\n{'model':32s} {'gen PPL':>8s} {'entropy':>8s} {'repeat %':>9s}")
    for name, ppl, ent, rep in rows:
        print(f"{name:32s} {ppl:8.1f} {ent:8.2f} {100 * rep:8.2f}%")


if __name__ == "__main__":
    main()
