"""Watch NCM write a story coarse-to-fine on the Hexagon NPU.

After each of the 12 levels, the story-so-far is decoded (one extra NPU decoder call) and shown,
so you can see it go from a rough sketch to the final text. The previews are only for the demo:
the reported generation time counts just the 12 generator calls + 1 final decode.

    python3 demo_ncm.py --seed 3                    # live in the terminal
    python3 demo_ncm.py --seed 3 --record demo.jsonl  # also save events for render_demo.py
"""
import argparse
import json
import os
import shutil
import sys
import textwrap
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_ncm_npu import NPU, contrib, head, mask, softmax_sample  # noqa: E402  (sets numpy threads)

import numpy as np  # noqa: E402

DIM, BOLD, CYAN, RESET, CLEAR = "\033[2m", "\033[1m", "\033[36m", "\033[0m", "\033[2J\033[H"


def decode(npu, bpe, zhat):
    toks = npu.run(1, {"zhat": zhat})[0].argmax(-1).tolist()
    toks = toks[: toks.index(1)] if 1 in toks else toks
    return bpe.decode([t for t in toks if t > 1]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--bits", default="a16w8mix")
    ap.add_argument("--pace", type=float, default=0.8, help="seconds to show each level (display only)")
    ap.add_argument("--record", default=None, help="write events as JSON lines")
    ap.add_argument("--quiet", action="store_true", help="no live display (for recording only)")
    args = ap.parse_args()

    from tokenizers import ByteLevelBPETokenizer
    w = dict(np.load(os.path.join(HERE, "tok_weights.npz")))
    hw = dict(np.load(os.path.join(HERE, "cpu_head.npz")))
    bpe = ByteLevelBPETokenizer(os.path.join(HERE, "bpe/vocab.json"), os.path.join(HERE, "bpe/merges.txt"))
    cb, levels = w["codebook"], w["levels"].tolist()
    L, d = levels[-1], cb.shape[2]
    masks = [mask(n, L) for n in levels]
    npu = NPU([f"gen_{args.bits}.bin", f"dec_{args.bits}.bin"])
    rng = np.random.default_rng(args.seed)
    width = min(shutil.get_terminal_size((100, 30)).columns, 100) - 2
    events = []

    zhat = np.zeros((L, d), np.float32)
    gen_ms = 0.0
    for k, n in enumerate(levels):
        pad = np.zeros((L, d), np.float32)
        pad[:n] = w[f"D{k}"] @ zhat
        t = time.perf_counter()
        h = npu.run(0, {"zin": pad, "lvl": hw["lvl"][k][None], "mask": masks[k]})[0][:n]
        idx = softmax_sample(head(hw, h, k), rng, 1.0, 50)
        zhat = zhat + contrib(w, k, cb[k][idx][None])[0].astype(np.float32)
        step_ms = 1000 * (time.perf_counter() - t)
        gen_ms += step_ms
        text = decode(npu, bpe, zhat)  # preview only, not counted in gen_ms
        events.append({"level": k + 1, "codes": n, "step_ms": round(step_ms, 1), "text": text})
        if not args.quiet:
            bar = "█" * (k + 1) + "░" * (len(levels) - k - 1)
            print(CLEAR + f"{BOLD}NCM on the Hexagon NPU{RESET}  {DIM}Radxa Q6A · QCS6490 · coarse-to-fine{RESET}\n")
            print(f"{CYAN}level {k + 1:2d}/12  {bar}  {n:3d} codes{RESET}  {DIM}step {step_ms:.1f} ms{RESET}\n")
            print("\n".join(textwrap.wrap(text, width)) or DIM + "(nothing readable yet)" + RESET)
            sys.stdout.flush()
            time.sleep(args.pace)
    t = time.perf_counter()
    final = decode(npu, bpe, zhat)
    gen_ms += 1000 * (time.perf_counter() - t)
    npu.close()
    summary = {"seed": args.seed, "bits": args.bits, "gen_ms": round(gen_ms, 1), "final": final}
    if not args.quiet:
        print(f"\n{BOLD}Done: 12 generator steps + 1 decode = {gen_ms:.0f} ms on the NPU{RESET}"
              f"  {DIM}(display slowed down to {args.pace} s per level){RESET}")
    if args.record:
        with open(args.record, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
            f.write(json.dumps({"summary": summary}) + "\n")


if __name__ == "__main__":
    main()
