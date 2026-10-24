"""Where do the milliseconds per story go? Times each stage of the NPU pipeline (warm, averaged).

    python3 profile_story.py -n 48                              # original: qnn_serve pipe, 1 story per call
    python3 profile_story.py -n 48 --engine shm --batch 1       # shared buffers
    python3 profile_story.py -n 48 --engine shm --batch 4       # shared buffers, 4 stories per call
    python3 profile_story.py -n 48 --engine shm --batch 4 --lanes 2   # + 2 batches in flight (CPU/NPU overlap)

With --lanes > 1 the stages run at the same time in different threads, so the stage times add up
to more than the wall time per story; the gap is the time saved by overlapping.
"""
import argparse
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_ncm_npu import Story, generate, open_npu  # noqa: E402  (sets numpy threads)

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=48, help="stories to time (rounded up to whole batches)")
    ap.add_argument("--bits", default="a16w8mix")
    ap.add_argument("--engine", choices=["pipe", "shm"], default="pipe")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lanes", type=int, default=1)
    args = ap.parse_args()
    w = dict(np.load(os.path.join(HERE, "tok_weights.npz")))
    hw = dict(np.load(os.path.join(HERE, "cpu_head.npz")))
    npu = open_npu(args.engine, args.bits, args.batch, args.lanes)
    lock = threading.Lock()
    stories = [Story(npu, w, hw, i, lock) for i in range(args.lanes)]
    B = stories[0].B
    rng = np.random.default_rng(0)

    generate(stories, 8 * args.lanes, rng)  # warm-up
    T = {}
    t0 = time.perf_counter()
    done = len(generate(stories, args.n, rng, T=T))
    total = 1000 * (time.perf_counter() - t0) / done
    npu.close()
    print(f"{args.engine}, batch {B}, lanes {args.lanes}, {args.bits}: {total:.1f} ms per story (warm, {done} stories)")
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
        print(f"  {1000 * v / done:6.1f} ms  {k}")
    rest = total - 1000 * sum(T.values()) / done
    print(f"  {rest:6.1f} ms  " + ("other (python loop)" if args.lanes == 1 else "saved by overlapping lanes"))


if __name__ == "__main__":
    main()
