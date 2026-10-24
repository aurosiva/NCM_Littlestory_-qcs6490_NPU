"""Fair CPU timing (copy into the edge folder on the board): same maths as run_ncm.py, but warm (sessions loaded once,
one warm-up story) and averaged over many stories, like run_ncm_npu.py.

    OPENBLAS_NUM_THREADS=4 python3 bench_cpu.py --ort_threads 4 --batch 1 -n 60
"""
import argparse
import os
import time

import numpy as np
import onnxruntime as ort

try:  # on the board the runner is copied as run_ncm.py
    from run_ncm import contrib, softmax_sample
except ImportError:
    from edge_run_ncm import contrib, softmax_sample

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=60, help="stories to time")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ort_threads", type=int, default=4)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.ort_threads
    sfx = ".onnx" if args.fp32 else ".int8.onnx"
    gen = ort.InferenceSession(os.path.join(HERE, "ncm_level" + sfx), so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(os.path.join(HERE, "decoder" + sfx), so, providers=["CPUExecutionProvider"])
    w = dict(np.load(os.path.join(HERE, "tok_weights.npz")))
    cb, levels, offs = w["codebook"], w["levels"].tolist(), w["offs"].tolist()
    L, d = levels[-1], cb.shape[2]
    rng = np.random.default_rng(0)

    def story_batch(B):
        zhat = np.zeros((B, L, d), np.float32)
        for k, n in enumerate(levels):
            m = np.ones((n + 1, n + 1), bool)
            m[0, 1:] = False
            lg = gen.run(None, {"zin": (w[f"D{k}"] @ zhat).astype(np.float32),
                                "pos_ids": np.arange(offs[k], offs[k] + n, dtype=np.int64),
                                "level": np.array([k], np.int64), "mask": m})[0]
            zhat += contrib(w, k, cb[k][softmax_sample(lg, rng, 1.0, 50)]).astype(np.float32)
        return dec.run(None, {"zhat": zhat})[0].argmax(-1)

    story_batch(args.batch)  # warm-up
    t = time.perf_counter()
    done = 0
    while done < args.n:
        story_batch(args.batch)
        done += args.batch
    dt = time.perf_counter() - t
    print(f"cpu {'fp32' if args.fp32 else 'int8'} batch={args.batch} ort_threads={args.ort_threads} "
          f"blas={os.environ.get('OPENBLAS_NUM_THREADS', 'default')}: {1000 * dt / done:.0f} ms per story")


if __name__ == "__main__":
    main()
