"""Run NCM on the Radxa Q6A (or any machine). Needs only: numpy, onnxruntime, tokenizers.

    python3 run_ncm.py -n 4                    # CPU, fp32
    python3 run_ncm.py -n 4 --int8             # CPU, int8 weights (smaller, usually faster)
    python3 run_ncm.py --qnn                   # try the Hexagon NPU via onnxruntime-qnn

Generation = K sequential steps (coarse -> fine); each step predicts every code of
that level in parallel with one small transformer pass, then one decoder pass.
"""
import argparse
import os
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))


def conv1d_k3(x, w, b):
    """Same as torch Conv1d(d, d, 3, padding=1) on x [B,L,d]."""
    xp = np.pad(x, ((0, 0), (1, 1), (0, 0)))
    L = x.shape[1]
    return b + xp[:, 0:L] @ w[:, :, 0].T + xp[:, 1:L + 1] @ w[:, :, 1].T + xp[:, 2:L + 2] @ w[:, :, 2].T


def contrib(w, k, q):
    """R_k(Up_k(q)) for code vectors q [B,l_k,d] -> [B,L,d]. Matches MultiscaleTokenizer.contrib."""
    if k == 0 and "up0_n" in w:  # v2: learned transposed-conv upsampling (kernel == stride)
        x = q
        for i in range(int(w["up0_n"])):
            W, b = w[f"up0_w{i}"], w[f"up0_b{i}"]  # W: [in, out, s]
            B, n, _ = x.shape
            x = np.einsum("bjc,cos->bjso", x, W).reshape(B, n * W.shape[2], W.shape[1]) + b
        up = x
    else:
        up = w[f"U{k}"] @ q
    return up + conv1d_k3(up, w["phi_w"][k], w["phi_b"][k])


def softmax_sample(logits, rng, temperature, top_k):
    logits = logits.astype(np.float64) / max(temperature, 1e-5)
    if top_k:
        kth = np.partition(logits, -top_k, axis=-1)[..., -top_k][..., None]
        logits = np.where(logits < kth, -np.inf, logits)
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    flat = p.reshape(-1, p.shape[-1])
    c = flat.cumsum(-1)
    r = rng.random((flat.shape[0], 1))
    return np.minimum((c < r).sum(-1), flat.shape[-1] - 1).reshape(p.shape[:-1])


def session(path, qnn):
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("NCM_THREADS", min(4, os.cpu_count())))  # Q6A: 4 big cores
    if qnn:
        prov = [("QNNExecutionProvider", {"backend_path": "libQnnHtp.so"}), "CPUExecutionProvider"]
        return ort.InferenceSession(path, so, providers=prov)
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1, help="stories to generate in parallel")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--qnn", action="store_true")
    ap.add_argument("--print_ids", action="store_true")
    args = ap.parse_args()

    sfx = ".int8.onnx" if args.int8 else ".onnx"
    gen = session(os.path.join(HERE, "ncm_level" + sfx), args.qnn)
    dec = session(os.path.join(HERE, "decoder" + sfx), args.qnn)
    w = dict(np.load(os.path.join(HERE, "tok_weights.npz")))  # load once; npz access is lazy and slow
    cb, levels, offs = w["codebook"], w["levels"].tolist(), w["offs"].tolist()
    K, V, d = cb.shape
    L = levels[-1]
    rng = np.random.default_rng(args.seed)
    B = args.n

    t0 = time.perf_counter()
    zhat = np.zeros((B, L, d), np.float32)
    for k, n in enumerate(levels):
        zin = (w[f"D{k}"] @ zhat).astype(np.float32)               # Down_k(zhat^(k-1))  [B,n,d]
        mask = np.ones((n + 1, n + 1), bool)
        mask[0, 1:] = False                                         # prefix sees only itself
        logits = gen.run(None, {"zin": zin, "pos_ids": np.arange(offs[k], offs[k] + n, dtype=np.int64),
                                "level": np.array([k], np.int64), "mask": mask})[0]
        idx = logits.argmax(-1) if args.greedy else softmax_sample(logits, rng, args.temperature, args.top_k)
        zhat += contrib(w, k, cb[k][idx]).astype(np.float32)       # R_k(Up(e_k[r_k]))
    toks = np.concatenate([dec.run(None, {"zhat": zhat[i:i + 1]})[0] for i in range(B)]).argmax(-1)
    dt = time.perf_counter() - t0

    if args.print_ids:
        print(" ".join(map(str, toks[0].tolist())))
        return
    from tokenizers import ByteLevelBPETokenizer
    bpe = ByteLevelBPETokenizer(os.path.join(HERE, "bpe/vocab.json"), os.path.join(HERE, "bpe/merges.txt"))
    for row in toks.tolist():
        row = row[: row.index(1)] if 1 in row else row             # stop at <eos>
        print("-" * 60 + "\n" + bpe.decode([t for t in row if t > 1]).strip())
    print("-" * 60 + f"\n{B} sample(s) in {dt * 1000:.0f} ms  ({K} generator steps + 1 decoder pass)")


if __name__ == "__main__":
    main()
