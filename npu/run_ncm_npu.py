"""NCM on the Q6A Hexagon NPU. The transformer (gen) and the decoder (dec) run on the NPU through
qnn_serve; the small code head and the tokenizer maths run on the CPU in numpy.

    python3 run_ncm_npu.py -n 4                # generate
    python3 run_ncm_npu.py --check 8           # NPU vs fp32 CPU (needs ../ncm_v2/ncm_level.onnx)
    python3 run_ncm_npu.py -n 256 --out s.txt  # many stories to a file
"""
import argparse
import os
import subprocess
import sys
import time

# numpy's BLAS starts one spinning thread per core (8 on the Q6A), which kept every core ~96% busy
# and made this slower. 4 threads measured best on the board: 141 vs 225 ms per story. Must be set
# before numpy is imported. Override with OPENBLAS_NUM_THREADS=... if needed.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_ncm import contrib, softmax_sample  # noqa: E402  (same maths as the CPU runner)

NEG = -60.0  # must match export_npu.py


class NPU:
    def __init__(self, bins):
        q = os.path.join(HERE, "qnn")
        env = dict(os.environ, LD_LIBRARY_PATH=q + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
                   ADSP_LIBRARY_PATH=f"{q};/usr/lib/rfsa/adsp;/dsp")
        self.p = subprocess.Popen([os.path.join(HERE, "qnn_serve"), f"{q}/libQnnHtp.so", f"{q}/libQnnSystem.so"]
                                  + [os.path.join(HERE, b) for b in bins],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        self.t = {}  # (graph, io) -> list of (name, dtype, scale, offset, shape)
        while True:
            line = self.p.stdout.readline().decode()
            if not line:
                raise RuntimeError("qnn_serve exited during load")
            if line.strip() == "READY":
                break
            if not line.startswith("T "):  # the QNN library prints its own log lines to stdout
                print(line.rstrip(), file=sys.stderr)
                continue
            _, g, io, _, name, bits, s, o, rank, *dims = line.split()
            dt = {8: np.uint8, 16: np.uint16, 32: np.float32}[int(bits)]
            self.t.setdefault((int(g), io), []).append((name, dt, float(s), int(o), tuple(map(int, dims))))

    def run(self, g, feeds):
        buf = [bytes([g])]
        for name, dt, s, o, shape in self.t[(g, "in")]:
            x = np.asarray(feeds[name], np.float32).reshape(shape)
            if dt is not np.float32:
                info = np.iinfo(dt)
                x = np.clip(np.rint(x / s) - o, info.min, info.max)
            buf.append(x.astype(dt).tobytes())
        self.p.stdin.write(b"".join(buf))
        self.p.stdin.flush()
        outs = []
        for name, dt, s, o, shape in self.t[(g, "out")]:
            n = int(np.prod(shape)) * np.dtype(dt).itemsize
            raw = np.frombuffer(self.p.stdout.read(n), dt).reshape(shape)
            outs.append(raw if dt is np.float32 else (raw.astype(np.float32) + o) * s)
        return outs

    def close(self):
        self.p.stdin.write(b"\xff")
        self.p.stdin.flush()
        self.p.wait()


def head(hw, h, k):
    z = h @ hw["head_proj"][k]
    return (2 * z @ hw["cb"][k].T - hw["cb_sq"][k]) / hw["d_scale"][k] * np.exp(hw["log_inv_T"][k])


def mask(n, L):
    m = np.full((L + 1, L + 1), NEG, np.float32)
    m[0, 0] = 0
    m[1:, : n + 1] = 0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("--bits", default="a8w8")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--check", type=int, default=0, help="compare N stories' level outputs against fp32 CPU")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    w = dict(np.load(os.path.join(HERE, "tok_weights.npz")))
    hw = dict(np.load(os.path.join(HERE, "cpu_head.npz")))
    cb, levels = w["codebook"], w["levels"].tolist()
    L, d = levels[-1], cb.shape[2]
    masks = [mask(n, L) for n in levels]
    t0 = time.perf_counter()
    npu = NPU([f"gen_{args.bits}.bin", f"dec_{args.bits}.bin"])
    print(f"loaded NPU graphs in {time.perf_counter() - t0:.2f} s", file=sys.stderr)
    rng = np.random.default_rng(args.seed)

    if args.check:
        import onnxruntime as ort
        ref = ort.InferenceSession(os.path.join(HERE, "../ncm_v2/ncm_level.onnx"), providers=["CPUExecutionProvider"])
        dref = ort.InferenceSession(os.path.join(HERE, "../ncm_v2/decoder.onnx"), providers=["CPUExecutionProvider"])
        offs = w["offs"].tolist()
        agree = np.zeros(len(levels))
        tok_agree = 0.0
        for _ in range(args.check):
            zhat = np.zeros((L, d), np.float32)
            for k, n in enumerate(levels):
                zin = w[f"D{k}"] @ zhat
                m = np.ones((n + 1, n + 1), bool)
                m[0, 1:] = False
                lr = ref.run(None, {"zin": zin[None].astype(np.float32), "pos_ids": np.arange(offs[k], offs[k] + n),
                                    "level": np.array([k]), "mask": m})[0][0]
                pad = np.zeros((L, d), np.float32)
                pad[:n] = zin
                h = npu.run(0, {"zin": pad, "lvl": hw["lvl"][k][None], "mask": masks[k]})[0][:n]
                agree[k] += (head(hw, h, k).argmax(-1) == lr.argmax(-1)).mean()
                idx = softmax_sample(lr, rng, args.temperature, args.top_k)  # follow the fp32 path
                zhat = zhat + contrib(w, k, cb[k][idx][None])[0].astype(np.float32)
            a = npu.run(1, {"zhat": zhat})[0].argmax(-1)
            b = dref.run(None, {"zhat": zhat[None]})[0][0].argmax(-1)
            tok_agree += (a == b).mean()
        print("NPU vs fp32 top-1 agreement per level (%):", " ".join(f"{100 * x / args.check:.0f}" for x in agree))
        print(f"decoder token agreement: {100 * tok_agree / args.check:.2f}%")
        npu.close()
        return

    from tokenizers import ByteLevelBPETokenizer
    bpe = ByteLevelBPETokenizer(os.path.join(HERE, "bpe/vocab.json"), os.path.join(HERE, "bpe/merges.txt"))
    texts, t_npu, t_all = [], 0.0, time.perf_counter()
    for _ in range(args.n):
        zhat = np.zeros((L, d), np.float32)
        for k, n in enumerate(levels):
            pad = np.zeros((L, d), np.float32)
            pad[:n] = w[f"D{k}"] @ zhat
            t = time.perf_counter()
            h = npu.run(0, {"zin": pad, "lvl": hw["lvl"][k][None], "mask": masks[k]})[0][:n]
            t_npu += time.perf_counter() - t
            idx = softmax_sample(head(hw, h, k), rng, args.temperature, args.top_k)
            zhat = zhat + contrib(w, k, cb[k][idx][None])[0].astype(np.float32)
        t = time.perf_counter()
        toks = npu.run(1, {"zhat": zhat})[0].argmax(-1).tolist()
        t_npu += time.perf_counter() - t
        toks = toks[: toks.index(1)] if 1 in toks else toks
        texts.append(bpe.decode([x for x in toks if x > 1]).strip())
    t_all = time.perf_counter() - t_all
    npu.close()
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n\n---\n\n".join(texts))
    else:
        for x in texts:
            print("-" * 60 + "\n" + x)
    print("-" * 60 + f"\n{args.n} stor{'y' if args.n == 1 else 'ies'}: {1000 * t_all / args.n:.0f} ms per story "
          f"({1000 * t_npu / args.n:.0f} ms of it in NPU calls: 12 generator + 1 decoder)", file=sys.stderr)


if __name__ == "__main__":
    main()
