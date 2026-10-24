"""NCM on the Q6A Hexagon NPU. The transformer (gen) and the decoder (dec) run on the NPU through
qnn_serve; the small code head and the tokenizer maths run on the CPU in numpy.

    python3 run_ncm_npu.py -n 4                # generate
    python3 run_ncm_npu.py --check 8           # NPU vs fp32 CPU (needs ../ncm_v2/ncm_level.onnx)
    python3 run_ncm_npu.py -n 256 --out s.txt  # many stories to a file
    python3 run_ncm_npu.py -n 48 --engine shm --batch 4   # shared buffers, 4 stories per NPU call

Engines: "pipe" talks to the qnn_serve process over stdin/stdout (the original version). "shm" loads
libncm_qnn.so in-process and shares the input/output buffers with the NPU (see qnn_fast.py).
--batch B uses the gen_bB/dec_bB binaries from export_npu.py --batch B (decoder returns tokens).
"""
import argparse
import os
import subprocess
import sys
import threading
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

    # same low-level API as qnn_fast.FastNPU
    def quant(self, g, name, x):
        _, dt, s, o, shape = next(t for t in self.t[(g, "in")] if t[0] == name)
        if dt is np.float32:
            return np.asarray(x, np.float32).reshape(shape)
        info = np.iinfo(dt)
        return np.clip(np.rint(np.asarray(x, np.float32).reshape(shape) / s) - o, info.min, info.max).astype(dt)

    def put(self, g, name, x):
        dt = next(t[1] for t in self.t[(g, "in")] if t[0] == name)
        self.pending = getattr(self, "pending", {})
        self.pending[(g, name)] = x if np.asarray(x).dtype == dt else self.quant(g, name, x)

    def execute(self, g):
        self.p.stdin.write(bytes([g]) + b"".join(np.ascontiguousarray(self.pending[(g, t[0])]).tobytes()
                                                 for t in self.t[(g, "in")]))
        self.p.stdin.flush()
        self.last = getattr(self, "last", {})
        self.last[g] = [np.frombuffer(self.p.stdout.read(int(np.prod(sh)) * np.dtype(dt).itemsize), dt).reshape(sh)
                        for _, dt, _, _, sh in self.t[(g, "out")]]

    def out(self, g, i=0):
        return self.last[g][i]

    def deq(self, g, raw, i=0):
        _, dt, s, o, _ = self.t[(g, "out")][i]
        return raw if dt is np.float32 else (raw.astype(np.float32) + o) * s

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


def open_npu(engine, bits, batch=None, lanes=1):
    """lanes > 1 loads the graphs that many times (graphs 2i, 2i+1), one set of buffers per lane."""
    sfx = f"_b{batch}" if batch else ""
    bins = [f"gen{sfx}_{bits}.bin", f"dec{sfx}_{bits}.bin"] * lanes
    if lanes > 1 and engine != "shm":
        raise SystemExit("--lanes needs --engine shm")
    if engine == "shm":
        from qnn_fast import FastNPU
        return FastNPU(bins)
    return NPU(bins)


def dec_tokens(npu, B, L, g=1):
    """Decoder output -> token ids [B, L]: the graph's argmax, or the argmax of the raw quantized
    logits (dequantizing is increasing in q, so the order is the same and the copy is skipped)."""
    raw = npu.out(g)
    if raw.ndim == 1:
        return raw.reshape(B, L).astype(np.int64)
    return raw.reshape(B, L, -1).argmax(-1)


class Story:
    """Constant per-level inputs, quantized once, and the batched generation loop."""

    def __init__(self, npu, w, hw, lane=0, lock=None):
        self.npu, self.w, self.hw = npu, w, hw
        self.gg, self.gd = 2 * lane, 2 * lane + 1  # this lane's generator / decoder graph
        self.lock = lock or threading.Lock()  # one NPU call at a time across lanes
        self.cb, self.levels = w["codebook"], w["levels"].tolist()
        self.L, self.d = self.levels[-1], self.cb.shape[2]
        self.B = npu.t[(0, "in")][0][4][0] // self.L  # zin is [B*L, d]
        self.lvl_q = [npu.quant(0, "lvl", hw["lvl"][k][None]) for k in range(len(self.levels))]
        self.mask_q = [npu.quant(0, "mask", mask(n, self.L)) for n in self.levels]

    def tokens(self):
        return dec_tokens(self.npu, self.B, self.L, self.gd)

    def execute(self, g):
        with self.lock:  # the GIL is released inside the C call, so other lanes keep computing
            self.npu.execute(g)

    def run(self, rng, temperature=1.0, top_k=50, T=None):
        """Generate B stories. T (optional dict) collects seconds spent per stage. With several lanes,
        "NPU call" also includes waiting for another lane's call to finish."""
        npu, w, hw, B, L, d = self.npu, self.w, self.hw, self.B, self.L, self.d
        clock = time.perf_counter
        zhat = np.zeros((B, L, d), np.float32)
        pad = np.zeros((B, L, d), np.float32)
        for k, n in enumerate(self.levels):
            t0 = clock()
            pad[:, :n] = np.matmul(w[f"D{k}"], zhat)
            npu.put(self.gg, "zin", pad.reshape(B * L, d))
            npu.put(self.gg, "lvl", self.lvl_q[k])
            npu.put(self.gg, "mask", self.mask_q[k])
            t1 = clock()
            self.execute(self.gg)
            t2 = clock()
            h = npu.deq(self.gg, npu.out(self.gg).reshape(B, L, -1)[:, :n])
            t3 = clock()
            idx = softmax_sample(head(hw, h, k), rng, temperature, top_k)
            t4 = clock()
            zhat += contrib(w, k, self.cb[k][idx]).astype(np.float32)
            t5 = clock()
            if T is not None:
                for key, dt in (("prepare + quantize inputs", t1 - t0), ("NPU call (generator)", t2 - t1),
                                ("dequantize outputs", t3 - t2), ("head + sampling", t4 - t3),
                                ("add level to latent", t5 - t4)):
                    T[key] = T.get(key, 0.0) + dt
        t0 = clock()
        npu.put(self.gd, "zhat", zhat.reshape(B * L, d))
        t1 = clock()
        self.execute(self.gd)
        t2 = clock()
        toks = self.tokens()
        if T is not None:
            for key, dt in (("prepare + quantize inputs", t1 - t0), ("NPU call (decoder)", t2 - t1),
                            ("dequantize outputs", clock() - t2)):
                T[key] = T.get(key, 0.0) + dt
        return toks


def generate(stories, n, rng, temperature=1.0, top_k=50, T=None):
    """Make at least n stories with one thread per lane. While one lane waits on the NPU, the
    others run their CPU steps (numpy and the NPU call both release the GIL)."""
    seeds = rng.integers(2 ** 63, size=len(stories))
    out, ticket = [], threading.Lock()
    todo = [-(-n // stories[0].B)]  # batches left

    times = [{} for _ in stories]  # per lane, merged into T at the end

    def lane(i):
        r = np.random.default_rng(seeds[i])
        while True:
            with ticket:
                if todo[0] <= 0:
                    return
                todo[0] -= 1
            toks = stories[i].run(r, temperature, top_k, times[i])
            with ticket:
                out.extend(toks.tolist())

    if len(stories) == 1:
        lane(0)
    else:
        th = [threading.Thread(target=lane, args=(i,)) for i in range(len(stories))]
        for t in th:
            t.start()
        for t in th:
            t.join()
    if T is not None:
        for d in times:
            for k, v in d.items():
                T[k] = T.get(k, 0.0) + v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("--bits", default="a8w8")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--check", type=int, default=0, help="compare N stories' level outputs against fp32 CPU")
    ap.add_argument("--out", default=None)
    ap.add_argument("--engine", choices=["pipe", "shm"], default="pipe")
    ap.add_argument("--batch", type=int, default=None, help="use gen_bB/dec_bB binaries (B stories per call)")
    ap.add_argument("--lanes", type=int, default=1, help="batches in flight, overlapping CPU and NPU (shm only)")
    args = ap.parse_args()

    w = dict(np.load(os.path.join(HERE, "tok_weights.npz")))
    hw = dict(np.load(os.path.join(HERE, "cpu_head.npz")))
    cb, levels = w["codebook"], w["levels"].tolist()
    L, d = levels[-1], cb.shape[2]
    masks = [mask(n, L) for n in levels]
    t0 = time.perf_counter()
    npu = open_npu(args.engine, args.bits, args.batch, args.lanes)
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
                B = npu.t[(0, "in")][0][4][0] // L  # batched binaries: same story in every slot
                h = npu.run(0, {"zin": np.tile(pad, (B, 1)), "lvl": hw["lvl"][k][None], "mask": masks[k]})[0][:n]
                agree[k] += (head(hw, h, k).argmax(-1) == lr.argmax(-1)).mean()
                idx = softmax_sample(lr, rng, args.temperature, args.top_k)  # follow the fp32 path
                zhat = zhat + contrib(w, k, cb[k][idx][None])[0].astype(np.float32)
            npu.put(1, "zhat", np.tile(zhat, (B, 1)))
            npu.execute(1)
            a = dec_tokens(npu, B, L)[0]
            b = dref.run(None, {"zhat": zhat[None]})[0][0].argmax(-1)
            tok_agree += (a == b).mean()
        print("NPU vs fp32 top-1 agreement per level (%):", " ".join(f"{100 * x / args.check:.0f}" for x in agree))
        print(f"decoder token agreement: {100 * tok_agree / args.check:.2f}%")
        npu.close()
        return

    from tokenizers import ByteLevelBPETokenizer
    bpe = ByteLevelBPETokenizer(os.path.join(HERE, "bpe/vocab.json"), os.path.join(HERE, "bpe/merges.txt"))
    lock = threading.Lock()
    stories = [Story(npu, w, hw, i, lock) for i in range(args.lanes)]
    st = stories[0]
    texts, T, t_all = [], {}, time.perf_counter()
    for toks in generate(stories, args.n, rng, args.temperature, args.top_k, T):
        toks = toks[: toks.index(1)] if 1 in toks else toks
        texts.append(bpe.decode([x for x in toks if x > 1]).strip())
    t_all = time.perf_counter() - t_all
    n_done = len(texts)  # whole batches are generated; time is per story actually made
    texts = texts[: args.n]
    t_npu = T["NPU call (generator)"] + T["NPU call (decoder)"]
    npu.close()
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n\n---\n\n".join(texts))
    else:
        for x in texts:
            print("-" * 60 + "\n" + x)
    print("-" * 60 + f"\n{n_done} stor{'y' if n_done == 1 else 'ies'} ({args.engine}, batch {st.B}, lanes {args.lanes}): "
          f"{1000 * t_all / n_done:.0f} ms per story " + (
              f"({1000 * t_npu / n_done:.0f} ms of it in NPU calls: 12 generator + 1 decoder per batch)"
              if args.lanes == 1 else "(lanes overlap CPU and NPU work, so no per-stage split)"), file=sys.stderr)


if __name__ == "__main__":
    main()
