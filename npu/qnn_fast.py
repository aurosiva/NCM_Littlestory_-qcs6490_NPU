"""Shared-buffer NPU driver: loads libncm_qnn.so in-process (no qnn_serve pipe).

Each graph input/output is a numpy array that views the same memory QNN gives to the NPU, so
inputs are quantized straight into it and outputs are read from it with no copies in between.
Same run(g, feeds) interface as run_ncm_npu.NPU, plus lower-level calls for the fast path:
    npu.quant(g, name, x)   -> quantized copy of x (cache constant inputs with this)
    npu.put(g, name, x)     write a float array (quantized in place) or an already quantized one
    npu.execute(g)
    npu.out(g, i)           raw output view (valid until the next execute of g)
"""
import ctypes
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# QNN dtype "kind" nibble: 0 int, 1 uint, 2 float, 3 signed fixed point, 4 unsigned fixed point
_DT = {(0, 32): np.int32, (1, 32): np.uint32, (2, 32): np.float32, (3, 8): np.int8, (3, 16): np.int16,
       (4, 8): np.uint8, (4, 16): np.uint16, (1, 8): np.uint8, (0, 8): np.int8}


class FastNPU:
    def __init__(self, bins, qdir=None):
        qdir = qdir or os.path.join(HERE, "qnn")
        os.environ["ADSP_LIBRARY_PATH"] = f"{qdir};/usr/lib/rfsa/adsp;/dsp"  # read by the DSP loader at open
        # libQnnHtp opens its DSP stub by name; preload it so no LD_LIBRARY_PATH is needed
        for so in ("libQnnSystem.so", "libQnnHtpV68Stub.so"):
            ctypes.CDLL(os.path.join(qdir, so), mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL(os.path.join(HERE, "libncm_qnn.so"))
        lib.ncm_buffer.restype = ctypes.c_void_p
        lib.ncm_buffer.argtypes = [ctypes.c_int] * 3
        arr = (ctypes.c_char_p * len(bins))(*[os.path.join(HERE, b).encode() for b in bins])
        if lib.ncm_open(f"{qdir}/libQnnHtp.so".encode(), f"{qdir}/libQnnSystem.so".encode(), len(bins), arr):
            raise RuntimeError("ncm_open failed (see stderr)")
        self.lib = lib
        self.t, self.v, self.idx = {}, {}, {}
        self.shared = True
        buf = ctypes.create_string_buffer(512)
        for g in range(len(bins)):
            for io, key in ((0, "in"), (1, "out")):
                for i in range(lib.ncm_count(g, io)):
                    lib.ncm_info(g, io, i, buf, 512)
                    name, bits, kind, s, o, rank, *dims = buf.value.decode().split()
                    dt, shape = _DT[(int(kind), int(bits))], tuple(map(int, dims))
                    n = int(np.prod(shape)) * np.dtype(dt).itemsize
                    raw = (ctypes.c_char * n).from_address(lib.ncm_buffer(g, io, i))
                    view = np.frombuffer(raw, dt).reshape(shape)
                    self.t.setdefault((g, key), []).append((name, dt, float(s), int(o), shape))
                    self.v.setdefault((g, key), []).append(view)
                    self.idx[(g, key, name)] = i
                    self.shared &= bool(lib.ncm_shared(g, io, i))

    def _spec(self, g, name):
        i = self.idx[(g, "in", name)]
        return i, self.t[(g, "in")][i]

    def quant(self, g, name, x):
        i, (_, dt, s, o, shape) = self._spec(g, name)
        x = np.asarray(x, np.float32).reshape(shape)
        if dt in (np.float32,):
            return x.copy()
        info = np.iinfo(dt)
        return np.clip(np.rint(x * (1.0 / s)) - o, info.min, info.max).astype(dt)

    def put(self, g, name, x):
        i, (_, dt, s, o, shape) = self._spec(g, name)
        dst = self.v[(g, "in")][i]
        x = np.asarray(x)
        if x.dtype == dt:
            np.copyto(dst, x.reshape(shape))
        else:
            dst[...] = self.quant(g, name, x)

    def execute(self, g):
        if self.lib.ncm_execute(g):
            raise RuntimeError(f"graph {g} execute failed")

    def out(self, g, i=0):
        return self.v[(g, "out")][i]

    def deq(self, g, raw, i=0):
        _, dt, s, o, _ = self.t[(g, "out")][i]
        return raw if dt is np.float32 else (raw.astype(np.float32) + o) * s

    def run(self, g, feeds):
        for name, x in feeds.items():
            self.put(g, name, x)
        self.execute(g)
        return [self.deq(g, self.out(g, i), i) for i in range(len(self.t[(g, "out")]))]

    def close(self):
        if self.lib:
            self.lib.ncm_close()
            self.lib = None
