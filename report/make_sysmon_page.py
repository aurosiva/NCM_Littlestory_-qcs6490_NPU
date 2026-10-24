"""Build report/ncm_sysmon_comparison.html from the parsed sysMon captures in logs/sysmon/run3.

Each capture folder holds the Hexagon SDK parser output (sysmon_report_*/): Summary.csv,
dcvs_metric.txt, post_processed_metrics.csv and, when the NPU ran, hvx_invokes.csv. This script
reads them and writes one self-contained page with the five captures side by side.

    python3 report/make_sysmon_page.py logs/sysmon/run3 report/ncm_sysmon_comparison.html
"""
import csv
import glob
import json
import os
import re
import sys

CAPTURES = [  # id, label, what ran, story speed normally (ms), story speed while sysMon recorded
    ("idle", "Nothing running", "Board idle: the NPU should do nothing.", None, None),
    ("cpu", "NCM on the CPU", "onnxruntime INT8 on the CPU only (control: the NPU should stay idle).", None, None),
    ("v1", "First version", "NPU, qnn_serve over a pipe, 1 story per call.", 125.6, 140),
    ("v2b1", "Shared buffers", "NPU, libncm_qnn.so with rpcmem buffers, 1 story per call.", 105.3, 139),
    ("v2b8x2", "Batch 8, 2 in flight", "NPU, shared buffers, 8 stories per call, 2 batches overlapping CPU and NPU.", 66.1, 201),
]
STORIES_PER_CALL = {"v1": 1 / 13, "v2b1": 1 / 13, "v2b8x2": 8 / 13}  # 12 generator + 1 decoder calls per batch
QNN_NPU_MS = {"v1": 58, "v2b1": 58, "v2b8x2": 60}  # NPU compute per story, qnn-net-run profiling


def ts(s):
    h, m, sec, ms = map(int, s.strip().split(":"))
    return ((h * 60 + m) * 60 + sec) * 1000 + ms


def num(x):
    try:
        return float(x)
    except ValueError:
        return None


def read_dcvs(path):
    t = open(path).read()
    prof = t[t.rindex("Profile Packet"):]
    g = lambda pat: num(re.search(pat, prof).group(1)) if re.search(pat, prof) else None
    return {
        "total_s": g(r"Total Time in sec:([\d.]+)"), "idle_s": g(r"idle time in sec:([\d.]+)"),
        "active_s": g(r"active time in sec:([\d.]+)"), "mpps": g(r"\bMPPS:([\d.]+)"),
        "hvx_mpps": g(r"HVX MPPS:([\d.]+)"), "load_mhz": g(r"QDSP6 load in MHz:([\d.]+)"),
        "axi_rd": g(r"AXI_RD_BW:([\d.]+)"), "axi_wr": g(r"AXI_WR_BW:([\d.]+)"),
        "clk_mhz": g(r"NPA_core_clk:([\d.]+)"), "util": g(r"QDSP6 Utilization:([\d.]+)"),
        "clock_votes": len(re.findall("NPA Packet", t)),
    }


def read_summary(path):
    """Processor and HVX summary rows: {section: [[metric, unit, avg, max], ...]}, plus invoke stats."""
    out, inv, sec, api = {}, [], None, None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("Metric,"):
            continue
        m = re.match(r"HVX invoke stats:API - (\d+),Invokes detected : (\d+)", line)
        if m:
            api = {"api": int(m.group(1)), "count": int(m.group(2)), "rows": []}
            inv.append(api)
            continue
        parts = line.split(",")
        if len(parts) == 1 or (len(parts) == 2 and not parts[1]):
            sec, api = parts[0].strip(), None
            continue
        if len(parts) == 4 and num(parts[2]) is not None:
            if api is not None:
                api["rows"].append([parts[0], parts[1], num(parts[2]), num(parts[3])])
            elif sec:
                out.setdefault(sec, []).append([parts[0], parts[1], num(parts[2]), num(parts[3])])
    return out, inv


def read_series(path):
    """1 ms samples while busy, ~200 ms while idle. Columns follow the parser's header."""
    rows = [r for r in csv.reader(open(path))][1:]
    rows = [r for r in rows if len(r) > 18 and r[2].strip()]
    t0 = ts(rows[0][2]) - (num(rows[0][1]) or 0)
    S = {k: [] for k in ("t", "per", "load", "hvx", "mpps", "rd", "wr")}
    clk, bus, share, tot = [], [], {}, 0.0
    for r in rows:
        # column 2 (SysClock Time) is the real time the sample covers, ending at its timestamp:
        # ~1 ms while the NPU is busy, ~200 ms while idle. Column 1 is the nominal period (1 ms).
        per = num(r[1]) or 0
        t = ts(r[2]) - t0 - per
        S["t"].append(t)
        S["per"].append(round(per))
        S["load"].append(round(num(r[6]) or 0))
        S["hvx"].append(round(num(r[18]) or 0))
        S["mpps"].append(round(num(r[11]) or 0))
        S["rd"].append(round(num(r[13]) or 0, 1))
        S["wr"].append(round(num(r[14]) or 0, 1))
        c, b = round(num(r[7]) or 0), round(num(r[8]) or 0)
        if not clk or clk[-1][1] != c:
            clk.append([t, c])
        if not bus or bus[-1][1] != b:
            bus.append([t, b])
        share[c] = share.get(c, 0) + per
        tot += per
    S["clk"], S["bus"] = clk, bus
    span = ts(rows[-1][2]) - t0
    S["coverage"] = round(tot / span, 3) if span else 0
    if S["coverage"] > 0.9:  # time-weighted averages need samples that cover the capture
        S["eff_avg"] = round(sum(e * p for e, p in zip(S["load"], S["per"])) / tot, 1)
        S["clk_avg"] = round(sum(num(r[7]) * (num(r[1]) or 0) for r in rows) / tot, 1)
        S["rd_avg"] = round(sum(v * p for v, p in zip(S["rd"], S["per"])) / tot, 1)
        S["wr_avg"] = round(sum(v * p for v, p in zip(S["wr"], S["per"])) / tot, 1)
    S["share"] = {str(k): round(100 * v / tot, 1) for k, v in sorted(share.items())}
    S["end"] = S["t"][-1] + S["per"][-1]
    print(f"  samples cover {tot / 1000:.2f} s of {span / 1000:.2f} s")
    return S


def read_invokes(path):
    rows = [r for r in csv.reader(open(path))][1:]
    return [[round(num(r[2]), 1), round(num(r[3]), 2), round(num(r[6]), 2), round(num(r[7]), 2)]
            for r in rows if len(r) >= 8 and num(r[2]) is not None]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    data = []
    for cid, label, what, ms, ms_sys in CAPTURES:
        d = glob.glob(os.path.join(src, cid, "sysmon_report_*"))[0]
        summ, inv = read_summary(os.path.join(d, "Summary.csv"))
        hv = os.path.join(d, "hvx_invokes.csv")
        invokes = read_invokes(hv) if os.path.exists(hv) else []
        c = {"id": cid, "label": label, "what": what, "ms": ms, "ms_sysmon": ms_sys,
             "dcvs": read_dcvs(os.path.join(d, "dcvs_metric.txt")), "summary": summ, "invoke_stats": inv,
             "invokes": invokes, "series": read_series(os.path.join(d, "post_processed_metrics.csv"))}
        if invokes:
            stories = len(invokes) * STORIES_PER_CALL[cid]
            c["per_story"] = {
                "stories": round(stories), "window_ms": round(1000 * c["dcvs"]["total_s"] / stories),
                "mcycles": round(sum(i[1] for i in invokes) / stories, 1),
                "ddr_rd": round(sum(i[2] for i in invokes) / stories, 1),
                "ddr_wr": round(sum(i[3] for i in invokes) / stories, 1),
                "qnn_npu_ms": QNN_NPU_MS[cid], "busy_qnn": round(100 * QNN_NPU_MS[cid] / ms),
            }
        data.append(c)
        print(f"{cid}: {len(c['series']['t'])} samples, {len(invokes)} invokes")
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncm_sysmon_comparison.template.html")).read()
    html = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    open(dst, "w").write(html)
    print(f"wrote {dst} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
