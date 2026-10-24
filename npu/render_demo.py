"""Render a demo_ncm.py recording (JSON lines) into a GIF and an MP4.

    python npu/render_demo.py demo/seed0.jsonl --out report/ncm_npu_demo
    -> report/ncm_npu_demo.gif, report/ncm_npu_demo.mp4

Text and timings are exactly what the board produced; only the display is slowed down.
Needs: pillow, imageio, imageio-ffmpeg.
"""
import argparse
import json
import textwrap

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
BG, PANEL, INK, DIM, FAINT, ACCENT = (15, 18, 22), (24, 28, 33), (232, 234, 237), (141, 148, 155), (52, 58, 64), (57, 135, 229)
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANS_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LEVELS = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]


def font(path, size):
    return ImageFont.truetype(path, size)


F_TITLE, F_SUB, F_TEXT, F_SMALL, F_TAG, F_BIG = (font(SANS_B, 40), font(SANS, 22), font(MONO, 21),
                                                 font(MONO, 16), font(MONO_B, 18), font(SANS_B, 30))


def header(d, subtitle="Radxa Dragon Q6A  ·  Qualcomm QCS6490  ·  Hexagon NPU"):
    d.text((64, 40), "Next-Concept Modeling on the Hexagon NPU", font=F_BIG, fill=INK)
    d.text((64, 82), subtitle, font=F_SMALL, fill=DIM)


def level_strip(d, done, y=128):
    """12 cells, width proportional to the number of codes at that level (coarse -> fine)."""
    x0, x1, gap = 64, W - 64, 4
    total = sum(LEVELS)
    usable = (x1 - x0) - gap * (len(LEVELS) - 1)
    x = x0
    for i, n in enumerate(LEVELS):
        w = max(6, usable * n / total)
        d.rounded_rectangle([x, y, x + w, y + 14], radius=3, fill=ACCENT if i < done else FAINT)
        x += w + gap
    d.text((64, y + 22), "coarse", font=F_SMALL, fill=DIM)
    d.text((x1, y + 22), "fine", font=F_SMALL, fill=DIM, anchor="ra")


def body_text(d, text, y=210, color=INK, max_lines=14):
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para.strip(), 88) or [""]
    for i, ln in enumerate(lines[:max_lines]):
        d.text((64, y + i * 30), ln, font=F_TEXT, fill=color)


def frame_title():
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((64, 230), "Next-Concept Modeling", font=F_TITLE, fill=INK)
    d.text((64, 285), "on the Qualcomm Hexagon NPU", font=F_TITLE, fill=ACCENT)
    d.text((64, 360), "A 20M-parameter language model, trained from scratch,", font=F_SUB, fill=DIM)
    d.text((64, 392), "writes a whole story coarse-to-fine in 12 steps instead of 128.", font=F_SUB, fill=DIM)
    d.text((64, 470), "Radxa Dragon Q6A  ·  QCS6490  ·  QAIRT 2.50", font=F_SMALL, fill=DIM)
    return im


def frame_level(e):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    header(d)
    level_strip(d, e["level"])
    tag = f"level {e['level']:>2}/12   {e['codes']:>3} codes   step {e['step_ms']:.1f} ms on the NPU"
    d.text((64, 176), tag, font=F_TAG, fill=ACCENT)
    body_text(d, e["text"] or "(nothing readable yet)")
    return im


def frame_final(summary, last):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    header(d)
    level_strip(d, 12)
    d.text((64, 176), "final story (unedited model output)", font=F_TAG, fill=ACCENT)
    body_text(d, summary["final"], max_lines=11)
    d.rectangle([0, H - 120, W, H], fill=PANEL)
    d.text((64, H - 100), f"{summary['gen_ms']:.0f} ms", font=F_TITLE, fill=INK)
    d.text((64 + d.textlength(f"{summary['gen_ms']:.0f} ms", font=F_TITLE) + 18, H - 88),
           "this story on the NPU: 12 generator steps + 1 decode", font=F_SUB, fill=DIM)
    d.text((64, H - 44), "average 143 ms over 48 stories  ·  1.9× faster than a tuned CPU (276 ms)  ·  "
           "display slowed down", font=F_SMALL, fill=DIM)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--out", default="report/ncm_npu_demo")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()
    ev = [json.loads(l) for l in open(args.record)]
    summary = ev[-1]["summary"]
    levels = ev[:-1]

    # (frame, seconds on screen)
    seq = [(frame_title(), 3.0)]
    for e in levels:
        seq.append((frame_level(e), 1.1 if e["level"] < 12 else 1.6))
    seq.append((frame_final(summary, levels[-1]), 6.0))

    seq[0][0].save(args.out + ".gif", save_all=True, append_images=[f for f, _ in seq[1:]],
                   duration=[int(s * 1000) for _, s in seq], loop=0, optimize=True)
    with imageio.get_writer(args.out + ".mp4", fps=args.fps, codec="libx264", quality=8,
                            pixelformat="yuv420p", macro_block_size=8) as wr:
        for f, s in seq:
            a = np.asarray(f)
            for _ in range(int(round(s * args.fps))):
                wr.append_data(a)
    total = sum(s for _, s in seq)
    print(f"wrote {args.out}.gif and {args.out}.mp4 ({len(seq)} frames, {total:.1f} s)")


if __name__ == "__main__":
    main()
