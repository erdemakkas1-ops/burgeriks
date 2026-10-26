"""Render the scroll-scrubbed frame sequence used by index.html from the Gemini stills.

    python tools/cut_layers.py _src/gemini/exploded.webp _src/gemini/closed.webp
    python tools/make_film.py

Same film format as the Oha Burger site, without a video clip: the six ingredient layers are
composited on the orange backdrop for every frame. The burger stands assembled, the ingredients
lift off one after another from the top, hang apart for the labels, then settle back from the
bottom up. A slight tilt while they move, a shadow on the layer below and a contact shadow on
the floor keep it physical.

Writes frames/d/0000.webp… (1280x720, desktop) and frames/m/… (768x432, phones) and prints the
film section's data-frames and the ingredient labels' data-x / data-y (percent of the frame).
"""
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
LAYERS = ROOT / "_src" / "layers"
OUT = ROOT / "frames"

W, H = 1280, 720
FPS = 15
CX = .58 * W                        # burger centre line; captions sit on the left, as in Oha
CLOSED_FLOOR = .82 * H              # where the assembled burger stands
OPEN_TOP, OPEN_FLOOR = .06 * H, .94 * H
# film seconds: hold assembled, lift apart, hang (labels), settle back, hold
T_SPLIT, T_OPEN, T_JOIN, T_SHUT, T_END = 1.0, 3.2, 4.8, 6.6, 7.4
STAGGER = .07                       # each layer starts this much later (fraction of the move)
ROT = [-3, 2.2, -1.6, 1.4, -2, .8]  # degrees of tilt mid-move
DRIFT = 6                           # px the open stack rises while it hangs
LABEL_KEYS = (T_OPEN, T_JOIN)
# the backdrop: a hot spot behind the burger cooling to brick (index.html --brick-deep is the edge)
STOPS = [(0, (255, 166, 79)), (.22, (245, 113, 31)), (.46, (210, 68, 26)), (.74, (163, 42, 20)), (1, (116, 27, 14))]
SHADOW = (70, 14, 4)


def ease(t):
    return t * t * (3 - 2 * t)


def clamp(t):
    return min(1.0, max(0.0, t))


def backdrop():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    sx, sy = CX, .5 * H
    far = max(math.hypot(cx - sx, cy - sy) for cx in (0, W) for cy in (0, H))
    t = np.hypot(x - sx, y - sy) / far
    img = np.zeros((H, W, 3), np.float32)
    for (t0, c0), (t1, c1) in zip(STOPS, STOPS[1:]):
        k = np.clip((t - t0) / (t1 - t0), 0, 1)[..., None]
        band = (t >= t0) & (t <= t1)
        img[band] = (np.array(c0) + (np.array(c1) - np.array(c0)) * k)[band]
    img += np.random.default_rng(7).normal(0, 2.2, img.shape)  # fine grain: no banding in webp
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGBA")


def separation(t, i, n):
    """0 = in place in the assembled burger, 1 = where it floats in the exploded view."""
    span = 1 - STAGGER * (n - 1)
    if t < T_JOIN:   # lifting off: top layer first
        u = (t - T_SPLIT) / (T_OPEN - T_SPLIT)
        return ease(clamp((u - STAGGER * i) / span))
    v = (t - T_JOIN) / (T_SHUT - T_JOIN)  # settling back: bottom layer first
    return 1 - ease(clamp((v - STAGGER * (n - 1 - i)) / span))


def main():
    d = json.loads((LAYERS / "layers.json").read_text(encoding="utf-8"))
    boxes, (qx, qy, qw, qh) = d["layers"], d["closed"]
    n = len(boxes)
    top = min(b[1] for b in boxes)
    bottom = max(b[1] + b[3] for b in boxes)
    s = (OPEN_FLOOR - OPEN_TOP) / (bottom - top)
    ex_cx = (min(b[0] for b in boxes) + max(b[0] + b[2] for b in boxes)) / 2
    cl_cx, cl_floor = qx + qw / 2, qy + qh

    pieces, shadows, spots, label_pts = [], [], [], []
    for i, (x, y, w, h, px, py) in enumerate(boxes):
        im = Image.open(LAYERS / f"l{i}.png").convert("RGBA")
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        pieces.append(im)
        pad = 40
        a = Image.new("L", (im.width + 2 * pad, im.height + 2 * pad), 0)
        a.paste(im.getchannel("A"), (pad, pad))
        a = a.filter(ImageFilter.GaussianBlur(9))
        sh = Image.new("RGBA", a.size, SHADOW + (255,))
        sh.putalpha(a)
        shadows.append(sh)
        open_xy = (CX + (x - ex_cx) * s, OPEN_TOP + (y - top) * s)
        shut_xy = (CX + (px - cl_cx) * s, CLOSED_FLOOR - (cl_floor - py) * s)
        spots.append((shut_xy, open_xy))
        # label dot: on the food, 70% across the layer at its middle row
        alpha = np.asarray(im.getchannel("A")) > 128
        row = int(np.average(np.arange(alpha.shape[0]), weights=alpha.sum(1)))
        xs = np.flatnonzero(alpha[row])
        dot_x = xs[np.abs(xs - (xs[0] + .7 * (xs[-1] - xs[0]))).argmin()]
        label_pts.append((open_xy[0] + dot_x, open_xy[1] + row))

    back = backdrop()
    shutil.rmtree(OUT, ignore_errors=True)
    for folder in ("d", "m"):
        (OUT / folder).mkdir(parents=True)
    count = round(T_END * FPS) + 1
    sizes = {"d": 0, "m": 0}
    for f in range(count):
        t = f / FPS
        frame = back.copy()
        drift = DRIFT * clamp((t - T_OPEN) / (T_JOIN - T_OPEN)) if t >= T_OPEN else 0
        placed = []
        for i in range(n):
            p = separation(t, i, n)
            (sx, sy), (ox, oy) = spots[i]
            x = sx + (ox - sx) * p
            y = sy + (oy - sy) * p - drift * p
            moving = 0 < p < 1
            angle = ROT[i] * math.sin(math.pi * p) if moving else 0
            placed.append((x, y, angle, p))

        # contact shadow under the bottom bun
        bx, by, _, _ = placed[-1]
        bw, bh = pieces[-1].size
        floor = by + bh
        cs = Image.new("L", (W, H), 0)
        ImageDraw.Draw(cs).ellipse((bx + bw * .06, floor - bw * .05, bx + bw * .94, floor + bw * .035), fill=170)
        cs = cs.filter(ImageFilter.GaussianBlur(bw * .045))
        dark = Image.new("RGBA", (W, H), SHADOW + (255,))
        dark.putalpha(cs)
        frame.alpha_composite(dark)

        # bottom first, so each layer shades the one below as they part; but the ham hangs in front
        # of the top bun's cut face (as in the closed render), so the bun goes down before it
        for i in [k for k in range(n - 1, 1, -1)] + [0, 1]:
            x, y, angle, p = placed[i]
            im, sh = pieces[i], shadows[i]
            if angle:
                im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
                sh = sh.rotate(angle, resample=Image.BICUBIC, expand=True)
            cx, cy = x + pieces[i].width / 2, y + pieces[i].height / 2
            if i < n - 1 and p > 0:
                a = sh.getchannel("A").point(lambda v, k=.5 * p: int(v * k))
                sh = sh.copy()
                sh.putalpha(a)
                frame.alpha_composite(sh, (round(cx - sh.width / 2), round(cy - sh.height / 2 + 14 * s * p)))
            frame.alpha_composite(im, (round(cx - im.width / 2), round(cy - im.height / 2)))

        rgb = frame.convert("RGB")
        for folder, size, quality in [("d", (W, H), 70), ("m", (768, 432), 62)]:
            path = OUT / folder / f"{f:04d}.webp"
            (rgb if size == (W, H) else rgb.resize(size, Image.LANCZOS)).save(path, "WEBP", quality=quality, method=6)
            sizes[folder] += path.stat().st_size

    for folder in ("d", "m"):
        print(f"frames/{folder}: {count} frames, {sizes[folder] / 1e6:.1f} MB")
    print(f'data-frames="{count}" data-fps="{FPS}"')
    print(f'labels data-keys="{",".join(f"{k:g}" for k in LABEL_KEYS)}"')
    for i, (lx, ly) in enumerate(label_pts):
        xs = [lx / W * 100] * 2
        ys = [ly / H * 100, (ly - DRIFT) / H * 100]
        print(f'  {d["names"][i]}: data-x="{",".join(f"{v:.1f}" for v in xs)}" data-y="{",".join(f"{v:.1f}" for v in ys)}"')


if __name__ == "__main__":
    main()
