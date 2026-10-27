"""Render the scroll-scrubbed frame sequence used by index.html from the Gemini stills.

    python tools/cut_layers.py _src/gemini/exploded.webp _src/gemini/closed.webp
    python tools/make_film.py

Same film format as the Oha Burger site, without a video clip.

1. The burger: the six ingredient layers are composited on the orange backdrop for every frame.
   It stands assembled, the ingredients lift off one after another from the top, hang apart for
   the labels, then settle back from the bottom up. A slight tilt while they move, a shadow on
   the layer below and a contact shadow on the floor keep it physical.
2. The box (_src/gemini/box-*.webp, one camera): the camera pulls back from the burger's bun to
   the open box between the cola and the fries, the burger dissolving into the one in the box on
   the way, as Oha's transition did. Then the lid swings shut through Gemini's four lid poses
   (open, half, nearly shut, closed), changing over the box only.

Writes frames/d/0000.webp… (1280x720, desktop) and frames/m/… (768x432, phones) and prints the
film section's data-frames and the ingredient labels' data-x / data-y (percent of the frame).
"""
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
LAYERS = ROOT / "_src" / "layers"
BOX_POSES = [ROOT / "_src" / "gemini" / f"box-{n}.webp" for n in ("open", "mid", "shut", "closed")]
OUT = ROOT / "frames"

W, H = 1280, 720
FPS = 15
CX = .58 * W                        # burger centre line; captions sit on the left, as in Oha
CLOSED_FLOOR = .82 * H              # where the assembled burger stands
OPEN_TOP, OPEN_FLOOR = .06 * H, .94 * H
# film seconds: hold assembled, lift apart, hang (labels), settle back, hold
T_SPLIT, T_OPEN, T_JOIN, T_SHUT = 1.0, 3.2, 4.8, 6.6
# then: pull back to the box (burger dissolving into it), hold, lid closes, hold
T_PULL, T_BOXED, T_LID, T_SEALED, T_END = 7.0, 9.2, 10.0, 11.0, 11.6
DISSOLVE = .5
STAGGER = .07                       # each layer starts this much later (fraction of the move)
ROT = [-3, 2.2, -1.6, 1.4, -2, .8]  # degrees of tilt mid-move
DRIFT = 6                           # px the open stack rises while it hangs
LABEL_KEYS = (T_OPEN, T_JOIN)
# the backdrop: a hot spot behind the burger cooling to brick (index.html --brick-deep is the edge)
STOPS = [(0, (255, 166, 79)), (.22, (245, 113, 31)), (.46, (210, 68, 26)), (.74, (163, 42, 20)), (1, (116, 27, 14))]
SHADOW = (70, 14, 4)

# The box renders (1024x572), measured once: the top bun in the open box spans x 423-636 with its
# top at y 216; the lid (everything that changes between the two) spans y 85-450 (BOX_LID).
BOX_BUN = (529.5, 216, 213)         # centre x, top y, width
BOX_LID = (382, 95, 682, 445)
BOX_SCALE = .95                     # the box scene's final size in the frame
BOX_AT = (.65 * W, (H - 572 * BOX_SCALE) / 2 + 216 * BOX_SCALE)  # where the open box's bun lands
PAD = 640                           # render px of wall and floor added round the box renders


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


class Burger:
    """The ingredient layers, placed for any film second up to T_SHUT."""

    def __init__(self):
        d = json.loads((LAYERS / "layers.json").read_text(encoding="utf-8"))
        self.names = d["names"]
        boxes, (qx, qy, qw, qh) = d["layers"], d["closed"]
        self.n = len(boxes)
        top = min(b[1] for b in boxes)
        bottom = max(b[1] + b[3] for b in boxes)
        s = self.s = (OPEN_FLOOR - OPEN_TOP) / (bottom - top)
        ex_cx = (min(b[0] for b in boxes) + max(b[0] + b[2] for b in boxes)) / 2
        cl_cx, cl_floor = qx + qw / 2, qy + qh

        self.pieces, self.shadows, self.spots, self.label_pts = [], [], [], []
        for i, (x, y, w, h, px, py) in enumerate(boxes):
            im = Image.open(LAYERS / f"l{i}.png").convert("RGBA")
            im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
            self.pieces.append(im)
            pad = 40
            a = Image.new("L", (im.width + 2 * pad, im.height + 2 * pad), 0)
            a.paste(im.getchannel("A"), (pad, pad))
            a = a.filter(ImageFilter.GaussianBlur(9))
            sh = Image.new("RGBA", a.size, SHADOW + (255,))
            sh.putalpha(a)
            self.shadows.append(sh)
            open_xy = (CX + (x - ex_cx) * s, OPEN_TOP + (y - top) * s)
            shut_xy = (CX + (px - cl_cx) * s, CLOSED_FLOOR - (cl_floor - py) * s)
            self.spots.append((shut_xy, open_xy))
            # label dot: on the food, 70% across the layer at its middle row
            alpha = np.asarray(im.getchannel("A")) > 128
            row = int(np.average(np.arange(alpha.shape[0]), weights=alpha.sum(1)))
            xs = np.flatnonzero(alpha[row])
            dot_x = xs[np.abs(xs - (xs[0] + .7 * (xs[-1] - xs[0]))).argmin()]
            self.label_pts.append((open_xy[0] + dot_x, open_xy[1] + row))
        (sx, sy), _ = self.spots[0]
        self.bun = (sx + self.pieces[0].width / 2, sy, self.pieces[0].width)  # assembled: centre x, top, width
        self.back = backdrop()

    def frame(self, t):
        frame = self.back.copy()
        n = self.n
        drift = DRIFT * clamp((t - T_OPEN) / (T_JOIN - T_OPEN)) if t >= T_OPEN else 0
        placed = []
        for i in range(n):
            p = separation(t, i, n)
            (sx, sy), (ox, oy) = self.spots[i]
            x = sx + (ox - sx) * p
            y = sy + (oy - sy) * p - drift * p
            angle = ROT[i] * math.sin(math.pi * p) if 0 < p < 1 else 0
            placed.append((x, y, angle, p))

        # contact shadow under the bottom bun
        bx, by, _, _ = placed[-1]
        bw, bh = self.pieces[-1].size
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
            im, sh = self.pieces[i], self.shadows[i]
            if angle:
                im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
                sh = sh.rotate(angle, resample=Image.BICUBIC, expand=True)
            cx, cy = x + self.pieces[i].width / 2, y + self.pieces[i].height / 2
            if i < n - 1 and p > 0:
                a = sh.getchannel("A").point(lambda v, k=.5 * p: int(v * k))
                sh = sh.copy()
                sh.putalpha(a)
                frame.alpha_composite(sh, (round(cx - sh.width / 2), round(cy - sh.height / 2 + 14 * self.s * p)))
            frame.alpha_composite(im, (round(cx - im.width / 2), round(cy - im.height / 2)))
        return np.asarray(frame.convert("RGB"))


def box_poses():
    """The four lid renders, each brought to the open render's exposure (Gemini drifts it a
    little per image) by a per-channel gain measured on the wall and table beside the box."""
    imgs = [np.asarray(Image.open(p).convert("RGB")).astype(np.float32) for p in BOX_POSES]
    side = np.ones(imgs[0].shape[:2], bool)
    side[:, BOX_LID[0] - 40:BOX_LID[2] + 40] = False
    ref = imgs[0][side].mean(0)
    return [img * (ref / img[side].mean(0)) for img in imgs]


def padded(img):
    """A box render with PAD px of its wall and floor carried on all round, so the frame can show
    more than the render. Sideways, every row continues with the colour of its own edge columns
    (so the horizon runs straight on); up and down, the top and bottom rows carry on."""
    h, w = img.shape[:2]
    left = cv2.GaussianBlur(np.median(img[:, 8:48], axis=1)[:, None], (0, 0), 3)   # h x 1 x 3
    right = cv2.GaussianBlur(np.median(img[:, -48:-8], axis=1)[:, None], (0, 0), 3)
    # ease the render's outer columns into those colours, wide enough that its vignette doesn't
    # end in a visible seam
    ramp = np.clip(np.arange(w, dtype=np.float32) / 220, 0, 1)
    ramp = np.minimum(ramp, ramp[::-1])
    ramp = (ramp * ramp * (3 - 2 * ramp))[None, :, None]
    side = np.where(np.arange(w)[None, :, None] < w / 2, left, right)
    img = img * ramp + side * (1 - ramp)
    rows = np.concatenate([np.repeat(left, PAD, 1), img, np.repeat(right, PAD, 1)], axis=1)
    big = np.concatenate([np.repeat(rows[:1], PAD, 0), rows, np.repeat(rows[-1:], PAD, 0)], axis=0)
    big += np.random.default_rng(11).normal(0, 2.2, big.shape)  # same grain as the burger backdrop
    return np.clip(big, 0, 255).astype(np.float32)


def lid_region(open_big, shut_big):
    """Where the two box renders really differ (the lid, the logo, the burger it hides), solid
    inside and feathered a little: elsewhere they match, so no edge shows."""
    d = np.abs(open_big - shut_big).sum(2) > 45
    d[:, :PAD + BOX_LID[0] - 30] = d[:, PAD + BOX_LID[2] + 30:] = False  # the cup and fries stay put
    d = cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    d = cv2.dilate(d, np.ones((19, 19), np.uint8))
    d = clean_holes(d)
    return cv2.GaussianBlur(d.astype(np.float32), (0, 0), 5)


def clean_holes(m):
    flood = np.pad(m, 1)
    cv2.floodFill(flood, None, (0, 0), 1)
    return m | (1 - flood[1:-1, 1:-1])


def camera(k, anchor_img, anchor_frame):
    """Affine that scales the box render by k and puts anchor_img (render px) at anchor_frame."""
    ax, ay = anchor_img[0] + PAD, anchor_img[1] + PAD
    return np.float32([[k, 0, anchor_frame[0] - ax * k], [0, k, anchor_frame[1] - ay * k]])


def main():
    burger = Burger()
    poses = [padded(img) for img in box_poses()]
    open_big = poses[0]
    region = np.max([lid_region(open_big, big) for big in poses[1:]], axis=0)[..., None]
    last = burger.frame(T_SHUT)

    # the pull-back: from the burger's bun filling the frame as it does now, to the box scene
    bx, by, bw = burger.bun
    k0, k1 = bw / BOX_BUN[2], BOX_SCALE
    a0, a1 = (bx, by), BOX_AT

    def boxed(t):
        u = ease(clamp((t - T_PULL) / (T_BOXED - T_PULL)))
        k = math.exp(math.log(k0) + (math.log(k1) - math.log(k0)) * u)
        at = (a0[0] + (a1[0] - a0[0]) * u, a0[1] + (a1[1] - a0[1]) * u)
        src = open_big
        if t > T_LID:  # the lid swings shut: pose to pose, each held and changed over quickly
            s = ease(clamp((t - T_LID) / (T_SEALED - T_LID))) * (len(poses) - 1)
            i = min(int(s), len(poses) - 2)
            f = clamp((s - i - .3) / .4)
            lid = poses[i] * (1 - f) + poses[i + 1] * f
            src = open_big * (1 - region) + lid * region
        img = cv2.warpAffine(src, camera(k, BOX_BUN[:2], at), (W, H), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        d = clamp((t - T_PULL) / DISSOLVE)
        if d < 1:  # the burger shot rides the same camera move while it dissolves into the box
            r = k / k0
            m = np.float32([[r, 0, at[0] - a0[0] * r], [0, r, at[1] - a0[1] * r]])
            shot = cv2.warpAffine(last.astype(np.float32), m, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            img = shot * (1 - ease(d)) + img * ease(d)
        return np.clip(img, 0, 255).astype(np.uint8)

    shutil.rmtree(OUT, ignore_errors=True)
    for folder in ("d", "m"):
        (OUT / folder).mkdir(parents=True)
    count = round(T_END * FPS) + 1
    sizes = {"d": 0, "m": 0}
    for f in range(count):
        t = f / FPS
        rgb = Image.fromarray(burger.frame(min(t, T_SHUT)) if t <= T_PULL else boxed(t))
        for folder, size, quality in [("d", (W, H), 70), ("m", (768, 432), 62)]:
            path = OUT / folder / f"{f:04d}.webp"
            (rgb if size == (W, H) else rgb.resize(size, Image.LANCZOS)).save(path, "WEBP", quality=quality, method=6)
            sizes[folder] += path.stat().st_size

    for folder in ("d", "m"):
        print(f"frames/{folder}: {count} frames, {sizes[folder] / 1e6:.1f} MB")
    print(f'data-frames="{count}" data-fps="{FPS}"')
    print(f'labels data-keys="{",".join(f"{k:g}" for k in LABEL_KEYS)}"')
    for i, (lx, ly) in enumerate(burger.label_pts):
        xs = [lx / W * 100] * 2
        ys = [ly / H * 100, (ly - DRIFT) / H * 100]
        print(f'  {burger.names[i]}: data-x="{",".join(f"{v:.1f}" for v in xs)}" data-y="{",".join(f"{v:.1f}" for v in ys)}"')


if __name__ == "__main__":
    main()
