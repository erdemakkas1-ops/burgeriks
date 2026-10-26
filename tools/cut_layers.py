"""Cut the Gurme Burgeriks product photo into ingredient layers for the scroll film.

    python tools/cut_layers.py _src/tgo/p/gurme.jpg

Writes layers/l0.webp (top bun) ... l5.webp (pickles, sauce and bottom bun) with alpha, each
cropped to its own box, plus layers/s0..s5.webp, their soft drop shadows, and prints the boxes
([x, y, w, h] in the 1280 px source) for the film's data-boxes attribute, so the page can
re-stack them exactly as photographed.

Layers follow the food, not straight lines: every pixel gets an ingredient label from rough
seams plus colour (pink/red meat goes with the meat layer wherever it hangs, pale shoestring
fries with the fries, the cheese edge decides where the onions end). Gaps a layer had under
the one above it are filled from its own colours, and the grey studio fringe on the outline
is replaced with the food's colour so the layers sit cleanly on any backdrop.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "layers"

NAMES = ["Üst ekmek", "Pastırma & jambon", "Karamelize soğan", "Cheddar & köfte", "Kibrit patates", "Turşu, sos & alt ekmek"]
# Rough seams between layers, top to bottom, as (x, y) points in the 1280 px photo; colour
# rules below move pixels across them.
SEAMS = [
    [(0, 430), (230, 425), (330, 400), (530, 392), (750, 388), (950, 400), (1050, 415), (1280, 430)],  # bun | meat
    [(0, 560), (250, 555), (400, 548), (640, 545), (900, 550), (1100, 560), (1280, 565)],              # meat | onion
    [(0, 668), (200, 668), (640, 668), (1000, 668), (1280, 668)],                                       # onion | patty (cheese edge wins)
    [(0, 785), (300, 790), (640, 795), (950, 790), (1280, 785)],                                        # patty | fries
    [(0, 880), (300, 885), (640, 885), (950, 885), (1280, 880)],                                        # fries | pickles + bottom bun
]
SOLID_LAYERS = {0, 2, 3, 5}  # filled column by column; meat and fries keep their ragged shapes
SHADOW_PAD = 40  # source px around each layer's shadow sprite; the page's SHADOW_PAD must match


def burger_mask(im: Image.Image) -> Image.Image:
    """Foreground = anything with colour; the studio backdrop and table are neutral grey."""
    r, g, b = im.split()
    hi = ImageChops.lighter(ImageChops.lighter(r, g), b)
    lo = ImageChops.darker(ImageChops.darker(r, g), b)
    chroma = ImageChops.subtract(hi, lo)
    dark = lo.point(lambda v: 255 if v < 70 else 0)  # charred patty edges are dark but grey
    m = ImageChops.lighter(chroma.point(lambda v: 255 if v > 26 else 0), dark)
    # close small holes (sesame seeds, cheese highlights), then drop specks
    m = m.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))
    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    # keep only the burger: flood from a point on it, everything else unreachable
    filled = m.copy()
    ImageDraw.floodfill(filled, (640, 700), 128, thresh=0)
    m = filled.point(lambda v: 255 if v == 128 else 0)
    # fill interior holes: flood the background from a corner, invert
    bg = m.copy()
    ImageDraw.floodfill(bg, (2, 2), 128, thresh=0)
    m = bg.point(lambda v: 0 if v == 128 else 255)
    # the bun's contact shadow on the table is warm enough to pass as burger: in each column,
    # nothing below the lowest properly saturated pixel belongs to it
    w, h = m.size
    px, mp = im.load(), m.load()
    sat = lambda c: (max(c) - min(c)) / max(max(c), 1)
    for x in range(w):
        low = next((y for y in range(h - 1, -1, -1) if sat(px[x, y]) > .32), -1)
        for y in range(low + 2, h):
            mp[x, y] = 0
    return m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.2))


def seam_y(points, w):
    xs, ys = zip(*points)
    return np.interp(np.arange(w), xs, ys)


def clean(mask, close=9, min_area=400, holes=True):
    """Binary mask → closed, small specks dropped, interior holes filled (unless holes=False)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    m = keep[cc].astype(np.uint8)
    if not holes:
        return m.astype(bool)
    # flood the outside from a 1 px frame, so areas cut off by things touching the edges
    # still count as outside
    flood = np.pad(m, 1)
    cv2.floodFill(flood, None, (0, 0), 1)
    return (m | (1 - flood[1:-1, 1:-1])).astype(bool)


def reddish(H, S, V):
    return ((H <= 9) | (H >= 165)) & (S > 60) & (V > 55)


def labels_for(rgb, solid):
    h, w = solid.shape
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(int)  # H 0-180
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    Y = np.arange(h)[:, None].repeat(w, 1)
    seams = [seam_y(s, w)[None, :] for s in SEAMS]

    meat = ((H <= 9) | (H >= 165)) & (S > 60) & (V > 95)  # caramelised onions are reddish but darker
    cheese = (H >= 11) & (H <= 25) & (S > 150) & (V > 160)
    fry = (H >= 14) & (H <= 30) & (S > 50) & (S < 175) & (V > 150)

    # 1. rough bands from the seams
    lab = np.zeros((h, w), np.int8)
    for i, s in enumerate(seams):
        lab[Y >= s] = i + 1

    # 2. onions end where the cheese starts: per column, the top of the first solid cheese run
    top = np.full(w, np.nan)
    band = cheese & (Y >= 625) & (Y <= 735)
    run = cv2.erode(band.astype(np.uint8), np.ones((5, 1), np.uint8)).astype(bool)
    for x in range(w):
        ys = np.flatnonzero(run[:, x])
        if ys.size:
            top[x] = ys[0] - 2
    known = ~np.isnan(top)
    top = np.where(known, top, seam_y(SEAMS[2], w))
    top = np.convolve(np.pad(top, 20, mode="edge"), np.ones(41) / 41, mode="valid")
    lab[(lab == 3) & (Y < top[None, :])] = 2
    lab[(lab == 2) & (Y >= top[None, :])] = 3

    # 3. meat: everything pink/red between the bun and the cheese (not hole-filled: onions
    # show between the ham and the bacon strip below it)
    m = clean(meat & (Y > seams[0] - 25) & (Y < 700), close=11, min_area=600, holes=False)
    lab[m & (lab >= 1) & (lab <= 3)] = 1
    # below the ham's seam only meat stays meat (the onions start there); above it the dark
    # folds between slices belong to the ham, and the bun's underside peeking over it to the bun
    bun = (H >= 8) & (H <= 20) & (V > 90) & ~meat
    lab[(lab == 1) & ~m & bun & (Y < seams[0] + 35)] = 0
    lab[(lab == 1) & ~m & (Y >= seams[1] - 15)] = 2
    # and onions reaching up between the slices join them, so their top follows the ham's
    # underside rather than the seam
    nm = (lab == 1) & ~m & solid & (Y > seams[1] - 50)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(nm.astype(np.uint8), 8)
    below = cv2.dilate((lab == 2).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    touching = np.unique(cc[below & nm])
    lab[np.isin(cc, touching[touching > 0])] = 2
    # the ham's darker, shaded edges (too dim for `meat`) go with the ham they hang from
    near = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    lab[(lab == 2) & reddish(H, S, V) & near] = 1
    lab[(lab == 2) & (Y < seams[1] - 30)] = 1  # the shade right under the ham stays with it

    # 4. fries vs patty: pale shoestrings go with the fries, saturated cheese drips with the patty
    lab[(lab == 4) & cheese & (Y < seams[3] + 25)] = 3
    f = clean(fry & (Y > seams[3] - 45) & (Y < seams[4] + 70), close=11, min_area=150)
    lab[f & ((lab == 3) | (lab == 5))] = 4
    # the fries keep their dense middle; along the bottom only actual fries stay, so the layer
    # ends in loose sticks and the pickles under them go to the bottom layer
    lab[(lab == 4) & ~f & (Y > seams[4] - 30)] = 5
    sauce = ((H <= 12) | (H >= 170)) & (S > 50) & (V > 150) & ~fry
    lab[(lab == 4) & sauce] = 5

    # 5. tidy: tiny islands join the label around them
    lab[~solid] = -1
    for i in range(len(SEAMS) + 1):
        n, cc, stats, _ = cv2.connectedComponentsWithStats((lab == i).astype(np.uint8), 8)
        for c in range(1, n):
            if stats[c, cv2.CC_STAT_AREA] >= 260:
                continue
            blob = cc == c
            ring = cv2.dilate(blob.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~blob
            vals = lab[ring]
            vals = vals[(vals >= 0) & (vals != i)]
            if vals.size:
                lab[blob] = np.bincount(vals).argmax()
    return lab


def fill_from(rgb, known, todo):
    """Fill `todo` pixels with colours spread from `known` pixels only (normalised blur)."""
    out = rgb.astype(np.float32).copy()
    have = known.astype(np.float32)
    acc = out * have[..., None]
    for sigma in (2, 4, 8, 16, 32):
        num = cv2.GaussianBlur(acc, (0, 0), sigma)
        den = cv2.GaussianBlur(have, (0, 0), sigma)
        ok = todo & (den > .02)
        out[ok] = num[ok] / den[ok][:, None]
        acc[ok] = out[ok]
        have[ok] = 1
        todo = todo & ~ok
    return out


def main(src: str) -> None:
    im = Image.open(src).convert("RGB")
    rgb = np.asarray(im)
    alpha = np.asarray(burger_mask(im)).astype(np.float32) / 255
    solid = alpha > .5
    lab = labels_for(rgb, solid)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(int)
    red = reddish(hsv[..., 0], hsv[..., 1], hsv[..., 2])
    OUT.mkdir(exist_ok=True)
    layers = []
    for i in range(len(SEAMS) + 1):
        own = lab == i
        # the shape of the ingredient: own pixels closed over the gaps the layer above covered
        shape = clean(own, close=31, min_area=2000) & solid
        if i in SOLID_LAYERS:
            # buns, onions and patty are solid discs: fill each column between the layer's own
            # (smoothed) top and bottom, so what the layer above hid is filled in, not a hole
            body = clean(own, close=9, min_area=3000)
            cols = np.flatnonzero(body.any(0))
            tops = np.full(body.shape[1], np.nan)
            bots = np.full(body.shape[1], np.nan)
            for x in cols:
                ys = np.flatnonzero(body[:, x])
                tops[x], bots[x] = ys[0], ys[-1]
            k = np.ones(25) / 25
            for arr in (tops, bots):
                v = ~np.isnan(arr)
                sm = np.convolve(np.where(v, arr, 0), k, "same") / np.maximum(np.convolve(v.astype(float), k, "same"), 1e-6)
                arr[v] = sm[v]
            Yc = np.arange(body.shape[0])[:, None]
            span = (Yc >= tops[None, :]) & (Yc <= bots[None, :])
            shape |= span & solid
        shape |= own
        # only the ingredient itself: crumbs far from it (stray pixels of its colour) go
        n, cc, stats, _ = cv2.connectedComponentsWithStats(shape.astype(np.uint8), 8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        shape = np.isin(cc, 1 + np.flatnonzero(areas >= .02 * areas.max()))
        # 2 px of overlap into the neighbours hides seams while the stack is closed
        shape = cv2.dilate(shape.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & solid
        # colours: own pixels, gaps filled from them, and the outline fringe (grey studio mixed
        # in) replaced from the pixels just inside it
        core = cv2.erode(own.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if i == 2:
            core &= ~red  # stray ham shading left on the onions is repainted with onion

        col = fill_from(rgb, core, shape & ~core)
        a = shape.astype(np.float32) * np.clip(alpha * 1.15, 0, 1)
        a = cv2.GaussianBlur(a, (0, 0), .7)
        rgba = np.dstack([np.clip(col, 0, 255), a * 255]).astype(np.uint8)
        ys, xs = np.nonzero(a > .03)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        piece = Image.fromarray(rgba, "RGBA").crop(box)
        piece.save(OUT / f"l{i}.webp", quality=88, method=6)
        # pre-blurred shadow of the layer (canvas shadowBlur is far too slow per frame): the
        # layer's box plus SHADOW_PAD on every side, at half size, drawn scaled up under it
        pad = Image.new("L", (piece.width + 2 * SHADOW_PAD, piece.height + 2 * SHADOW_PAD), 0)
        pad.paste(piece.getchannel("A"), (SHADOW_PAD, SHADOW_PAD))
        pad = pad.resize((pad.width // 2, pad.height // 2), Image.LANCZOS).filter(ImageFilter.GaussianBlur(9))
        shadow = Image.new("RGBA", pad.size, (70, 14, 4, 0))
        shadow.putalpha(pad.point(lambda v: int(v * .62)))
        shadow.save(OUT / f"s{i}.webp", quality=70, method=6)
        layers.append({"name": NAMES[i], "x": box[0], "y": box[1], "w": box[2] - box[0], "h": box[3] - box[1]})
    (OUT / "layers.json").write_text(json.dumps(layers, ensure_ascii=False), encoding="utf-8")
    print(json.dumps([[l["x"], l["y"], l["w"], l["h"]] for l in layers]))


if __name__ == "__main__":
    main(sys.argv[1])
