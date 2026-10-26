"""Cut the Gurme Burgeriks film layers from the two Gemini renders.

    python tools/cut_layers.py _src/gemini/exploded.webp _src/gemini/closed.webp

exploded: the burger's six ingredients floating apart on flat grey, with clear gaps between them.
closed:   the same burger assembled, same camera and backdrop.

Writes _src/layers/l0.png (top bun) ... l5.png (pickles, sauce, bottom bun), each cropped to its
own box, and _src/layers/layers.json: per layer its box in the exploded render [x, y, w, h] and
where it sits in the closed burger [x, y] (closed px scaled to the exploded render's size), plus
the closed burger's box. tools/make_film.py renders the film's frames from them.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_src" / "layers"  # inputs for tools/make_film.py, not served

NAMES = ["Üst ekmek", "Pastırma & jambon", "Karamelize soğan", "Cheddar & köfte", "Kibrit patates", "Turşu, sos & alt ekmek"]


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


def food_mask(rgb):
    """Food on the flat grey backdrop: anything with colour or clearly darker than the grey."""
    c = rgb.astype(int)
    chroma = c.max(2) - c.min(2)
    m = (chroma > 28) | (c.max(2) < 150)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = clean(m.astype(bool), close=5, min_area=60)
    # the render's anti-aliased outline is half grey backdrop: step 1 px inside it
    return cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def rgba(rgb, mask):
    """Cut-out with the grey fringe on the outline repainted from just inside it."""
    core = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    col = fill_from(rgb, core, mask & ~core)
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), .7)
    return np.dstack([np.clip(col, 0, 255), a * 255]).astype(np.uint8)


def pieces(mask, n):
    """The n floating layers, top to bottom: the n biggest connected pieces of food (the ham's
    hanging edges share rows with the onions, so rows alone can't split them); any crumb or
    stray fry joins the piece nearest to it."""
    count, cc, stats, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    big = sorted(range(1, count), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:n]
    if len(big) != n:
        raise SystemExit(f"expected {n} layers, found {len(big)} pieces of food")
    big.sort(key=lambda i: cen[i][1])
    out = [cc == i for i in big]
    for i in range(1, count):
        if i not in big:
            j = min(range(n), key=lambda j: abs(cen[i][1] - cen[big[j]][1]))
            out[j] |= cc == i
    return out


def place(layer_rgba, closed_rgb, closed_mask, box):
    """Where a layer sits in the closed burger: masked template match on colour, searched in a
    window around the expected spot (the layer's share of the burger's height)."""
    tpl = layer_rgba[..., :3].astype(np.float32)
    msk = (layer_rgba[..., 3] > 200).astype(np.float32)
    msk = np.dstack([msk] * 3)
    img = closed_rgb.astype(np.float32)
    x0, y0, x1, y1 = box
    res = cv2.matchTemplate(img[y0:y1, x0:x1], tpl, cv2.TM_SQDIFF, mask=msk)
    res = np.where(np.isfinite(res), res, res[np.isfinite(res)].max())
    _, _, (mx, my), _ = cv2.minMaxLoc(res)
    return x0 + mx, y0 + my


def main(exploded_path, closed_path):
    ex = np.asarray(Image.open(exploded_path).convert("RGB"))
    cl_full = np.asarray(Image.open(closed_path).convert("RGB"))
    OUT.mkdir(exist_ok=True)

    ex_mask = food_mask(ex)
    parts = pieces(ex_mask, len(NAMES))
    rows = [np.flatnonzero(p.any(1)) for p in parts]
    runs = [(int(r[0]), int(r[-1]) + 1) for r in rows]
    cl_mask_full = food_mask(cl_full)

    # The closed render frames the burger tighter: scale it so its top bun matches the exploded
    # top bun's width, then every position below is in exploded-render px.
    bun_w = np.ptp(np.flatnonzero(parts[0].any(0)))
    ys, xs = np.nonzero(cl_mask_full)
    cl_bun_rows = cl_mask_full[ys.min():ys.min() + int((runs[0][1] - runs[0][0]) * .6)]
    k = bun_w / np.ptp(np.flatnonzero(cl_bun_rows.any(0)))
    size = (round(cl_full.shape[1] * k), round(cl_full.shape[0] * k))
    cl = np.asarray(Image.fromarray(cl_full).resize(size, Image.LANCZOS))
    cl_mask = food_mask(cl)
    ys, xs = np.nonzero(cl_mask)
    cbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    # both stacks' centre line: the middle of the top bun
    ex_cx = int(np.flatnonzero(parts[0].any(0)).mean())
    cl_cx = int(np.flatnonzero(cl_mask[cbox[1]:cbox[1] + 40].any(0)).mean())

    layers = []
    ch = cbox[3] - cbox[1]
    for i, (a, b) in enumerate(runs):
        m = parts[i]
        cut = rgba(ex, m)
        yy, xx = np.nonzero(m)
        box = (int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)
        piece = cut[box[1]:box[3], box[0]:box[2]]
        Image.fromarray(piece, "RGBA").save(OUT / f"l{i}.png")

        # where it sits in the closed burger. Bun, meat, onions and patty look the same assembled:
        # find them by colour, on the burger's centre line, below the layer above. The fries
        # nest spreads wider once pressed and the bottom bun's top is hidden, so those two are
        # placed by rule: the bun on the burger's base, the fries tucked under the patty.
        h, w = piece.shape[:2]
        px = box[0] - ex_cx + cl_cx
        if i < 4:
            lo = cbox[1] if i == 0 else layers[-1]["closed"][1] + 4
            win = (max(0, px - 8), max(0, lo - 10), min(cl.shape[1], px + w + 8), min(cl.shape[0], lo + h + int(.2 * ch)))
            px, py = place(piece, cl, cl_mask, win)
        elif i == 4:
            _, ppy = layers[-1]["closed"]
            py = ppy + int(.72 * layers[-1]["box"][3])
        else:
            py = cbox[3] - h

        layers.append({"name": NAMES[i], "box": [box[0], box[1], box[2] - box[0], box[3] - box[1]], "closed": [int(px), int(py)]})

    data = {"layers": [[*l["box"], *l["closed"]] for l in layers],
            "closed": [cbox[0], cbox[1], cbox[2] - cbox[0], cbox[3] - cbox[1]],
            "scale": round(k, 4)}
    (OUT / "layers.json").write_text(json.dumps({"names": NAMES, **data}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(data))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
