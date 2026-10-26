"""Cut the Gurme Burgeriks product photo into stacked layers for the scroll film.

    python tools/cut_layers.py _src/tgo/p/gurme.jpg

Writes layers/l0.webp (top bun) ... layers/l4.webp (bottom bun) with alpha, each cropped to
its own box, plus layers/s0..s4.webp, their soft drop shadows, and prints the boxes ([x, y, w, h] in the 1280 px source) for the film's
data-boxes attribute, so the page can re-stack them exactly as photographed.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "layers"

# Cut lines between layers, top to bottom, as (x, y) points in the 1280 px photo.
# They follow the natural seams: bun underside, under the cured meats, under the patty,
# and through the sauce on the bottom bun.
CUTS = [
    [(0, 445), (130, 440), (230, 430), (330, 407), (430, 400), (530, 395), (650, 387), (750, 390), (850, 400), (950, 405), (1050, 420), (1160, 432), (1280, 440)],
    [(0, 650), (130, 642), (230, 602), (330, 578), (430, 572), (530, 562), (630, 557), (730, 562), (830, 567), (930, 590), (1050, 612), (1170, 640), (1280, 648)],
    [(0, 792), (100, 790), (240, 790), (340, 800), (450, 806), (600, 806), (750, 801), (900, 796), (1000, 791), (1120, 790), (1280, 790)],
    [(0, 895), (150, 895), (250, 905), (350, 945), (500, 960), (650, 965), (800, 960), (950, 945), (1030, 912), (1130, 895), (1280, 895)],
]
SHADOW_PAD = 40  # source px around each layer's shadow sprite; the page's SHADOW_PAD must match
NAMES = ["Üst ekmek", "Pastırma & jambon", "Köfte & cheddar", "Kibrit patates", "Alt ekmek"]


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


def band(size, top, bottom) -> Image.Image:
    w, h = size
    m = Image.new("L", size, 0)
    upper = top if top else [(0, 0), (w, 0)]
    lower = bottom if bottom else [(0, h), (w, h)]
    ImageDraw.Draw(m).polygon(upper + lower[::-1], fill=255)
    return m.filter(ImageFilter.GaussianBlur(0.8))


def main(src: str) -> None:
    im = Image.open(src).convert("RGB")
    alpha = burger_mask(im)
    OUT.mkdir(exist_ok=True)
    edges = [None] + CUTS + [None]
    layers = []
    for i in range(len(edges) - 1):
        # each layer runs a few px past its lower cut: drawn over the layer below, the overlap
        # hides the seam when the stack is closed
        lower = edges[i + 1] and [(x, y + 6) for x, y in edges[i + 1]]
        a = ImageChops.multiply(alpha, band(im.size, edges[i], lower))
        rgba = im.copy()
        rgba.putalpha(a)
        box = a.point(lambda v: 255 if v > 8 else 0).getbbox()
        piece = rgba.crop(box)
        piece.save(OUT / f"l{i}.webp", quality=88, method=6)
        # pre-blurred shadow of the layer (canvas shadowBlur is far too slow per frame): the
        # layer's box plus SHADOW_PAD on every side, at half size, drawn scaled up under it
        pad = Image.new("L", (piece.width + 2 * SHADOW_PAD, piece.height + 2 * SHADOW_PAD), 0)
        pad.paste(piece.getchannel("A"), (SHADOW_PAD, SHADOW_PAD))
        pad = pad.resize((pad.width // 2, pad.height // 2), Image.LANCZOS).filter(ImageFilter.GaussianBlur(9))
        shadow = Image.new("RGBA", pad.size, (45, 28, 18, 0))
        shadow.putalpha(pad.point(lambda v: int(v * .6)))
        shadow.save(OUT / f"s{i}.webp", quality=70, method=6)
        layers.append({"name": NAMES[i], "x": box[0], "y": box[1], "w": box[2] - box[0], "h": box[3] - box[1]})
    (OUT / "layers.json").write_text(json.dumps(layers, ensure_ascii=False), encoding="utf-8")
    print(json.dumps([[l["x"], l["y"], l["w"], l["h"]] for l in layers]))


if __name__ == "__main__":
    main(sys.argv[1])
