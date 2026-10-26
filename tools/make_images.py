"""Build img/ from the restaurant's own photos (Trendyol Go product shots, Instagram).

    python tools/make_images.py

Sources live in _src/ (not deployed): _src/tgo/p/*.jpg|png are the 1280 px product photos,
_src/ig/p0.jpg the Instagram profile picture, _src/ig/p4.jpg the tray photo, and
_src/gemini/closed.webp the assembled burger render (for the poster).
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cut_layers import clean, fill_from
from cut_layers import food_mask as burger_mask, rgba as burger_rgba

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "_src"
OUT = ROOT / "img"


def product(name: str) -> Image.Image:
    p = next((SRC / "tgo" / "p").glob(name + ".*"))
    return Image.open(p).convert("RGB")


def save(im: Image.Image, name: str, size: int, quality: int = 80) -> None:
    im = im.copy()
    im.thumbnail((size, size), Image.LANCZOS)
    im.save(OUT / name, quality=quality, method=6)
    print(name, im.size)


def food_mask(im: Image.Image, grey_boxes=(), min_area=6000) -> np.ndarray:
    """Everything with colour on the grey studio sweep (several items, unlike the single
    burger in cut_layers): closed, specks and the grey table shadows dropped, holes filled.
    grey_boxes (x0, y0, x1, y1) also take colourless parts darker than the sweep, like a
    can's metal rim."""
    rgb = np.asarray(im).astype(int)
    chroma = rgb.max(2) - rgb.min(2)
    m = (chroma > 30) | (rgb.max(2) < 60)
    keep_grey = np.zeros(m.shape, bool)
    for x0, y0, x1, y1 in grey_boxes:
        keep_grey[y0:y1, x0:x1] = rgb[y0:y1, x0:x1].max(2) < 185
    m |= keep_grey
    m = clean(m, close=9, min_area=min_area)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = clean(m.astype(bool), close=15, min_area=min_area)
    # closing bridges into the grey table between items; peel off colourless pixels that
    # reach the outside (the same pixels inside the food come back as filled holes)
    neutral = (chroma < .32 * rgb.max(2)) & (rgb.max(2) > 70) & ~keep_grey  # table and its warm shadows
    return clean(m & ~neutral, close=5, min_area=min_area)


def cutout(im: Image.Image, mask: np.ndarray, side_fade=0) -> Image.Image:
    """The food on transparency, its grey-studio fringe repainted from just inside the edge.
    side_fade px fade out things the photo's frame cuts through at the left and right."""
    rgb = np.asarray(im)
    core = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    col = fill_from(rgb, core, mask & ~core)
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)
    if side_fade:
        x = np.arange(a.shape[1])
        ramp = np.clip(np.minimum(x, a.shape[1] - 1 - x) / side_fade, 0, 1)
        a *= (ramp * ramp * (3 - 2 * ramp))[None, :]
    return Image.fromarray(np.dstack([np.clip(col, 0, 255), a * 255]).astype(np.uint8), "RGBA")


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # Lineup cards and menu tab photos: 4:3 crops, burger centred
    for name, out in [
        ("gurme", "card-gurme"), ("dcheese", "card-double"), ("alaturka", "card-alaturka"),
        ("chtruff", "card-truff"), ("pilic", "card-pilic"), ("menuburg", "card-menu"),
        ("classic", "tab-classic"), ("chicken", "tab-chicken"), ("menu3", "tab-menu"), ("halka", "tab-halka"),
    ]:
        im = product(name)
        im = im.crop((0, 110, 1280, 1070))  # 4:3
        save(im, out + ".webp", 960)

    # Film poster for reduced motion / no JS: the assembled Gurme Burgeriks render, cut out
    closed = np.asarray(Image.open(SRC / "gemini" / "closed.webp").convert("RGB"))
    m = burger_mask(closed)
    ys, xs = np.nonzero(m)
    poster = Image.fromarray(burger_rgba(closed, m)).crop((xs.min() - 8, ys.min() - 8, xs.max() + 9, ys.max() + 9))
    save(poster, "film-poster.webp", 1100, 86)

    # Film, scene 3: the Burgeriks Menü photo cut out, drawn under the stacked burger
    menu = product("menuburg")
    save(cutout(menu, food_mask(menu, grey_boxes=[(1045, 235, 1280, 300)]), side_fade=170), "film-menu.webp", 1280, 82)

    # Story section: the tray as served, from Instagram
    tray = Image.open(SRC / "ig" / "p4.jpg").convert("RGB")
    save(tray, "story-tray.webp", 1200, 82)

    # Icons from the Instagram profile picture (150 px)
    logo = Image.open(SRC / "ig" / "p0.jpg").convert("RGB")
    mask = Image.new("L", logo.size, 0)
    ImageDraw.Draw(mask).ellipse((1, 1, logo.width - 2, logo.height - 2), fill=255)
    for size, name in [(32, "favicon-32.png"), (180, "apple-touch-icon.png")]:
        im = logo.resize((size, size), Image.LANCZOS)
        if size == 32:
            im = im.convert("RGBA")
            im.putalpha(mask.resize((size, size), Image.LANCZOS))
        im.save(OUT / name)
        print(name, im.size)

    # Share image: the menu photo, letterboxed to 1200 x 630
    og = product("menuburg").resize((1200, 1200), Image.LANCZOS).crop((0, 300, 1200, 930))
    og.save(OUT / "og.jpg", quality=84)
    print("og.jpg", og.size)


if __name__ == "__main__":
    main()
