"""Build img/ from the restaurant's own photos (Trendyol Go product shots, Instagram).

    python tools/make_images.py

Sources live in _src/ (not deployed): _src/tgo/p/*.jpg|png are the 1280 px product photos,
_src/ig/p0.jpg the Instagram profile picture, _src/ig/p4.jpg the tray photo.
"""
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

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


def feathered(im: Image.Image, edges=(0.16, 0.10, 0.16, 0.06)) -> Image.Image:
    """Fade the photo's edges (left, top, right, bottom fractions) to transparent so it
    sits on the film's backdrop without a visible rectangle."""
    w, h = im.size
    a = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(a)
    l, t, r, b = (int(f * s) for f, s in zip(edges, (w, h, w, h)))
    for i in range(l):
        d.line([(i, 0), (i, h)], fill=int(255 * (i / l) ** 1.6))
    for i in range(r):
        d.line([(w - 1 - i, 0), (w - 1 - i, h)], fill=int(255 * (i / r) ** 1.6))
    vert = Image.new("L", (w, h), 255)
    dv = ImageDraw.Draw(vert)
    for i in range(t):
        dv.line([(0, i), (w, i)], fill=int(255 * (i / t) ** 1.6))
    for i in range(b):
        dv.line([(0, h - 1 - i), (w, h - 1 - i)], fill=int(255 * (i / b) ** 1.6))
    rgba = im.copy()
    rgba.putalpha(ImageChops.multiply(a, vert))
    return rgba


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

    # Film poster for reduced motion / no JS: the Gurme Burgeriks photo itself
    save(product("gurme"), "film-poster.webp", 1100, 82)

    # Film, scene 3: the Burgeriks Menü photo with soft edges, drawn under the stacked burger
    save(feathered(product("menuburg")), "film-menu.webp", 1280, 82)

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
