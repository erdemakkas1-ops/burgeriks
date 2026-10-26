"""Build img/ from the restaurant's own photos (Trendyol Go product shots, Instagram).

    python tools/make_images.py

Sources live in _src/ (not deployed): _src/tgo/p/*.jpg|png are the 1280 px product photos,
_src/ig/p0.jpg the Instagram profile picture, _src/ig/p4.jpg the tray photo. The film's frames
come from tools/make_film.py.
"""
from pathlib import Path

from PIL import Image, ImageDraw

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
