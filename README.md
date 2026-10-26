# Burgeriks — scroll film demo

Static one-page demo for Burgeriks (Ufuktepe / Keçiören, Ankara), built on the same reference
as the Oha Burger demo: the hero is a film that plays as you scroll. The Gurme Burgeriks lifts
apart into labelled layers, stacks back, drops into the Burgeriks Menü photo, and the lights go
out on the neon sign. Below it: the lineup ("Tezgâh"), the full menu, story, reviews and location.
No build step: `index.html` + `img/` + `layers/` is the whole site.

Live: https://erdemakkas1-ops.github.io/burgeriks/ (GitHub Pages, `gh-pages` branch). A copy also
runs on Vercel at https://burgeriks.vercel.app, but some Turkish connections reset `*.vercel.app`,
so canonical and share URLs point at GitHub Pages.

## The film

- Unlike Oha's frame sequence, nothing is pre-rendered: `layers/l0–l4.webp` are the restaurant's
  Gurme Burgeriks photo cut into five layers, drawn live on a canvas and moved by page scroll
  (~200 KB in total, sharp at any size). `layers/s0–s4.webp` are their baked drop shadows.
- In `index.html`, `data-map` on the film maps scroll progress → film time (0–9); scene
  `data-from`/`data-to` are film time. `layout()` in the script decides where the burger sits
  and when it separates (1.2–2.7), shows labels (2.6–4.7), drops into the menu photo (5.0–6.5)
  and when the lights go out (7.0+).
- `vercel.json` caches `layers/` as immutable: after re-cutting, bump the film's `data-v` and the
  `?v=` on the layer preload.
- `prefers-reduced-motion` (and no JS) shows a still hero with the photo instead of the pinned film.

Rebuilding (Python with Pillow; the source photos in `_src/` are not in the repo):

```bash
python tools/cut_layers.py _src/tgo/p/gurme.jpg   # prints the boxes for data-boxes
python tools/make_images.py
```

The cut lines between layers are hand-placed in `CUTS` in `tools/cut_layers.py`.

## Content sources

- Menu, prices and product photos: Burgeriks' Trendyol Go page (October 2026). Hours and the two
  quoted reviews: Yemeksepeti. Address, phone and Maps link: the Instagram bio (@burgeriks).
- Logo icons: upscaled from the 150 px Instagram profile picture; the neon sign and wordmark are
  redrawn in SVG/CSS after it.
- The story photo is from the restaurant's Instagram (640 px; ask them for the original).

## Before going live

- Confirm menu, prices, hours and payment options with the restaurant; Yemeksepeti prices differ.
- Get permission to use the product photos and higher-resolution originals.
