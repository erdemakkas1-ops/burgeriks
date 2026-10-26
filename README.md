# Burgeriks — scroll film demo

Static one-page demo for Burgeriks (Ufuktepe / Keçiören, Ankara), built on the same reference
as the Oha Burger demo: the hero is a film that plays as you scroll. The Gurme Burgeriks lifts
apart into labelled layers and settles back. Below it: the lineup ("Tezgâh"), the full menu,
story, reviews and location. No build step: `index.html` + `img/` + `frames/` is the whole site.

Live: https://erdemakkas1-ops.github.io/burgeriks/ (GitHub Pages, `gh-pages` branch). A copy also
runs on Vercel at https://burgeriks.vercel.app, but some Turkish connections reset `*.vercel.app`,
so canonical and share URLs point at GitHub Pages.

## The film

- Same engine as the Oha Burger site: `frames/d/` (1280×720) and `frames/m/` (768×432, phones) are a
  frame sequence drawn to a canvas and scrubbed by page scroll, streamed in coarse-to-fine. On
  portrait screens the 16:9 frame is a band at the top, zoomed in on the burger.
- No video clip: `tools/make_film.py` renders the frames from two Gemini renders of the Gurme
  Burgeriks (assembled, and exploded on flat grey), cut into six ingredient layers by
  `tools/cut_layers.py`. The burger stands assembled, the ingredients lift off one after another
  from the top, hang apart for the labels, and settle back from the bottom up.
- In `index.html`, the film's `data-map` maps scroll progress → film seconds; scene
  `data-from`/`data-to` and the labels' `data-keys` are film seconds. `make_film.py` prints
  `data-frames` and each label's `data-x`/`data-y` (percent of the frame); paste them in after a
  re-render.
- `vercel.json` caches `frames/` as immutable: after re-rendering, bump the film's `data-v` and the
  `?v=` on the two frame preloads and the poster.
- `prefers-reduced-motion` (and no JS) shows the first frame as a still hero.

Rebuilding (Python with Pillow, NumPy and OpenCV; the sources in `_src/` are not in the repo):

```bash
python tools/cut_layers.py _src/gemini/exploded.webp _src/gemini/closed.webp
python tools/make_film.py
python tools/make_images.py
```

The exploded render must keep a clear gap between ingredients (each layer is one connected piece).
Bun, meat, onions and patty are found in the closed render by colour; the fries and the bottom
bun are placed by rule (they look different pressed together).

## Content sources

- Menu, prices and product photos: Burgeriks' Trendyol Go page (October 2026). Hours and the two
  quoted reviews: Yemeksepeti. Address, phone and Maps link: the Instagram bio (@burgeriks).
- Film burger: Gemini renders made from the restaurant's Gurme Burgeriks photo (`_src/gemini/`).
- Logo icons: upscaled from the 150 px Instagram profile picture; the wordmark is redrawn in CSS after it.
- The story photo is from the restaurant's Instagram (640 px; ask them for the original).

## Before going live

- Confirm menu, prices, hours and payment options with the restaurant; Yemeksepeti prices differ.
- Get permission to use the product photos and higher-resolution originals.
