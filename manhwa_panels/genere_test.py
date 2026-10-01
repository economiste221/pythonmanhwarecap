"""Génère une bande webtoon factice (800 px de large) pour tester le découpage.

Contenu, de haut en bas :
  1-3 : trois panneaux séparés par des gouttières blanches (cas idéal)
  4-5 : deux panneaux séparés par une gouttière NOIRE (cas limite)
  6   : un dernier panneau après une gouttière blanche
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W = 800
rng = np.random.default_rng(0)


def panel(h: int, color: tuple[int, int, int], label: str) -> Image.Image:
    arr = np.clip(np.array(color) + rng.normal(0, 25, (h, W, 3)), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    d.ellipse([W // 3, h // 4, 2 * W // 3, 3 * h // 4], fill=(240, 200, 170), outline=(0, 0, 0), width=4)
    d.rectangle([20, 20, 300, 70], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    d.text((35, 38), label, fill=(0, 0, 0))
    return img


def gutter(h: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (W, h), color)


blocks = [
    gutter(150, (255, 255, 255)),
    panel(900, (60, 90, 160), "Panneau 1"),
    gutter(300, (255, 255, 255)),
    panel(700, (160, 60, 60), "Panneau 2"),
    gutter(250, (255, 255, 255)),
    panel(1100, (50, 140, 80), "Panneau 3"),
    gutter(300, (255, 255, 255)),
    panel(800, (120, 80, 150), "Panneau 4"),
    gutter(250, (0, 0, 0)),  # gouttière noire : non détectée par le seuil blanc
    panel(800, (170, 130, 40), "Panneau 5"),
    gutter(300, (255, 255, 255)),
    panel(900, (40, 130, 140), "Panneau 6"),
    gutter(150, (255, 255, 255)),
]

strip = Image.new("RGB", (W, sum(b.height for b in blocks)), (255, 255, 255))
y = 0
for b in blocks:
    strip.paste(b, (0, y))
    y += b.height

out = Path(__file__).parent / "exemples"
out.mkdir(exist_ok=True)
strip.save(out / "webtoon_test.png")
print(f"Bande de test : {out / 'webtoon_test.png'} ({strip.width}x{strip.height})")
