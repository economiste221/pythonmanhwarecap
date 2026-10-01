"""Découpage autonome des panneaux de manhwa / webtoon.

Extrait du projet ACMP (vc-tr/acmp, acmp/panels/detector.py) sans le reste
du pipeline. Seules dépendances : opencv-python-headless, numpy, Pillow.

Méthode (identique à ACMP) :
  1. Bande verticale : on repère les lignes horizontales quasi blanches
     (moyenne de gris > seuil) = gouttières, et on coupe entre elles.
  2. Si on trouve <= 1 panneau, repli sur la détection par contours OpenCV.

Usage :
    python decoupe_panels.py chemin/image_ou_dossier -o sortie/
    python decoupe_panels.py chapitre/ -o sortie/ --seuil 230
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

Box = tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class PanelConfig:
    min_area_ratio: float = 0.01   # aire minimale d'un panneau (fraction de la page)
    max_area_ratio: float = 0.95   # aire maximale d'un panneau
    padding: int = 5               # marge autour des panneaux (méthode contours)
    white_threshold: int = 240     # luminosité moyenne au-delà de laquelle une ligne = gouttière


def _gray(page: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(page.convert("RGB")), cv2.COLOR_RGB2GRAY)


def detect_panels(page: Image.Image, config: PanelConfig | None = None) -> list[Box]:
    """Détection par contours (seuillage adaptatif + contours externes)."""
    config = config or PanelConfig()
    gray = _gray(page)
    h, w = gray.shape
    page_area = h * w
    min_area = page_area * config.min_area_ratio
    max_area = page_area * config.max_area_ratio

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panels = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if area < min_area or area > max_area:
            continue
        if min(cw, ch) / max(cw, ch) < 0.1:
            continue
        pad = config.padding
        x = max(0, x - pad)
        y = max(0, y - pad)
        cw = min(w - x, cw + 2 * pad)
        ch = min(h - y, ch + 2 * pad)
        panels.append((x, y, cw, ch))

    return _remove_overlapping(panels)


def detect_panels_vertical_scroll(page: Image.Image, config: PanelConfig | None = None) -> list[Box]:
    """Détection manhwa : coupe la bande sur les gouttières horizontales blanches."""
    config = config or PanelConfig()
    gray = _gray(page)
    h, w = gray.shape

    is_gutter = np.mean(gray, axis=1) > config.white_threshold
    min_h = h * config.min_area_ratio * 2
    panels = []
    panel_start = None

    for y_pos in range(h):
        if not is_gutter[y_pos]:
            if panel_start is None:
                panel_start = y_pos
        elif panel_start is not None:
            if y_pos - panel_start > min_h:
                panels.append((0, panel_start, w, y_pos - panel_start))
            panel_start = None

    if panel_start is not None and h - panel_start > min_h:
        panels.append((0, panel_start, w, h - panel_start))

    if len(panels) <= 1:
        return detect_panels(page, config)
    return panels


def _remove_overlapping(panels: list[Box], iou_threshold: float = 0.5) -> list[Box]:
    keep: list[Box] = []
    for panel in sorted(panels, key=lambda p: p[2] * p[3], reverse=True):
        if all(_compute_iou(panel, kept) <= iou_threshold for kept in keep):
            keep.append(panel)
    return keep


def _compute_iou(box1: Box, box2: Box) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def draw_preview(page: Image.Image, boxes: list[Box], max_height: int = 4000) -> Image.Image:
    """Image de contrôle : rectangles rouges numérotés sur la page."""
    preview = page.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    line = max(3, page.width // 200)
    for i, (x, y, w, h) in enumerate(boxes, 1):
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(255, 0, 0), width=line)
        draw.rectangle([x, y, x + 60, y + 40], fill=(255, 0, 0))
        draw.text((x + 10, y + 10), str(i), fill=(255, 255, 255))
    if preview.height > max_height:
        ratio = max_height / preview.height
        preview = preview.resize((max(1, int(preview.width * ratio)), max_height))
    return preview


def process_image(path: Path, out_dir: Path, config: PanelConfig) -> list[Box]:
    page = Image.open(path).convert("RGB")
    boxes = sorted(detect_panels_vertical_scroll(page, config), key=lambda b: b[1])

    page_dir = out_dir / path.stem
    page_dir.mkdir(parents=True, exist_ok=True)
    for i, (x, y, w, h) in enumerate(boxes, 1):
        page.crop((x, y, x + w, y + h)).save(page_dir / f"panel_{i:03d}.png")
    draw_preview(page, boxes).save(page_dir / "_apercu.jpg", quality=85)

    print(f"{path.name} ({page.width}x{page.height}) -> {len(boxes)} panneaux")
    for i, (x, y, w, h) in enumerate(boxes, 1):
        print(f"   #{i:02d}  y={y:>6}  h={h:>5}  x={x:>4}  w={w:>4}")
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description="Découpe les panneaux d'un manhwa.")
    parser.add_argument("entree", type=Path, help="Image ou dossier d'images")
    parser.add_argument("-o", "--sortie", type=Path, default=Path("sortie_panels"))
    parser.add_argument("--seuil", type=int, default=240,
                        help="Seuil de blanc des gouttières (0-255, défaut 240)")
    parser.add_argument("--min-area", type=float, default=0.01,
                        help="Taille minimale relative d'un panneau (défaut 0.01)")
    args = parser.parse_args()

    config = PanelConfig(white_threshold=args.seuil, min_area_ratio=args.min_area)
    if args.entree.is_dir():
        files = sorted(p for p in args.entree.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        files = [args.entree]
    if not files:
        raise SystemExit(f"Aucune image trouvée dans {args.entree}")

    total = sum(len(process_image(f, args.sortie, config)) for f in files)
    print(f"\nTotal : {total} panneaux -> {args.sortie.resolve()}")


if __name__ == "__main__":
    main()
