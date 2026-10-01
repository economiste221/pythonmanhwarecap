"""Découpage de panneaux manhwa avec les modèles YOLO d'Aniflow.

Extrait de ghozali25/Aniflow (pipeline/panel_extractor_yolo.py, licence MIT),
sans le reste du projet. Dépendances : ultralytics (+ torch), opencv, Pillow.

Deux modèles (dossier models/) :
  bulles  : panel_with_bubble_text.pt — panneau AVEC ses bulles de dialogue (défaut)
  panel   : manhwa_panel_cropped.pt   — dessin seul (classes: panel, speech_bubble, watermark)

Sur Mac Apple Silicon (M1-M4), l'inférence passe automatiquement par MPS.

Usage :
    python decoupe_yolo.py chapitre/ -o resultats_yolo
    python decoupe_yolo.py page.webp --modele panel --conf 0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

MODELS_DIR = Path(__file__).parent / "models"
MODELS = {
    "bulles": MODELS_DIR / "panel_with_bubble_text.pt",
    "panel": MODELS_DIR / "manhwa_panel_cropped.pt",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

Image.MAX_IMAGE_PIXELS = 500_000_000  # les bandes webtoon peuvent être très hautes


def best_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(path: Path):
    import torch
    from ultralytics import YOLO

    # torch >= 2.6 charge en weights_only=True, ce qui bloque ces .pt
    orig_load = torch.load
    torch.load = lambda *a, **k: orig_load(*a, **{**k, "weights_only": False})
    try:
        return YOLO(str(path))
    finally:
        torch.load = orig_load


def detect(model, page: Image.Image, conf: float, imgsz: int, device: str,
           min_side: int, iou: float = 0.3) -> list[tuple[int, int, int, int]]:
    """Retourne les boîtes (x1, y1, x2, y2) de classe 0, triées en ordre de lecture."""
    result = model.predict(source=page, classes=[0], conf=conf, imgsz=imgsz,
                           device=device, verbose=False)[0]
    boxes = [tuple(map(int, b)) for b in result.boxes.xyxy.cpu().numpy()]
    boxes = [b for b in boxes if b[2] - b[0] >= min_side and b[3] - b[1] >= min_side]
    return sorted(_suppress_overlaps(boxes, iou), key=lambda b: (b[1], b[0]))


def _suppress_overlaps(boxes, iou_threshold):
    """Supprime les doublons : garde la plus grande boîte quand IoU > seuil."""
    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    kept = []
    for b in sorted(boxes, key=area, reverse=True):
        ok = True
        for k in kept:
            iw = min(b[2], k[2]) - max(b[0], k[0])
            ih = min(b[3], k[3]) - max(b[1], k[1])
            if iw > 0 and ih > 0:
                inter = iw * ih
                if inter / max(area(b) + area(k) - inter, 1) > iou_threshold:
                    ok = False
                    break
        if ok:
            kept.append(b)
    return kept


def draw_preview(page: Image.Image, boxes, max_height: int = 4000) -> Image.Image:
    preview = page.copy()
    draw = ImageDraw.Draw(preview)
    line = max(3, page.width // 150)
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(0, 200, 0), width=line)
        draw.rectangle([x1, y1, x1 + 60, y1 + 40], fill=(0, 200, 0))
        draw.text((x1 + 10, y1 + 10), str(i), fill=(255, 255, 255))
    if preview.height > max_height:
        ratio = max_height / preview.height
        preview = preview.resize((max(1, int(preview.width * ratio)), max_height))
    return preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Découpe les panneaux d'un manhwa (YOLO).")
    parser.add_argument("entree", type=Path, help="Image ou dossier d'images")
    parser.add_argument("-o", "--sortie", type=Path, default=Path("resultats_yolo"))
    parser.add_argument("--modele", choices=MODELS, default="bulles")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance (défaut 0.25)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Taille d'inférence (défaut 1024)")
    parser.add_argument("--min-side", type=int, default=60,
                        help="Côté minimal d'un panneau en px (défaut 60)")
    parser.add_argument("--device", default=None, help="mps | cuda | cpu (auto par défaut)")
    args = parser.parse_args()

    if args.entree.is_dir():
        files = sorted(p for p in args.entree.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        files = [args.entree]
    if not files:
        raise SystemExit(f"Aucune image trouvée dans {args.entree}")

    device = args.device or best_device()
    model = load_model(MODELS[args.modele])
    print(f"Modèle : {args.modele}  |  device : {device}\n")

    total = 0
    for path in files:
        page = Image.open(path).convert("RGB")
        boxes = detect(model, page, args.conf, args.imgsz, device, args.min_side)
        page_dir = args.sortie / path.stem
        page_dir.mkdir(parents=True, exist_ok=True)
        for i, box in enumerate(boxes, 1):
            page.crop(box).save(page_dir / f"panel_{i:03d}.png")
        draw_preview(page, boxes).save(page_dir / "_apercu.jpg", quality=85)

        print(f"{path.name} ({page.width}x{page.height}) -> {len(boxes)} panneaux")
        for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
            print(f"   #{i:02d}  y={y1:>6}-{y2:<6}  x={x1:>4}-{x2:<4}")
        total += len(boxes)

    print(f"\nTotal : {total} panneaux -> {args.sortie.resolve()}")


if __name__ == "__main__":
    main()
