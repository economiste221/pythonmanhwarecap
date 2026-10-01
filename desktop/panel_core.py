"""Moteur de découpage de panneaux manhwa (sans interface).

Modèle : manhwa_panel_cropped.pt (YOLO, projet Aniflow, licence MIT).

  1. Les pages d'un chapitre sont triées (ordre naturel) et fusionnées en UNE
     bande verticale à largeur commune -> un panneau coupé entre deux images
     redevient entier.
  2. La bande est analysée par fenêtres glissantes très hautes (le modèle a été
     entraîné sur de longues bandes : fenêtre ≈ 18 × largeur).
  3. Les détections des fenêtres sont fusionnées, puis les panneaux découpés.
"""

from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None  # bandes webtoon très hautes

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MODEL_NAME = "manhwa_panel_cropped.pt"
MODEL_URL = ("https://raw.githubusercontent.com/ghozali25/Aniflow/main/models/" + MODEL_NAME)
_HERE = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    _HERE / "models" / MODEL_NAME,
    _HERE.parent / "manhwa_panels" / "models" / MODEL_NAME,
    Path.home() / ".cache" / "manhwa_panels" / MODEL_NAME,
]

Box = tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class Settings:
    conf: float = 0.25        # confiance minimale
    imgsz: int = 1024         # taille d'inférence YOLO
    tile_ratio: float = 18.0  # hauteur de fenêtre = tile_ratio × largeur
    overlap: float = 0.5      # chevauchement des fenêtres
    min_side: int = 60        # côté minimal d'un panneau (px)
    padding: int = 0          # marge ajoutée autour de chaque panneau (px)
    keep_strip: bool = False  # enregistrer aussi la bande fusionnée


# ───────────────────────── modèle ─────────────────────────

def best_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"   # Mac Apple Silicon (M1–M4)
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def find_or_download_model(log: Callable[[str], None] = print) -> Path:
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    dest = MODEL_CANDIDATES[-1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"Téléchargement du modèle ({MODEL_URL})…")
    urllib.request.urlretrieve(MODEL_URL, dest)
    return dest


def load_model(path: Path):
    import torch
    from ultralytics import YOLO

    # torch >= 2.6 charge en weights_only=True, ce qui bloque ce .pt
    orig = torch.load
    torch.load = lambda *a, **k: orig(*a, **{**k, "weights_only": False})
    try:
        return YOLO(str(path))
    finally:
        torch.load = orig


# ───────────────────────── chapitres ─────────────────────────

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _images_in(folder: Path, recursive: bool) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    return [p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXT
            and "__MACOSX" not in p.parts and not p.name.startswith("._")]


def chapters_from_folder(folder: Path) -> dict[str, list[Path]]:
    """Dossier d'images = 1 chapitre ; dossier de sous-dossiers = 1 chapitre par sous-dossier."""
    chapters = {}
    direct = _images_in(folder, recursive=False)
    if direct:
        chapters[folder.name] = direct
    for sub in sorted((d for d in folder.iterdir() if d.is_dir()), key=lambda d: natural_key(d.name)):
        imgs = _images_in(sub, recursive=True)
        if imgs:
            chapters[f"{folder.name}/{sub.name}"] = imgs
    return chapters


def chapters_from_zip(zip_path: Path, extract_root: Path) -> dict[str, list[Path]]:
    dest = extract_root / zip_path.stem
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    chapters: dict[str, list[Path]] = {}
    for img in _images_in(dest, recursive=True):
        rel = img.parent.relative_to(dest)
        name = zip_path.stem if str(rel) == "." else f"{zip_path.stem}/{rel.as_posix()}"
        chapters.setdefault(name, []).append(img)
    return chapters


def sort_pages(pages: list[Path]) -> list[Path]:
    return sorted(pages, key=lambda p: natural_key(p.name))


# ───────────────────────── fusion + détection ─────────────────────────

def stitch(pages: list[Path]) -> Image.Image:
    """Fusionne les pages en une bande verticale à la largeur la plus fréquente."""
    sizes = []
    for p in pages:
        with Image.open(p) as im:
            sizes.append(im.size)
    widths = [w for w, _ in sizes]
    width = max(set(widths), key=widths.count)
    heights = [round(h * width / w) for w, h in sizes]
    strip = Image.new("RGB", (width, sum(heights)), "white")
    y = 0
    for p, (w, _), h in zip(pages, sizes, heights):
        with Image.open(p) as im:
            im = im.convert("RGB")
            if w != width:
                im = im.resize((width, h), Image.LANCZOS)
            strip.paste(im, (0, y))
        y += h
    return strip


def detect_strip(model, strip: Image.Image, s: Settings, device: str,
                 on_tile: Callable[[int, int], None] | None = None) -> list[Box]:
    """YOLO sur des fenêtres chevauchantes de hauteur tile_ratio × largeur."""
    w, h = strip.size
    tile_h = max(int(w * s.tile_ratio), 64)
    stride = max(int(tile_h * (1 - s.overlap)), 1)
    starts = list(range(0, max(h - tile_h, 0) + 1, stride))
    if starts[-1] + tile_h < h:
        starts.append(h - tile_h)

    boxes: list[Box] = []
    for k, y0 in enumerate(starts, 1):
        tile = strip.crop((0, y0, w, min(y0 + tile_h, h)))
        res = model.predict(source=tile, classes=[0], conf=s.conf, imgsz=s.imgsz,
                            device=device, half=device == "cuda", verbose=False)[0]
        for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
            boxes.append((int(x1), int(y1) + y0, int(x2), int(y2) + y0))
        if on_tile:
            on_tile(k, len(starts))

    boxes = merge_boxes(boxes)
    boxes = [b for b in boxes if b[2] - b[0] >= s.min_side and b[3] - b[1] >= s.min_side]
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def merge_boxes(boxes: list[Box], contain_thr: float = 0.5) -> list[Box]:
    """Fusionne les boîtes qui se recouvrent fortement (doublons entre fenêtres
    ou morceaux d'un même grand panneau) : intersection / plus petite aire > seuil."""
    work = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        out = []
        while work:
            a = work.pop()
            i = 0
            while i < len(work):
                b = work[i]
                iw = min(a[2], b[2]) - max(a[0], b[0])
                ih = min(a[3], b[3]) - max(a[1], b[1])
                if iw > 0 and ih > 0:
                    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                    if iw * ih / max(small, 1) > contain_thr:
                        a = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                        work.pop(i)
                        merged = True
                        continue
                i += 1
            out.append(a)
        work = out
    return [tuple(b) for b in work]


def draw_preview(strip: Image.Image, boxes: list[Box], width: int = 500,
                 part_h: int = 8000) -> list[Image.Image]:
    """Aperçu réduit avec cadres verts, en morceaux de part_h px max
    (le JPEG est limité à 65 535 px de haut)."""
    width = min(width, strip.width)
    scale = width / strip.width
    prev = strip.resize((width, max(1, int(strip.height * scale))), Image.BILINEAR)
    d = ImageDraw.Draw(prev)
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        r = [x1 * scale, y1 * scale, x2 * scale - 1, y2 * scale - 1]
        d.rectangle(r, outline=(0, 220, 0), width=4)
        d.rectangle([r[0], r[1], r[0] + 34, r[1] + 22], fill=(0, 220, 0))
        d.text((r[0] + 5, r[1] + 5), str(i), fill="white")
    return [prev.crop((0, y, width, min(y + part_h, prev.height)))
            for y in range(0, prev.height, part_h)]


def safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_") or "chapitre"


def process_chapter(model, device: str, name: str, pages: list[Path], out_root: Path,
                    s: Settings, on_step: Callable[[str], None] = lambda m: None,
                    on_tile: Callable[[int, int], None] | None = None) -> dict:
    """Fusion + détection + export d'un chapitre. Retourne un résumé."""
    pages = sort_pages(pages)
    on_step(f"{name} : fusion de {len(pages)} pages…")
    strip = stitch(pages)
    on_step(f"{name} : détection ({strip.width}×{strip.height}px)…")
    boxes = detect_strip(model, strip, s, device, on_tile)

    ch_dir = out_root / safe_name(name)
    ch_dir.mkdir(parents=True, exist_ok=True)
    for old in ch_dir.glob("panel_*.png"):
        old.unlink()
    for old in ch_dir.glob("_apercu_*.jpg"):
        old.unlink()

    on_step(f"{name} : export de {len(boxes)} panneaux…")
    panels, manifest = [], []
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        box = (max(0, x1 - s.padding), max(0, y1 - s.padding),
               min(strip.width, x2 + s.padding), min(strip.height, y2 + s.padding))
        path = ch_dir / f"panel_{i:04d}.png"
        strip.crop(box).save(path)
        panels.append(path)
        manifest.append({"panel": path.name, "bbox": list(box)})

    previews = []
    for k, part in enumerate(draw_preview(strip, boxes), 1):
        p = ch_dir / f"_apercu_{k:02d}.jpg"
        part.save(p, quality=85)
        previews.append(p)
    if s.keep_strip:
        strip.save(ch_dir / "_bande_complete.png")

    (ch_dir / "panels.json").write_text(json.dumps(
        {"chapitre": name, "pages": [p.name for p in pages], "taille_bande": list(strip.size),
         "reglages": asdict(s), "panneaux": manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return {"name": name, "dir": ch_dir, "pages": len(pages), "size": strip.size,
            "panels": panels, "previews": previews}
