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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

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
    """Dossier d'images = 1 chapitre ; dossier de sous-dossiers = 1 chapitre par sous-dossier.
    Le nom du chapitre est le nom du (sous-)dossier, ex. ch0001 — c'est la clé du JSON."""
    chapters = {}
    direct = _images_in(folder, recursive=False)
    if direct:
        chapters[folder.name] = direct
    for sub in sorted((d for d in folder.iterdir() if d.is_dir()), key=lambda d: natural_key(d.name)):
        imgs = _images_in(sub, recursive=True)
        if imgs:
            chapters[sub.name] = imgs
    return chapters


def chapters_from_zip(zip_path: Path, extract_root: Path) -> dict[str, list[Path]]:
    dest = extract_root / zip_path.stem
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    chapters: dict[str, list[Path]] = {}
    for img in _images_in(dest, recursive=True):
        rel = img.parent.relative_to(dest)
        name = zip_path.stem if str(rel) == "." else rel.name
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


def safe_name(name: str) -> str:
    """Clé de chapitre utilisée dans strip_data.json et comme nom de dossier."""
    return re.sub(r"[^\w\-]+", "_", name).strip("_") or "chapitre"


def to_xywh(boxes: list[Box]) -> list[list[int]]:
    return [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in boxes]


def sort_boxes(boxes: list[list[int]]) -> list[list[int]]:
    """Ordre de lecture : haut -> bas, puis gauche -> droite."""
    return sorted((list(map(int, b)) for b in boxes), key=lambda b: (b[1], b[0]))


# ───────────────────────── strip_data.json ─────────────────────────
# Format : {"ch0001": [[x, y, largeur, hauteur], ...], ...}
# coordonnées en pixels dans la bande fusionnée du chapitre.

def load_strip_data(path: Path) -> dict[str, list[list[int]]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: sort_boxes(v) for k, v in data.items()}


def save_strip_data(path: Path, data: dict[str, list[list[int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {k: sort_boxes(data[k]) for k in sorted(data, key=natural_key)}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ordered, indent=4), encoding="utf-8")
    tmp.replace(path)


# ───────────────────────── traitement d'un chapitre ─────────────────────────

def detect_chapter(model, device: str, name: str, pages: list[Path], s: Settings,
                   on_step: Callable[[str], None] = lambda m: None,
                   on_tile: Callable[[int, int], None] | None = None):
    """Fusion + détection. Retourne (bande, boîtes [x, y, w, h])."""
    pages = sort_pages(pages)
    on_step(f"{name} : fusion de {len(pages)} pages…")
    strip = stitch(pages)
    on_step(f"{name} : détection ({strip.width}×{strip.height}px)…")
    return strip, to_xywh(detect_strip(model, strip, s, device, on_tile))


def export_chapter(strip: Image.Image, boxes: list[list[int]], name: str, pages: list[Path],
                   out_root: Path, padding: int = 0, keep_strip: bool = False) -> list[Path]:
    """Écrit panel_0001.png… (ordre de lecture) + panels.json dans out_root/<clé>/."""
    ch_dir = out_root / safe_name(name)
    ch_dir.mkdir(parents=True, exist_ok=True)
    for old in list(ch_dir.glob("panel_*.png")) + list(ch_dir.glob("_apercu_*.jpg")):
        old.unlink()

    panels, manifest = [], []
    for i, (x, y, w, h) in enumerate(sort_boxes(boxes), 1):
        box = (max(0, x - padding), max(0, y - padding),
               min(strip.width, x + w + padding), min(strip.height, y + h + padding))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        path = ch_dir / f"panel_{i:04d}.png"
        strip.crop(box).save(path)
        panels.append(path)
        manifest.append({"panel": path.name, "bbox_xywh": [x, y, w, h]})
    if keep_strip:
        strip.save(ch_dir / "_bande_complete.png")
    (ch_dir / "panels.json").write_text(json.dumps(
        {"chapitre": name, "cle": safe_name(name), "pages": [p.name for p in sort_pages(pages)],
         "taille_bande": list(strip.size), "panneaux": manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return panels
