"""Découpeur de panneaux manhwa — Google Colab + Gradio.

Modèle : manhwa_panel_cropped.pt (YOLO, projet Aniflow, licence MIT).

Étapes :
  1. Les pages d'un même chapitre sont triées (ordre naturel des noms) puis
     fusionnées en UNE bande verticale continue (même largeur pour toutes).
     -> un panneau coupé entre deux images redevient entier.
  2. La bande est analysée par fenêtres glissantes qui se chevauchent
     (YOLO ne peut pas lire une bande de 50 000 px d'un coup).
  3. Les détections des fenêtres sont fusionnées (doublons / morceaux).
  4. Les panneaux sont découpés dans l'ordre de lecture et exportés en ZIP.

Envoi : images (png/jpg/webp…) = un chapitre, ou des .zip :
chaque .zip (ou chaque sous-dossier d'un .zip) = un chapitre.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None  # bandes webtoon très hautes

MODEL_PATH = os.environ.get("MANHWA_MODEL", "/content/manhwa_panel_cropped.pt")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(path: str) -> YOLO:
    # torch >= 2.6 charge en weights_only=True, ce qui bloque ce .pt
    orig = torch.load
    torch.load = lambda *a, **k: orig(*a, **{**k, "weights_only": False})
    try:
        return YOLO(path)
    finally:
        torch.load = orig


MODEL = _load_model(MODEL_PATH)


# ───────────────────────── chargement des chapitres ─────────────────────────

def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path))]


def collect_chapters(files: list[str], work: Path) -> dict[str, list[Path]]:
    """Regroupe les fichiers envoyés par chapitre, pages triées."""
    chapters: dict[str, list[Path]] = {}
    loose = []
    for f in files or []:
        p = Path(f)
        if p.suffix.lower() == ".zip":
            dest = work / "zips" / p.stem
            with zipfile.ZipFile(p) as z:
                z.extractall(dest)
            for img in dest.rglob("*"):
                if img.suffix.lower() in IMAGE_EXT and "__MACOSX" not in img.parts:
                    rel = img.parent.relative_to(dest)
                    name = p.stem if str(rel) == "." else f"{p.stem}/{rel}"
                    chapters.setdefault(name, []).append(img)
        elif p.suffix.lower() in IMAGE_EXT:
            loose.append(p)
    if loose:
        chapters["chapitre"] = loose
    return {k: sorted(v, key=lambda x: _natural_key(x.name)) for k, v in sorted(chapters.items())}


def stitch(pages: list[Path]) -> Image.Image:
    """Fusionne les pages en une bande verticale à largeur commune (largeur la plus fréquente)."""
    imgs = [Image.open(p).convert("RGB") for p in pages]
    widths = [im.width for im in imgs]
    width = max(set(widths), key=widths.count)
    resized = [im if im.width == width else
               im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
               for im in imgs]
    strip = Image.new("RGB", (width, sum(im.height for im in resized)), "white")
    y = 0
    for im in resized:
        strip.paste(im, (0, y))
        y += im.height
    return strip


# ───────────────────────── détection par fenêtres ─────────────────────────

def detect_strip(strip: Image.Image, conf: float, imgsz: int, tile_ratio: float,
                 overlap: float, min_side: int) -> list[tuple[int, int, int, int]]:
    """YOLO sur des fenêtres chevauchantes de hauteur tile_ratio × largeur."""
    w, h = strip.size
    tile_h = max(int(w * tile_ratio), 64)
    stride = max(int(tile_h * (1 - overlap)), 1)
    starts = list(range(0, max(h - tile_h, 0) + 1, stride))
    if starts[-1] + tile_h < h:
        starts.append(h - tile_h)

    boxes = []
    for y0 in starts:
        tile = strip.crop((0, y0, w, min(y0 + tile_h, h)))
        res = MODEL.predict(source=tile, classes=[0], conf=conf, imgsz=imgsz,
                            device=DEVICE, half=DEVICE == "cuda", verbose=False)[0]
        for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
            boxes.append((int(x1), int(y1) + y0, int(x2), int(y2) + y0))

    boxes = merge_boxes(boxes)
    boxes = [b for b in boxes if b[2] - b[0] >= min_side and b[3] - b[1] >= min_side]
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def merge_boxes(boxes, contain_thr: float = 0.5):
    """Fusionne les boîtes qui se recouvrent fortement (doublons entre fenêtres
    ou morceaux d'un même grand panneau) : intersection / plus petite aire > seuil."""
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        out = []
        while boxes:
            a = boxes.pop()
            i = 0
            while i < len(boxes):
                b = boxes[i]
                iw = min(a[2], b[2]) - max(a[0], b[0])
                ih = min(a[3], b[3]) - max(a[1], b[1])
                if iw > 0 and ih > 0:
                    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                    if iw * ih / max(small, 1) > contain_thr:
                        a = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                        boxes.pop(i)
                        merged = True
                        continue
                i += 1
            out.append(a)
        boxes = out
    return [tuple(b) for b in boxes]


def draw_preview(strip: Image.Image, boxes, width: int = 500) -> Image.Image:
    scale = width / strip.width
    prev = strip.resize((width, max(1, int(strip.height * scale))), Image.BILINEAR)
    d = ImageDraw.Draw(prev)
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        r = [x1 * scale, y1 * scale, x2 * scale - 1, y2 * scale - 1]
        d.rectangle(r, outline=(0, 220, 0), width=4)
        d.rectangle([r[0], r[1], r[0] + 34, r[1] + 22], fill=(0, 220, 0))
        d.text((r[0] + 5, r[1] + 5), str(i), fill="white")
    return prev


# ───────────────────────── traitement complet ─────────────────────────

def run(files, conf, imgsz, tile_ratio, overlap, min_side, padding, keep_strip,
        progress=gr.Progress()):
    if not files:
        raise gr.Error("Envoie des images ou un .zip de chapitre.")
    work = Path(tempfile.mkdtemp(prefix="panels_"))
    out_root = work / "panels"
    chapters = collect_chapters([getattr(f, "name", f) for f in files], work)
    if not chapters:
        raise gr.Error("Aucune image trouvée.")

    gallery, previews, report = [], [], []
    for ci, (name, pages) in enumerate(chapters.items(), 1):
        progress((ci - 1) / len(chapters), desc=f"{name} : fusion de {len(pages)} pages")
        strip = stitch(pages)
        progress((ci - 0.5) / len(chapters), desc=f"{name} : détection")
        boxes = detect_strip(strip, conf, int(imgsz), tile_ratio, overlap, int(min_side))

        ch_dir = out_root / re.sub(r"[^\w\-]+", "_", name)
        ch_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
            box = (max(0, x1 - padding), max(0, y1 - padding),
                   min(strip.width, x2 + padding), min(strip.height, y2 + padding))
            path = ch_dir / f"panel_{i:04d}.png"
            strip.crop(box).save(path)
            manifest.append({"panel": path.name, "bbox": list(box)})
            gallery.append((str(path), f"{name} #{i}"))

        prev = draw_preview(strip, boxes)
        prev_path = ch_dir / "_apercu.jpg"
        prev.save(prev_path, quality=85)
        previews.append((str(prev_path), name))
        if keep_strip:
            strip.save(ch_dir / "_bande_complete.png")
        (ch_dir / "panels.json").write_text(json.dumps(
            {"chapitre": name, "pages": [p.name for p in pages],
             "taille_bande": strip.size, "panneaux": manifest}, indent=2, ensure_ascii=False))
        report.append(f"**{name}** : {len(pages)} pages → bande {strip.width}×{strip.height}px "
                      f"→ **{len(boxes)} panneaux**")

    zip_path = shutil.make_archive(str(work / "panneaux_manhwa"), "zip", out_root)
    return gallery, previews, zip_path, "\n\n".join(report) + f"\n\n_Device : {DEVICE}_"


with gr.Blocks(title="Découpeur de panneaux manhwa") as demo:
    gr.Markdown("# Découpeur de panneaux manhwa\n"
                "Envoie les pages d'un chapitre (images) ou un/des **.zip** "
                "(1 zip ou 1 sous-dossier = 1 chapitre). Les pages sont fusionnées "
                "en une seule bande puis découpées avec `manhwa_panel_cropped.pt`.")
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(label="Pages ou .zip", file_count="multiple",
                            file_types=["image", ".zip", ".webp"])
            with gr.Accordion("Réglages", open=False):
                conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confiance minimale")
                imgsz = gr.Slider(640, 2048, value=1024, step=32, label="Taille d'inférence YOLO")
                tile_ratio = gr.Slider(10, 30, value=18, step=1,
                                       label="Hauteur de fenêtre (× largeur)")
                overlap = gr.Slider(0.2, 0.7, value=0.5, step=0.05, label="Chevauchement")
                min_side = gr.Slider(10, 300, value=60, step=10, label="Côté minimal (px)")
                padding = gr.Slider(0, 50, value=0, step=1, label="Marge autour du panneau (px)")
                keep_strip = gr.Checkbox(False, label="Inclure la bande fusionnée dans le ZIP")
            btn = gr.Button("Découper", variant="primary")
            report = gr.Markdown()
            zip_out = gr.File(label="Télécharger les panneaux (ZIP)")
        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Panneaux", columns=4, height=600)
            previews = gr.Gallery(label="Aperçu de la bande (cadres verts)", columns=3, height=600)
    btn.click(run, [files, conf, imgsz, tile_ratio, overlap, min_side, padding, keep_strip],
              [gallery, previews, zip_out, report])

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
