"""Découpeur de panneaux manhwa — application de bureau PyQt5.

1. Ajouter des chapitres (images, dossiers, zip)  2. Détecter (YOLO)
3. Corriger dans l'éditeur  4. Les panneaux + strip_data.json sont enregistrés.

Lancement :  python app_qt.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

from PyQt5.QtCore import QSize, Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QIcon, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QShortcut, QSpinBox, QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

import panel_core as core
from editor_qt import PanelEditor

STACK_SIZE = 64 * 1024 * 1024  # macOS : un QThread n'a que 512 Ko de pile, trop peu pour torch


class Worker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)          # valeur, maximum
    chapter_done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    _model = None
    _device = None

    def __init__(self, chapters: dict[str, list[Path]], out_root: Path, settings: core.Settings):
        super().__init__()
        self.chapters = chapters
        self.out_root = out_root
        self.settings = settings
        self._stop = False
        self.setStackSize(STACK_SIZE)

    def stop(self):
        self._stop = True

    def run(self):
        try:
            if Worker._model is None:
                self.log.emit("Chargement du modèle YOLO…")
                path = core.find_or_download_model(self.log.emit)
                Worker._model = core.load_model(path)
                Worker._device = core.best_device()
                self.log.emit(f"Modèle prêt ({path.name}) — calcul sur : {Worker._device.upper()}")

            n = len(self.chapters)
            s = self.settings
            for ci, (name, pages) in enumerate(self.chapters.items()):
                if self._stop:
                    break

                def on_tile(k, total, ci=ci):
                    if self._stop:
                        raise InterruptedError
                    self.progress.emit(ci * 1000 + int(1000 * k / total), n * 1000)

                try:
                    strip, boxes = core.detect_chapter(Worker._model, Worker._device, name, pages,
                                                       s, on_step=self.log.emit, on_tile=on_tile)
                except InterruptedError:
                    break
                self.log.emit(f"{name} : export de {len(boxes)} panneaux…")
                panels = core.export_chapter(strip, boxes, name, pages, self.out_root,
                                             s.padding, s.keep_strip)
                self.log.emit(f"✔ {name} : {len(pages)} pages → {len(boxes)} panneaux")
                self.chapter_done.emit({"name": name, "boxes": boxes, "panels": panels})
            self.log.emit("Arrêté." if self._stop else "Détection terminée — corrige dans l'onglet Éditeur.")
        except Exception as e:  # noqa: BLE001 - toute erreur doit remonter à l'interface
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__} : {e}")


class ChapterTree(QTreeWidget):
    """Liste des chapitres, accepte le glisser-déposer de fichiers / dossiers / zip."""
    dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setHeaderLabels(["Chapitre", "Pages", "Panneaux"])
        self.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setMinimumHeight(200)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    dragMoveEvent = dragEnterEvent

    def dropEvent(self, e):
        self.dropped.emit([Path(u.toLocalFile()) for u in e.mimeData().urls()])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Découpeur de panneaux manhwa")
        self.resize(1400, 900)
        self.chapters: dict[str, list[Path]] = {}       # nom -> pages
        self.boxes: dict[str, list[list[int]]] = {}     # clé -> [[x, y, w, h], ...]
        self.json_path: Path | None = None              # strip_data.json en cours
        self.extract_dir = Path(tempfile.mkdtemp(prefix="manhwa_zip_"))
        self.worker: Worker | None = None
        self.edit_name: str | None = None               # chapitre ouvert dans l'éditeur
        self.edit_dirty = False                          # panneaux PNG à ré-exporter

        self.setCentralWidget(self._build_ui())
        self.statusBar().showMessage("Ajoute des chapitres puis clique sur Détecter.")
        QShortcut(QKeySequence.Undo, self, self.editor.undo)
        QShortcut(QKeySequence.Redo, self, self.editor.redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self.editor.redo)
        QShortcut(QKeySequence.Save, self, self.export_current)

    # ───────────────────────── interface ─────────────────────────
    def _build_ui(self) -> QWidget:
        self.tree = ChapterTree()
        self.tree.dropped.connect(self.add_paths)
        self.tree.itemDoubleClicked.connect(self._tree_open_editor)

        btns = [("Ajouter des images…", self.pick_images), ("Ajouter un dossier…", self.pick_folder),
                ("Ajouter des ZIP…", self.pick_zips)]
        row1 = QHBoxLayout()
        btns = [("+ Images…", self.pick_images), ("+ Dossier…", self.pick_folder), ("+ ZIP…", self.pick_zips)]
        for text, slot in btns:
            b = QPushButton(text)
            b.clicked.connect(slot)
            row1.addWidget(b)
        row2 = QHBoxLayout()
        for text, slot in [("Retirer", self.remove_selected), ("Vider", self.clear_chapters)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            row2.addWidget(b)
        row2.addStretch()
        hint = QLabel("Glisse ici images, dossiers ou .zip. Le nom du dossier (ex. ch0001) "
                      "est la clé du chapitre dans strip_data.json. Double-clic : éditer.")
        hint.setStyleSheet("color: gray;")
        hint.setWordWrap(True)
        in_box = QGroupBox("Chapitres")
        lay = QVBoxLayout(in_box)
        lay.addLayout(row1)
        lay.addLayout(row2)
        lay.addWidget(self.tree)
        lay.addWidget(hint)

        d = core.Settings()
        self.conf = self._dspin(0.05, 0.9, 0.05, d.conf)
        self.imgsz = self._ispin(640, 2048, 32, d.imgsz)
        self.tile_ratio = self._dspin(10, 30, 1, d.tile_ratio)
        self.overlap = self._dspin(0.2, 0.7, 0.05, d.overlap)
        self.min_side = self._ispin(10, 500, 10, d.min_side)
        self.padding = self._ispin(0, 100, 1, d.padding)
        self.keep_strip = QCheckBox("Enregistrer aussi la bande fusionnée (PNG)")
        set_box = QGroupBox("Réglages de détection")
        form = QFormLayout(set_box)
        form.addRow("Confiance minimale", self.conf)
        form.addRow("Taille d'inférence", self.imgsz)
        form.addRow("Hauteur de fenêtre (× largeur)", self.tile_ratio)
        form.addRow("Chevauchement", self.overlap)
        form.addRow("Côté minimal (px)", self.min_side)
        form.addRow("Marge autour à l'export (px)", self.padding)
        form.addRow(self.keep_strip)

        self.out_edit = QLineEdit(str(Path.home() / "Desktop" / "panneaux_manhwa"))
        self.out_edit.editingFinished.connect(self._out_changed)
        btn_out = QPushButton("…")
        btn_out.setFixedWidth(36)
        btn_out.clicked.connect(self.pick_output)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_edit)
        out_row.addWidget(btn_out)
        self.json_label = QLabel()
        self.json_label.setStyleSheet("color: gray;")
        self.json_label.setWordWrap(True)
        btn_load_json = QPushButton("Charger un strip_data.json…")
        btn_load_json.clicked.connect(self.pick_json)
        out_box = QGroupBox("Sortie")
        ol = QVBoxLayout(out_box)
        ol.addLayout(out_row)
        ol.addWidget(btn_load_json)
        ol.addWidget(self.json_label)

        self.btn_run = QPushButton("▶  Détecter")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setStyleSheet("font-weight: bold; font-size: 15px;")
        self.btn_run.setToolTip("Détecte les panneaux des chapitres sélectionnés (ou de tous)")
        self.btn_run.clicked.connect(self.start)
        self.btn_stop = QPushButton("■  Arrêter")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        btn_open = QPushButton("Ouvrir le dossier de sortie")
        btn_open.clicked.connect(self.open_output)
        run_row = QHBoxLayout()
        run_row.addWidget(self.btn_run, 2)
        run_row.addWidget(self.btn_stop, 1)

        self.progress = QProgressBar()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setMaximumHeight(120)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(in_box, 1)
        ll.addWidget(set_box)
        ll.addWidget(out_box)
        ll.addLayout(run_row)
        ll.addWidget(btn_open)
        ll.addWidget(self.progress)
        ll.addWidget(self.log_view)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_editor(), "Éditeur")
        self.tabs.addTab(self._build_panels(), "Panneaux exportés")

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(380)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left_scroll)
        split.addWidget(self.tabs)
        split.setSizes([430, 970])
        self._out_changed()
        return split

    def _build_editor(self) -> QWidget:
        self.editor = PanelEditor()
        self.editor.changed.connect(self.on_editor_changed)
        self.ch_combo = QComboBox()
        self.ch_combo.setMinimumWidth(110)
        self.ch_combo.activated.connect(lambda _: self.open_in_editor(self.ch_combo.currentText()))

        def btn(text, slot, tip="", width=None):
            b = QPushButton(text)
            b.clicked.connect(slot)
            b.setToolTip(tip)
            if width:
                b.setFixedWidth(width)
            return b

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Chapitre :"))
        bar.addWidget(self.ch_combo)
        bar.addWidget(btn("↶", self.editor.undo, "Annuler (Ctrl+Z)", 36))
        bar.addWidget(btn("↷", self.editor.redo, "Rétablir (Ctrl+Maj+Z)", 36))
        bar.addWidget(btn("✕", self.editor.delete_selected, "Supprimer le cadre sélectionné (Suppr)", 36))
        bar.addWidget(btn("−", lambda: self.editor.zoom(1 / 1.25), "Zoom arrière (Ctrl+molette)", 36))
        bar.addWidget(btn("+", lambda: self.editor.zoom(1.25), "Zoom avant (Ctrl+molette)", 36))
        bar.addWidget(btn("↔", self.editor.fit_width, "Ajuster à la largeur", 36))
        bar.addStretch()
        self.edit_info = QLabel()
        bar.addWidget(self.edit_info)
        bar.addWidget(btn("💾 Exporter", self.export_current,
                          "Ré-écrit les PNG de ce chapitre (Ctrl+S). Le JSON est enregistré à chaque modification."))
        help_lbl = QLabel("Glisser dans le vide : tracer · glisser un cadre : déplacer · "
                          "bords/coins : redimensionner · double-clic : supprimer · Ctrl+Z : annuler")
        help_lbl.setStyleSheet("color: gray;")
        help_lbl.setWordWrap(True)
        w = QWidget()
        v = QVBoxLayout(w)
        v.addLayout(bar)
        v.addWidget(help_lbl)
        v.addWidget(self.editor, 1)
        return w

    def _build_panels(self) -> QWidget:
        self.panel_list = QListWidget()
        self.panel_list.setViewMode(QListWidget.IconMode)
        self.panel_list.setIconSize(QSize(150, 210))
        self.panel_list.setResizeMode(QListWidget.Adjust)
        self.panel_list.setMovement(QListWidget.Static)
        self.panel_list.setSpacing(8)
        self.panel_list.setUniformItemSizes(True)
        self.panel_list.itemDoubleClicked.connect(self._panel_to_editor)
        w = QWidget()
        v = QVBoxLayout(w)
        lbl = QLabel("Double-clic sur un panneau : le corriger dans l'éditeur.")
        lbl.setStyleSheet("color: gray;")
        v.addWidget(lbl)
        v.addWidget(self.panel_list)
        return w

    @staticmethod
    def _dspin(lo, hi, step, val):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(2)
        w.setValue(val)
        return w

    @staticmethod
    def _ispin(lo, hi, step, val):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setValue(val)
        return w

    def settings(self) -> core.Settings:
        return core.Settings(conf=self.conf.value(), imgsz=self.imgsz.value(),
                             tile_ratio=self.tile_ratio.value(), overlap=self.overlap.value(),
                             min_side=self.min_side.value(), padding=self.padding.value(),
                             keep_strip=self.keep_strip.isChecked())

    def log(self, msg: str):
        self.log_view.appendPlainText(msg)
        self.statusBar().showMessage(msg)

    def out_root(self) -> Path:
        return Path(self.out_edit.text()).expanduser()

    # ───────────────────────── strip_data.json ─────────────────────────
    def _out_changed(self):
        """Le JSON suit le dossier de sortie (sauf si on en a chargé un autre)."""
        path = self.out_root() / "strip_data.json"
        if self.json_path != path:
            self.json_path = path
            try:
                loaded = core.load_strip_data(path)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "JSON illisible", f"{path} : {e}")
                loaded = {}
            for k, v in loaded.items():
                self.boxes.setdefault(k, v)
        self._update_json_label()
        self.refresh_tree()

    def _update_json_label(self):
        n = sum(len(v) for v in self.boxes.values())
        self.json_label.setText(f"JSON : {self.json_path}  ({len(self.boxes)} chapitres, {n} panneaux)")

    def save_json(self):
        try:
            core.save_strip_data(self.json_path, self.boxes)
        except OSError as e:
            self.log(f"ERREUR d'écriture du JSON : {e}")
        self._update_json_label()

    def pick_json(self):
        f, _ = QFileDialog.getOpenFileName(self, "strip_data.json", str(self.out_root()), "JSON (*.json)")
        if not f:
            return
        try:
            data = core.load_strip_data(Path(f))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "JSON illisible", str(e))
            return
        self._close_editor_chapter()
        self.json_path = Path(f)
        self.out_edit.setText(str(Path(f).parent))
        self.boxes.update(data)
        self._update_json_label()
        self.refresh_tree()
        self.log(f"JSON chargé : {len(data)} chapitres. Ajoute les dossiers des chapitres pour les éditer.")
        if self.edit_name:
            self.open_in_editor(self.edit_name)

    # ───────────────────────── entrées ─────────────────────────
    def pick_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Pages d'un chapitre", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        self.add_paths([Path(f) for f in files])

    def pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Dossier de chapitre(s)")
        if d:
            self.add_paths([Path(d)])

    def pick_zips(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Archives ZIP", "", "ZIP (*.zip)")
        self.add_paths([Path(f) for f in files])

    def pick_output(self):
        d = QFileDialog.getExistingDirectory(self, "Dossier de sortie", self.out_edit.text())
        if d:
            self.out_edit.setText(d)
            self._out_changed()

    def add_paths(self, paths: list[Path]):
        loose: dict[str, list[Path]] = {}
        for p in paths:
            try:
                if p.is_dir():
                    self._merge(core.chapters_from_folder(p))
                elif p.suffix.lower() == ".zip":
                    self._merge(core.chapters_from_zip(p, self.extract_dir))
                elif p.suffix.lower() in core.IMAGE_EXT:
                    loose.setdefault(p.parent.name or "chapitre", []).append(p)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "Fichier ignoré", f"{p.name} : {e}")
        self._merge(loose)
        self.refresh_tree()

    def _merge(self, chapters: dict[str, list[Path]]):
        for name, pages in chapters.items():
            known = set(self.chapters.get(name, []))
            self.chapters.setdefault(name, []).extend(p for p in pages if p not in known)
        self.chapters = dict(sorted(self.chapters.items(), key=lambda kv: core.natural_key(kv[0])))

    def refresh_tree(self):
        self.tree.clear()
        for name, pages in self.chapters.items():
            key = core.safe_name(name)
            n = str(len(self.boxes[key])) if key in self.boxes else "—"
            it = QTreeWidgetItem([name, str(len(pages)), n])
            it.setData(0, Qt.UserRole, name)
            for p in core.sort_pages(pages):
                it.addChild(QTreeWidgetItem([p.name, "", ""]))
            self.tree.addTopLevelItem(it)
        current = self.ch_combo.currentText()
        self.ch_combo.clear()
        self.ch_combo.addItems(list(self.chapters))
        if current in self.chapters:
            self.ch_combo.setCurrentText(current)
        total = sum(len(v) for v in self.chapters.values())
        self.statusBar().showMessage(f"{len(self.chapters)} chapitre(s), {total} page(s).")

    def _selected_names(self) -> list[str]:
        names = []
        for it in self.tree.selectedItems():
            name = (it if it.parent() is None else it.parent()).data(0, Qt.UserRole)
            if name not in names:
                names.append(name)
        return names

    def remove_selected(self):
        for name in self._selected_names():
            if name == self.edit_name:
                self._close_editor_chapter()
                self.editor.clear()
                self.edit_name = None
            self.chapters.pop(name, None)
        self.refresh_tree()

    def clear_chapters(self):
        self._close_editor_chapter()
        self.editor.clear()
        self.edit_name = None
        self.chapters.clear()
        self.refresh_tree()

    # ───────────────────────── détection ─────────────────────────
    def start(self):
        if not self.chapters:
            QMessageBox.information(self, "Rien à faire", "Ajoute d'abord des images, un dossier ou un ZIP.")
            return
        names = self._selected_names() or list(self.chapters)
        already = [n for n in names if core.safe_name(n) in self.boxes]
        if already and QMessageBox.question(
                self, "Remplacer ?",
                f"{len(already)} chapitre(s) ont déjà des panneaux (éventuellement corrigés à la main).\n"
                "La détection va les remplacer. Continuer ?") != QMessageBox.Yes:
            return
        out_root = self.out_root()
        try:
            out_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Dossier de sortie", str(e))
            return
        self._close_editor_chapter()
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.worker = Worker({n: self.chapters[n] for n in names}, out_root, self.settings())
        self.worker.log.connect(self.log)
        self.worker.progress.connect(lambda v, m: (self.progress.setMaximum(m), self.progress.setValue(v)))
        self.worker.chapter_done.connect(self.on_chapter_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.log("Arrêt demandé…")

    def on_chapter_done(self, res: dict):
        key = core.safe_name(res["name"])
        self.boxes[key] = res["boxes"]
        self.save_json()
        self.refresh_tree()
        self.show_panels(res["name"], res["panels"])
        if self.edit_name is None or self.edit_name == res["name"]:
            self.open_in_editor(res["name"], force=True)

    def on_failed(self, msg: str):
        self.log(f"ERREUR : {msg}")
        QMessageBox.critical(self, "Erreur", msg)

    def on_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setValue(self.progress.maximum())

    # ───────────────────────── éditeur ─────────────────────────
    def _tree_open_editor(self, item, _col):
        name = (item if item.parent() is None else item.parent()).data(0, Qt.UserRole)
        self.open_in_editor(name)

    def open_in_editor(self, name: str, force: bool = False, focus_index: int = -1):
        if name not in self.chapters:
            return
        if name != self.edit_name or force:
            self._close_editor_chapter()
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self.statusBar().showMessage(f"Fusion des pages de {name}…")
                QApplication.processEvents()
                strip = core.stitch(core.sort_pages(self.chapters[name]))
                self.editor.load(strip, self.boxes.get(core.safe_name(name), []))
                self.edit_name = name
                self.edit_dirty = False
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                QMessageBox.critical(self, "Ouverture impossible", f"{type(e).__name__} : {e}")
                return
            finally:
                QApplication.restoreOverrideCursor()
        self.ch_combo.setCurrentText(name)
        self.tabs.setCurrentIndex(0)
        self._update_edit_info()
        if focus_index >= 0:
            self.editor.focus_box(focus_index)

    def on_editor_changed(self):
        if self.edit_name is None:
            return
        self.boxes[core.safe_name(self.edit_name)] = self.editor.get_boxes()
        self.edit_dirty = True
        self.save_json()
        self._update_edit_info()
        n = len(self.boxes[core.safe_name(self.edit_name)])
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if it.data(0, Qt.UserRole) == self.edit_name:
                it.setText(2, str(n))

    def _update_edit_info(self):
        if self.edit_name is None or self.editor.strip is None:
            self.edit_info.setText("")
            return
        s = self.editor.strip
        flag = "  •  PNG à ré-exporter" if self.edit_dirty else ""
        self.edit_info.setText(f"{len(self.editor.boxes)} panneaux{flag}")
        self.edit_info.setToolTip(f"bande {s.width}×{s.height}px")

    def export_current(self):
        if self.edit_name is None or self.editor.strip is None:
            return
        name = self.edit_name
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            panels = core.export_chapter(self.editor.strip, self.editor.get_boxes(), name,
                                         self.chapters[name], self.out_root(),
                                         self.padding.value(), self.keep_strip.isChecked())
        finally:
            QApplication.restoreOverrideCursor()
        self.edit_dirty = False
        self.save_json()
        self._update_edit_info()
        self.show_panels(name, panels)
        self.log(f"✔ {name} : {len(panels)} panneaux exportés + JSON enregistré")

    def _close_editor_chapter(self):
        """Avant de quitter un chapitre modifié : ré-exporter ses PNG."""
        if self.edit_name is not None and self.edit_dirty and self.editor.strip is not None:
            self.export_current()

    # ───────────────────────── panneaux exportés ─────────────────────────
    def show_panels(self, name: str, panels: list[Path]):
        for i in reversed(range(self.panel_list.count())):
            if self.panel_list.item(i).data(Qt.UserRole)[0] == name:
                self.panel_list.takeItem(i)
        for i, path in enumerate(panels):
            pix = QPixmap(str(path)).scaled(150, 210, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item = QListWidgetItem(QIcon(pix), f"{name} #{i + 1}")
            item.setData(Qt.UserRole, (name, i, str(path)))
            self.panel_list.addItem(item)

    def _panel_to_editor(self, item: QListWidgetItem):
        name, index, _ = item.data(Qt.UserRole)
        self.open_in_editor(name, focus_index=index)

    def open_output(self):
        out = self.out_root()
        out.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)
        self._close_editor_chapter()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Découpeur de panneaux manhwa")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
