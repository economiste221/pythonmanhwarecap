"""Découpeur de panneaux manhwa — application de bureau PyQt5.

Lancement :  python app_qt.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

from PyQt5.QtCore import QSize, Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

import panel_core as core


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
            for ci, (name, pages) in enumerate(self.chapters.items()):
                if self._stop:
                    break

                def on_tile(k, total, ci=ci):
                    if self._stop:
                        raise InterruptedError
                    self.progress.emit(ci * 1000 + int(1000 * k / total), n * 1000)

                try:
                    res = core.process_chapter(Worker._model, Worker._device, name, pages,
                                               self.out_root, self.settings,
                                               on_step=self.log.emit, on_tile=on_tile)
                except InterruptedError:
                    break
                self.log.emit(f"✔ {name} : {res['pages']} pages → {len(res['panels'])} panneaux")
                self.chapter_done.emit(res)
            self.log.emit("Arrêté." if self._stop else "Terminé.")
        except Exception as e:  # noqa: BLE001 - toute erreur doit remonter à l'interface
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__} : {e}")


class ChapterTree(QTreeWidget):
    """Liste des chapitres, accepte le glisser-déposer de fichiers / dossiers / zip."""
    dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setHeaderLabels(["Chapitre", "Pages"])
        self.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setMinimumHeight(220)

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
        self.resize(1300, 850)
        self.chapters: dict[str, list[Path]] = {}
        self.extract_dir = Path(tempfile.mkdtemp(prefix="manhwa_zip_"))
        self.worker: Worker | None = None

        # ── colonne gauche : entrées + réglages ──
        self.tree = ChapterTree()
        self.tree.dropped.connect(self.add_paths)

        btn_imgs = QPushButton("Ajouter des images…")
        btn_dir = QPushButton("Ajouter un dossier…")
        btn_zip = QPushButton("Ajouter des ZIP…")
        btn_del = QPushButton("Retirer")
        btn_clear = QPushButton("Vider")
        btn_imgs.clicked.connect(self.pick_images)
        btn_dir.clicked.connect(self.pick_folder)
        btn_zip.clicked.connect(self.pick_zips)
        btn_del.clicked.connect(self.remove_selected)
        btn_clear.clicked.connect(self.clear_chapters)
        row1 = QHBoxLayout()
        for b in (btn_imgs, btn_dir, btn_zip):
            row1.addWidget(b)
        row2 = QHBoxLayout()
        row2.addWidget(btn_del)
        row2.addWidget(btn_clear)
        row2.addStretch()

        hint = QLabel("Glisse ici des images, des dossiers ou des .zip.\n"
                      "1 dossier (ou 1 sous-dossier, ou 1 zip) = 1 chapitre.")
        hint.setStyleSheet("color: gray;")

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
        set_box = QGroupBox("Réglages")
        form = QFormLayout(set_box)
        form.addRow("Confiance minimale", self.conf)
        form.addRow("Taille d'inférence", self.imgsz)
        form.addRow("Hauteur de fenêtre (× largeur)", self.tile_ratio)
        form.addRow("Chevauchement", self.overlap)
        form.addRow("Côté minimal (px)", self.min_side)
        form.addRow("Marge autour (px)", self.padding)
        form.addRow(self.keep_strip)

        self.out_edit = QLineEdit(str(Path.home() / "Desktop" / "panneaux_manhwa"))
        btn_out = QPushButton("…")
        btn_out.setFixedWidth(36)
        btn_out.clicked.connect(self.pick_output)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_edit)
        out_row.addWidget(btn_out)
        out_box = QGroupBox("Dossier de sortie")
        out_box.setLayout(out_row)

        self.btn_run = QPushButton("▶  Découper")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setStyleSheet("font-weight: bold; font-size: 15px;")
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
        self.log_view.setMaximumHeight(130)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(in_box, 1)
        ll.addWidget(set_box)
        ll.addWidget(out_box)
        ll.addLayout(run_row)
        ll.addWidget(btn_open)
        ll.addWidget(self.progress)
        ll.addWidget(self.log_view)

        # ── colonne droite : résultats ──
        self.panel_list = QListWidget()
        self.panel_list.setViewMode(QListWidget.IconMode)
        self.panel_list.setIconSize(QSize(160, 220))
        self.panel_list.setResizeMode(QListWidget.Adjust)
        self.panel_list.setMovement(QListWidget.Static)
        self.panel_list.setSpacing(8)
        self.panel_list.setUniformItemSizes(True)
        self.panel_list.currentItemChanged.connect(self.show_panel)
        self.panel_list.itemDoubleClicked.connect(
            lambda it: QDesktopServices.openUrl(QUrl.fromLocalFile(it.data(Qt.UserRole))))

        self.viewer = QLabel("Clique sur un panneau pour l'agrandir\n(double-clic : ouvrir le fichier)")
        self.viewer.setAlignment(Qt.AlignCenter)
        viewer_scroll = QScrollArea()
        viewer_scroll.setWidgetResizable(True)
        viewer_scroll.setWidget(self.viewer)

        panels_split = QSplitter(Qt.Horizontal)
        panels_split.addWidget(self.panel_list)
        panels_split.addWidget(viewer_scroll)
        panels_split.setSizes([500, 400])

        self.strip_host = QWidget()
        self.strip_layout = QVBoxLayout(self.strip_host)
        self.strip_layout.setSpacing(0)
        self.strip_layout.addStretch()
        strip_scroll = QScrollArea()
        strip_scroll.setWidgetResizable(True)
        strip_scroll.setWidget(self.strip_host)

        self.tabs = QTabWidget()
        self.tabs.addTab(panels_split, "Panneaux")
        self.tabs.addTab(strip_scroll, "Aperçu de la bande")

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setSizes([420, 880])
        self.setCentralWidget(split)
        self.statusBar().showMessage("Ajoute des chapitres puis clique sur Découper.")

    # ── helpers ──
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

    # ── entrées ──
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
            it = QTreeWidgetItem([name, str(len(pages))])
            it.setData(0, Qt.UserRole, name)
            for p in core.sort_pages(pages):
                it.addChild(QTreeWidgetItem([p.name, ""]))
            self.tree.addTopLevelItem(it)
        total = sum(len(v) for v in self.chapters.values())
        self.statusBar().showMessage(f"{len(self.chapters)} chapitre(s), {total} page(s).")

    def remove_selected(self):
        for it in self.tree.selectedItems():
            name = (it if it.parent() is None else it.parent()).data(0, Qt.UserRole)
            self.chapters.pop(name, None)
        self.refresh_tree()

    def clear_chapters(self):
        self.chapters.clear()
        self.refresh_tree()

    # ── traitement ──
    def start(self):
        if not self.chapters:
            QMessageBox.information(self, "Rien à faire", "Ajoute d'abord des images, un dossier ou un ZIP.")
            return
        out_root = Path(self.out_edit.text()).expanduser()
        try:
            out_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Dossier de sortie", str(e))
            return

        self.panel_list.clear()
        self._clear_strip()
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.worker = Worker(dict(self.chapters), out_root, self.settings())
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
        for i, path in enumerate(res["panels"], 1):
            pix = QPixmap(str(path)).scaled(160, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item = QListWidgetItem(QIcon(pix), f"{res['name']} #{i}")
            item.setData(Qt.UserRole, str(path))
            self.panel_list.addItem(item)
        title = QLabel(f"<b>{res['name']}</b> — {len(res['panels'])} panneaux")
        self.strip_layout.insertWidget(self.strip_layout.count() - 1, title)
        for p in res["previews"]:
            lbl = QLabel()
            lbl.setPixmap(QPixmap(str(p)))
            self.strip_layout.insertWidget(self.strip_layout.count() - 1, lbl)

    def on_failed(self, msg: str):
        self.log(f"ERREUR : {msg}")
        QMessageBox.critical(self, "Erreur", msg)

    def on_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setValue(self.progress.maximum())

    def show_panel(self, item: QListWidgetItem | None, _prev=None):
        if item is None:
            return
        pix = QPixmap(item.data(Qt.UserRole))
        max_w = max(self.viewer.parentWidget().width() - 20, 200)
        if pix.width() > max_w:
            pix = pix.scaledToWidth(max_w, Qt.SmoothTransformation)
        self.viewer.setPixmap(pix)

    def _clear_strip(self):
        while self.strip_layout.count() > 1:
            w = self.strip_layout.takeAt(0).widget()
            if w:
                w.deleteLater()

    def open_output(self):
        out = Path(self.out_edit.text()).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Découpeur de panneaux manhwa")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
