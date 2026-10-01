"""Éditeur de panneaux : affiche la bande fusionnée d'un chapitre avec les cadres.

  • glisser dans le vide          -> tracer un nouveau panneau
  • glisser l'intérieur d'un cadre -> le déplacer
  • glisser un bord / un coin    -> le redimensionner
  • double-clic sur un cadre      -> le supprimer   (ou Suppr / Retour arrière)
  • Ctrl+Z / Ctrl+Maj+Z           -> annuler / rétablir
  • Ctrl+molette                  -> zoom
"""

from __future__ import annotations

from collections import OrderedDict

from PIL import Image
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QGraphicsItem, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView,
)

TILE_H = 2048          # hauteur des tuiles d'affichage (une bande peut faire 200 000 px)
MAX_CACHED_TILES = 48  # tuiles gardées en mémoire
EDGE_PX = 8            # tolérance (en pixels écran) pour attraper un bord
MIN_BOX = 12           # taille minimale d'un panneau tracé (px de la bande)

GREEN = QColor(0, 200, 0)
ORANGE = QColor(255, 140, 0)


class StripTile(QGraphicsItem):
    """Morceau de bande, converti en QPixmap seulement quand il est affiché."""

    def __init__(self, owner: "PanelEditor", y0: int, h: int):
        super().__init__()
        self.owner, self.y0, self.h = owner, y0, h
        self.setPos(0, y0)
        self.setZValue(0)

    def boundingRect(self):
        return QRectF(0, 0, self.owner.strip.width, self.h)

    def paint(self, painter, option, widget=None):
        painter.drawPixmap(0, 0, self.owner.tile_pixmap(self.y0, self.h))


class BoxItem(QGraphicsRectItem):
    def __init__(self, rect: QRectF):
        super().__init__(rect)
        self.setZValue(1)
        self.label = QGraphicsSimpleTextItem("", self)
        self.label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        f = QFont()
        f.setBold(True)
        f.setPointSize(11)
        self.label.setFont(f)
        self.label.setBrush(QBrush(Qt.white))
        self.selected = False
        self.restyle()

    def set_selected(self, on: bool):
        self.selected = on
        self.restyle()

    def restyle(self):
        color = ORANGE if self.selected else GREEN
        pen = QPen(color, 3)
        pen.setCosmetic(True)  # épaisseur constante quel que soit le zoom
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(45 if self.selected else 18)
        self.setBrush(QBrush(fill))

    def set_number(self, n: int):
        self.label.setText(f" {n} ")
        self.label.setPos(self.rect().topLeft())

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        # pastille derrière le numéro (dessinée en coordonnées écran)
        painter.save()
        t = painter.worldTransform()
        tl = t.map(self.rect().topLeft())
        painter.resetTransform()
        br = self.label.boundingRect()
        painter.fillRect(QRectF(tl.x(), tl.y(), br.width(), br.height()),
                         ORANGE if self.selected else GREEN)
        painter.restore()


class PanelEditor(QGraphicsView):
    changed = pyqtSignal()         # les cadres ont changé
    selection_changed = pyqtSignal(int)  # index (ordre de lecture) ou -1

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(60, 60, 60))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.strip: Image.Image | None = None
        self.boxes: list[BoxItem] = []
        self.current: BoxItem | None = None
        self._cache: OrderedDict[int, QPixmap] = OrderedDict()
        self._drag = None  # (mode, item, start_scene_pos, start_rect, snapshot_before)
        self._undo: list[list[list[int]]] = []
        self._redo: list[list[list[int]]] = []

    # ── chargement ──
    def load(self, strip: Image.Image, boxes: list[list[int]]):
        self.scene().clear()
        self._cache.clear()
        self.boxes, self.current = [], None
        self._undo.clear()
        self._redo.clear()
        self.strip = strip
        self.scene().setSceneRect(0, 0, strip.width, strip.height)
        for y0 in range(0, strip.height, TILE_H):
            self.scene().addItem(StripTile(self, y0, min(TILE_H, strip.height - y0)))
        self._set_boxes(boxes)
        self.fit_width()
        self.verticalScrollBar().setValue(0)

    def clear(self):
        self.scene().clear()
        self._cache.clear()
        self.boxes, self.current, self.strip = [], None, None

    def tile_pixmap(self, y0: int, h: int) -> QPixmap:
        pix = self._cache.get(y0)
        if pix is None:
            crop = self.strip.crop((0, y0, self.strip.width, y0 + h))
            data = crop.tobytes("raw", "RGB")
            img = QImage(data, crop.width, crop.height, 3 * crop.width, QImage.Format_RGB888)
            pix = QPixmap.fromImage(img.copy())
            self._cache[y0] = pix
            while len(self._cache) > MAX_CACHED_TILES:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(y0)
        return pix

    # ── cadres ──
    def get_boxes(self) -> list[list[int]]:
        out = []
        for it in self.boxes:
            r = it.rect().normalized()
            out.append([round(r.x()), round(r.y()), round(r.width()), round(r.height())])
        return sorted(out, key=lambda b: (b[1], b[0]))

    def _set_boxes(self, boxes: list[list[int]]):
        for it in self.boxes:
            self.scene().removeItem(it)
        self.boxes, self.current = [], None
        for x, y, w, h in boxes:
            it = BoxItem(QRectF(x, y, w, h))
            self.scene().addItem(it)
            self.boxes.append(it)
        self._renumber()

    def _renumber(self):
        self.boxes.sort(key=lambda it: (round(it.rect().y()), round(it.rect().x())))
        for i, it in enumerate(self.boxes, 1):
            it.set_number(i)

    def _select(self, item: BoxItem | None):
        if self.current is not None and self.current in self.boxes:
            self.current.set_selected(False)
        self.current = item
        if item is not None:
            item.set_selected(True)
        self.selection_changed.emit(self.boxes.index(item) if item in self.boxes else -1)

    def _push_undo(self, snapshot):
        self._undo.append(snapshot)
        self._redo.clear()
        if len(self._undo) > 200:
            self._undo.pop(0)

    def _commit(self):
        self._renumber()
        self.changed.emit()

    def undo(self):
        if self._undo:
            self._redo.append(self.get_boxes())
            self._set_boxes(self._undo.pop())
            self._commit()

    def redo(self):
        if self._redo:
            self._undo.append(self.get_boxes())
            self._set_boxes(self._redo.pop())
            self._commit()

    def delete_item(self, item: BoxItem):
        self._push_undo(self.get_boxes())
        if item is self.current:
            self._select(None)
        self.boxes.remove(item)
        self.scene().removeItem(item)
        self._commit()

    def delete_selected(self):
        if self.current is not None:
            self.delete_item(self.current)

    def focus_box(self, index: int):
        if 0 <= index < len(self.boxes):
            it = self.boxes[index]
            self._select(it)
            self.centerOn(it.rect().center())

    # ── zoom ──
    def fit_width(self):
        if self.strip is None:
            return
        self.resetTransform()
        avail = self.viewport().width() - 4
        s = avail / self.strip.width
        self.scale(s, s)

    def zoom(self, factor: float):
        self.scale(factor, factor)

    def wheelEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            self.zoom(1.15 if e.angleDelta().y() > 0 else 1 / 1.15)
        else:
            super().wheelEvent(e)

    # ── souris ──
    def _hit(self, p: QPointF):
        """Cadre sous le curseur + bords attrapés (l/r/t/b)."""
        tol = EDGE_PX / max(self.transform().m11(), 1e-6)
        for it in sorted(self.boxes, key=lambda b: b.rect().width() * b.rect().height()):
            r = it.rect().normalized()
            if not r.adjusted(-tol, -tol, tol, tol).contains(p):
                continue
            edges = set()
            if abs(p.x() - r.left()) <= tol:
                edges.add("l")
            if abs(p.x() - r.right()) <= tol:
                edges.add("r")
            if abs(p.y() - r.top()) <= tol:
                edges.add("t")
            if abs(p.y() - r.bottom()) <= tol:
                edges.add("b")
            if edges or r.contains(p):
                return it, edges
        return None, set()

    @staticmethod
    def _cursor_for(edges):
        if edges in ({"l", "t"}, {"r", "b"}):
            return Qt.SizeFDiagCursor
        if edges in ({"r", "t"}, {"l", "b"}):
            return Qt.SizeBDiagCursor
        if edges & {"l", "r"}:
            return Qt.SizeHorCursor
        if edges & {"t", "b"}:
            return Qt.SizeVerCursor
        return Qt.SizeAllCursor

    def _clamp(self, r: QRectF) -> QRectF:
        bounds = QRectF(0, 0, self.strip.width, self.strip.height)
        return r.normalized().intersected(bounds)

    def mousePressEvent(self, e):
        if self.strip is None or e.button() != Qt.LeftButton:
            return super().mousePressEvent(e)
        p = self.mapToScene(e.pos())
        it, edges = self._hit(p)
        snapshot = self.get_boxes()
        if it is not None:
            self._select(it)
            mode = ("resize", frozenset(edges)) if edges else ("move", None)
            self._drag = (mode, it, p, QRectF(it.rect()), snapshot)
        elif 0 <= p.x() <= self.strip.width and 0 <= p.y() <= self.strip.height:
            new = BoxItem(QRectF(p, p))
            self.scene().addItem(new)
            self.boxes.append(new)
            self._select(new)
            self._drag = (("draw", None), new, p, QRectF(p, p), snapshot)
        self.setFocus()

    def mouseMoveEvent(self, e):
        p = self.mapToScene(e.pos())
        if self._drag is None:
            if self.strip is not None:
                it, edges = self._hit(p)
                self.viewport().setCursor(self._cursor_for(edges) if it else Qt.CrossCursor)
            return super().mouseMoveEvent(e)
        (mode, edges), it, start, rect0, _ = self._drag
        d = p - start
        r = QRectF(rect0)
        if mode == "move":
            r.translate(d)
            # rester dans la bande
            r.moveLeft(min(max(r.left(), 0), self.strip.width - r.width()))
            r.moveTop(min(max(r.top(), 0), self.strip.height - r.height()))
        elif mode == "resize":
            if "l" in edges:
                r.setLeft(rect0.left() + d.x())
            if "r" in edges:
                r.setRight(rect0.right() + d.x())
            if "t" in edges:
                r.setTop(rect0.top() + d.y())
            if "b" in edges:
                r.setBottom(rect0.bottom() + d.y())
            r = self._clamp(r)
        else:  # draw
            r = self._clamp(QRectF(start, p))
        it.setRect(r)
        it.set_number(self.boxes.index(it) + 1 if it in self.boxes else 0)
        self._autoscroll(e.pos())

    def _autoscroll(self, pos):
        margin, step = 30, 40
        sb = self.verticalScrollBar()
        if pos.y() < margin:
            sb.setValue(sb.value() - step)
        elif pos.y() > self.viewport().height() - margin:
            sb.setValue(sb.value() + step)

    def mouseReleaseEvent(self, e):
        if self._drag is None:
            return super().mouseReleaseEvent(e)
        (mode, _), it, _, rect0, snapshot = self._drag
        self._drag = None
        r = it.rect().normalized()
        it.setRect(r)
        if mode == "draw" and (r.width() < MIN_BOX or r.height() < MIN_BOX):
            # simple clic dans le vide : pas de cadre
            self.boxes.remove(it)
            self.scene().removeItem(it)
            self._select(None)
            return
        if r != rect0:
            self._push_undo(snapshot)
            self._commit()

    def mouseDoubleClickEvent(self, e):
        if self.strip is None:
            return
        it, _ = self._hit(self.mapToScene(e.pos()))
        if it is not None:
            self.delete_item(it)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        elif e.key() == Qt.Key_Escape:
            self._select(None)
        else:
            super().keyPressEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
