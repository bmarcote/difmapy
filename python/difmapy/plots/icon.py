"""difmapy's icon, drawn rather than shipped: a rounded tile with the uv-coverage tracks of an array round a bright source. The app bundle's .icns is made from it (``packaging/macos/build_app.py``) and windows show it."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui

__all__ = ["icon_image", "app_icon"]


def icon_image(size=1024) -> QtGui.QImage:
    """The icon at `size` x `size` pixels, with a transparent margin as macOS icons have."""
    img = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32)
    img.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    s = size / 1024.0
    rect = QtCore.QRectF(100 * s, 100 * s, 824 * s, 824 * s)
    grad = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
    grad.setColorAt(0.0, QtGui.QColor("#0b1e5b"))
    grad.setColorAt(1.0, QtGui.QColor("#4b2a8c"))
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(grad)
    p.drawRoundedRect(rect, 185 * s, 185 * s)
    c = rect.center()
    # uv tracks: concentric tilted arcs, as an array traces them in a day.
    for i, (a, b, alpha) in enumerate(((330, 120, 230), (250, 92, 200), (170, 62, 170), (95, 35, 150))):
        pen = QtGui.QPen(QtGui.QColor(120, 200, 255, alpha), max(1.0, 22 * s))
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.save()
        p.translate(c)
        p.rotate(-28)
        p.drawArc(QtCore.QRectF(-a * s, -b * s, 2 * a * s, 2 * b * s), (20 + 25 * i) * 16, 260 * 16)
        p.restore()
    glow = QtGui.QRadialGradient(c, 120 * s)
    glow.setColorAt(0.0, QtGui.QColor(255, 236, 160, 255))
    glow.setColorAt(0.35, QtGui.QColor(255, 190, 90, 200))
    glow.setColorAt(1.0, QtGui.QColor(255, 120, 40, 0))
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(glow)
    p.drawEllipse(c, 120 * s, 120 * s)
    p.end()
    return img


def app_icon() -> QtGui.QIcon:
    """The icon as a QIcon, at the usual sizes."""
    icon = QtGui.QIcon()
    for size in (16, 32, 64, 128, 256, 512):
        icon.addPixmap(QtGui.QPixmap.fromImage(icon_image(size)))
    return icon
