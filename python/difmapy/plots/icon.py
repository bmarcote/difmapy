"""difmapy's icon, drawn rather than shipped: a rounded tile with the uv-coverage tracks of an array round a bright source. The app bundle's .icns is made from it (``packaging/macos/build_app.py``) and windows show it."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui

__all__ = ["icon_image", "app_icon"]


#: The layers of the icon, bottom to top (`icon_image(layers=...)`).
LAYERS = ("background", "tracks", "source")


def icon_image(size=1024, layers=LAYERS, tile=True) -> QtGui.QImage:
    """The icon at `size` x `size` pixels.

    Parameters
    ----------
    size : int, default 1024
        Width and height in pixels.
    layers : sequence of str, default LAYERS
        Which layers to draw: "background" (the gradient), "tracks" (the uv-coverage arcs), "source" (the bright source at the centre). One alone, on a transparent canvas, is a layer for Icon Composer.
    tile : bool, default True
        Draw the background as a rounded tile with the margin macOS icons have. ``False`` fills the whole canvas, as Icon Composer wants (it applies the shape itself), and scales the artwork to it.

    Returns
    -------
    QImage
        The image, ARGB with a transparent background where nothing is drawn.
    """
    img = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32)
    img.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    # The artwork is drawn on the 824-pixel tile of a 1024 canvas; without
    # the tile it fills the canvas instead.
    s = size / (1024.0 if tile else 824.0)
    off = 100 * s if tile else 0.0
    rect = QtCore.QRectF(off, off, 824 * s, 824 * s)
    c = rect.center()
    if "background" in layers:
        grad = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0.0, QtGui.QColor("#0b1e5b"))
        grad.setColorAt(1.0, QtGui.QColor("#4b2a8c"))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(grad)
        if tile:
            p.drawRoundedRect(rect, 185 * s, 185 * s)
        else:
            p.drawRect(QtCore.QRectF(0, 0, size, size))
    if "tracks" in layers:
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
    if "source" in layers:
        glow = QtGui.QRadialGradient(c, 120 * s)
        glow.setColorAt(0.0, QtGui.QColor(255, 236, 160, 255))
        glow.setColorAt(0.35, QtGui.QColor(255, 190, 90, 200))
        glow.setColorAt(1.0, QtGui.QColor(255, 120, 40, 0))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawEllipse(c, 120 * s, 120 * s)
    p.end()
    return img


def write_icon_files(folder, size=1024) -> list:
    """Write the icon as PNGs into `folder`: ``difmapy-icon.png`` (the finished icon) and, for Icon Composer, the full-bleed layers ``layer-1-background.png``, ``layer-2-tracks.png`` and ``layer-3-source.png``. Returns the paths."""
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = [folder / "difmapy-icon.png"]
    icon_image(size).save(str(out[0]))
    for i, name in enumerate(LAYERS, 1):
        path = folder / f"layer-{i}-{name}.png"
        icon_image(size, layers=(name,), tile=False).save(str(path))
        out.append(path)
    return out


def app_icon() -> QtGui.QIcon:
    """The icon as a QIcon, at the usual sizes."""
    icon = QtGui.QIcon()
    for size in (16, 32, 64, 128, 256, 512):
        icon.addPixmap(QtGui.QPixmap.fromImage(icon_image(size)))
    return icon
