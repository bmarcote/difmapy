"""The look of difmapy's windows: a web-like design, or the platform's own controls.

Everywhere but the macOS app, a style sheet draws every control the way a well-designed web page would: a clean sans-serif typeface, white rounded fields with a soft border and a blue focus ring, solid blue action buttons and outlined secondary ones, custom check boxes, underlined tabs, field captions in a lighter grey, cards with a soft shadow, and a dark console for the session output. It follows the system's light or dark appearance.

In the macOS app (``difmapy.app``) the controls are left to the native style instead - Liquid Glass, given the SDK its launcher is built with - and only the containers (cards, section titles, the output) are styled. ``$DIFMAPY_STYLE`` (``web`` or ``native``) overrides the choice.

`apply` installs the style sheet on the application; it is called when the Qt application is created (`difmapy.plots.base.ensure_app`) and again when the system switches between light and dark.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

__all__ = ["IS_MAC", "is_dark", "native_controls", "font_family", "palette", "stylesheet", "apply",
           "add_shadow"]

IS_MAC = sys.platform == "darwin"

#: Typefaces tried in order for the web style; the first one installed is used.
FONT_FAMILIES = ("Inter", "SF Pro Text", ".AppleSystemUIFont", "Segoe UI", "Cantarell", "Ubuntu",
                 "Noto Sans", "Roboto", "Open Sans", "DejaVu Sans", "Helvetica Neue", "Arial")


def is_dark() -> bool:
    """True when the system (or the application palette) is in dark mode; ``$DIFMAPY_APPEARANCE`` (``dark`` or ``light``) overrides it."""
    forced = os.environ.get("DIFMAPY_APPEARANCE", "").strip().lower()
    if forced in ("dark", "light"):
        return forced == "dark"
    app = QtGui.QGuiApplication.instance()
    if app is not None:
        try:
            scheme = app.styleHints().colorScheme()
            if scheme != QtCore.Qt.ColorScheme.Unknown:
                return scheme == QtCore.Qt.ColorScheme.Dark
        except AttributeError:  # Qt < 6.5
            pass
        return app.palette().color(QtGui.QPalette.ColorRole.Window).lightness() < 128
    return False


def native_controls() -> bool:
    """True where the controls are left to the platform: in the macOS app only (or as ``$DIFMAPY_STYLE`` says)."""
    choice = os.environ.get("DIFMAPY_STYLE", "").strip().lower()
    if choice in ("web", "native"):
        return choice == "native"
    return IS_MAC and bool(os.environ.get("DIFMAPY_APP_BUNDLE"))


def font_family() -> str:
    """The typeface of the web style: the first of `FONT_FAMILIES` that is installed."""
    try:
        have = set(QtGui.QFontDatabase.families())
    except Exception:  # no application yet
        have = set()
    for name in FONT_FAMILIES:
        if name in have:
            return name
    return QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.GeneralFont).family()


def palette(dark=None) -> dict:
    """The colours of the design, by role, for the light or dark appearance."""
    dark = is_dark() if dark is None else bool(dark)
    if dark:
        return dict(
            window="#0f1115", surface="#171a21", raised="#1d212a", text="#e6e8ee", muted="#9aa3b2",
            faint="#6b7383", border="#2a2f3a", field="#12151b", field_border="#323846",
            accent="#4c8dff", accent_hover="#6aa1ff", accent_pressed="#3b78e7", accent_text="#ffffff",
            hover="#222733", pressed="#2a303c", console="#0a0c10", console_text="#d7dde8",
            selection="#2d4f8f", shadow=QtGui.QColor(0, 0, 0, 120),
        )
    return dict(
        window="#f4f6f9", surface="#ffffff", raised="#ffffff", text="#111827", muted="#6b7280",
        faint="#9ca3af", border="#e5e7eb", field="#ffffff", field_border="#d1d5db",
        accent="#2563eb", accent_hover="#1d4ed8", accent_pressed="#1e40af", accent_text="#ffffff",
        hover="#f3f4f6", pressed="#e5e7eb", console="#0f172a", console_text="#e2e8f0",
        selection="#bfdbfe", shadow=QtGui.QColor(16, 24, 40, 28),
    )


def _assets(c) -> dict:
    """The little images a style sheet cannot draw (check mark, chevrons), written as SVG files and returned as ``url()`` paths."""
    folder = Path(tempfile.gettempdir()) / f"difmapy-style-{os.getuid() if hasattr(os, 'getuid') else 0}"
    folder.mkdir(parents=True, exist_ok=True)
    svgs = {
        "check": f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">'
                 f'<path d="M3.5 8.5l3 3 6-7" fill="none" stroke="{c["accent_text"]}" stroke-width="2.2" '
                 f'stroke-linecap="round" stroke-linejoin="round"/></svg>',
        "down": f'<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
                f'<path d="M3 4.5l3 3 3-3" fill="none" stroke="{c["muted"]}" stroke-width="1.6" '
                f'stroke-linecap="round" stroke-linejoin="round"/></svg>',
    }
    out = {}
    for name, svg in svgs.items():
        path = folder / f"{name}-{abs(hash(svg)) & 0xffffffff:08x}.svg"
        if not path.exists():
            path.write_text(svg)
        out[name] = path.as_posix()
    return out


def stylesheet(dark=None, native=None, glass=False) -> str:
    """The application's style sheet.

    Parameters
    ----------
    dark : bool | None, default None
        Dark appearance; ``None`` follows the system.
    native : bool | None, default None
        Leave buttons, fields, check boxes and tabs to the platform style, styling only the containers; ``None`` decides with `native_controls`.
    glass : bool, default False
        The window has a glass background (macOS, ``$DIFMAPY_GLASS``): keep the containers translucent.

    Returns
    -------
    str
        The Qt style sheet.
    """
    c = palette(dark)
    native = native_controls() if native is None else bool(native)
    if glass:
        dark_ = is_dark() if dark is None else dark
        c = dict(c, window="transparent",
                 surface="rgba(44,44,48,0.62)" if dark_ else "rgba(255,255,255,0.66)")
    if native:
        return _native_containers(c, dark=is_dark() if dark is None else bool(dark))
    # The containers of the web style.
    css = f"""
    QWidget#controlsPage, QScrollArea#controlsScroll,
    QScrollArea#controlsScroll > QWidget > QWidget {{ background: {c['window']}; }}
    QFrame#card {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 14px;
    }}
    QFrame#card QLabel, QFrame#card QCheckBox {{ background: transparent; }}
    QLabel#sectionTitle {{
        color: {c['faint']}; background: transparent;
        font-size: 11px; font-weight: 700; letter-spacing: 1px;
        padding: 8px 4px 0px 4px;
    }}
    QLabel#hint {{ color: {c['muted']}; background: transparent; font-size: 12px; }}
    QLabel#footer {{ color: {c['muted']}; background: transparent; border-top: 1px solid {c['border']};
                    padding: 3px 6px; font-size: 10px; }}
    QPlainTextEdit#output {{
        background: {c['console']}; color: {c['console_text']};
        border: 1px solid {c['border']}; border-radius: 12px; padding: 10px;
        selection-background-color: {c['selection']};
    }}
    """
    a = _assets(c)
    font = font_family()
    css += f"""
    QWidget {{ font-family: "{font}"; font-size: 13px; color: {c['text']}; }}
    QMainWindow, QDialog, QMessageBox, QTabWidget, QStackedWidget {{ background: {c['window']}; }}
    QToolTip {{
        background: {c['raised']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 6px; padding: 5px 8px;
    }}

    /* text fields */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {c['field']}; color: {c['text']};
        border: 1px solid {c['field_border']}; border-radius: 8px;
        padding: 6px 10px; min-height: 20px;
        selection-background-color: {c['selection']}; selection-color: {c['text']};
        placeholder-text-color: {c['faint']};
    }}
    QLineEdit:hover, QComboBox:hover {{ border-color: {c['faint']}; }}
    QLineEdit:focus, QComboBox:focus {{ border: 2px solid {c['accent']}; padding: 5px 9px; }}
    QLineEdit:disabled {{ color: {c['faint']}; background: {c['hover']}; }}
    QComboBox::drop-down {{ border: none; width: 26px; }}
    QComboBox::down-arrow {{ image: url("{a['down']}"); width: 12px; height: 12px; }}
    QComboBox QAbstractItemView {{
        background: {c['raised']}; border: 1px solid {c['border']}; border-radius: 8px;
        padding: 4px; selection-background-color: {c['hover']}; selection-color: {c['text']};
        outline: none;
    }}

    /* buttons: outlined by default, solid blue for actions */
    QPushButton {{
        background: {c['raised']}; color: {c['text']};
        border: 1px solid {c['field_border']}; border-radius: 8px;
        padding: 7px 16px; min-height: 20px; font-weight: 600;
    }}
    QPushButton:hover {{ background: {c['hover']}; }}
    QPushButton:pressed {{ background: {c['pressed']}; }}
    QPushButton:focus {{ border: 2px solid {c['accent']}; padding: 6px 15px; }}
    QPushButton:disabled {{ color: {c['faint']}; }}
    QPushButton#primary, QPushButton#action {{
        background: {c['accent']}; color: {c['accent_text']}; border: 1px solid {c['accent']};
    }}
    QPushButton#primary:hover, QPushButton#action:hover {{
        background: {c['accent_hover']}; border-color: {c['accent_hover']};
    }}
    QPushButton#primary:pressed, QPushButton#action:pressed {{
        background: {c['accent_pressed']}; border-color: {c['accent_pressed']};
    }}
    QPushButton#primary:focus, QPushButton#action:focus {{ border: 2px solid {c['accent_hover']}; }}

    /* check boxes */
    QCheckBox {{ spacing: 8px; background: transparent; }}
    QCheckBox::indicator {{
        width: 16px; height: 16px; border-radius: 5px;
        border: 1px solid {c['field_border']}; background: {c['field']};
    }}
    QCheckBox::indicator:hover {{ border-color: {c['accent']}; }}
    QCheckBox::indicator:checked {{
        background: {c['accent']}; border-color: {c['accent']}; image: url("{a['check']}");
    }}

    /* tabs: underlined, as on a web page */
    QTabWidget::pane {{ border: none; border-top: 1px solid {c['border']}; top: -1px; }}
    QTabWidget::tab-bar {{ left: 8px; }}
    QTabBar {{ background: transparent; qproperty-drawBase: 0; }}
    QTabBar::tab {{
        background: transparent; color: {c['muted']};
        padding: 9px 14px 8px 14px; margin-right: 4px;
        border: none; border-bottom: 2px solid transparent; font-weight: 600;
    }}
    QTabBar::tab:hover {{ color: {c['text']}; }}
    QTabBar::tab:selected {{ color: {c['accent']}; border-bottom: 2px solid {c['accent']}; }}

    /* chrome */
    QStatusBar {{
        background: {c['window']}; color: {c['muted']};
        border-top: 1px solid {c['border']}; font-size: 12px;
    }}
    QStatusBar QLabel {{ color: {c['muted']}; }}
    QSplitter::handle {{ background: {c['border']}; }}
    QSplitter::handle:horizontal {{ width: 1px; }}
    QSplitter::handle:vertical {{ height: 1px; }}
    QGraphicsView {{ border: none; }}
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle {{ background: {c['field_border']}; border-radius: 4px; min-height: 30px; min-width: 30px; }}
    QScrollBar::handle:hover {{ background: {c['faint']}; }}
    QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
        height: 0px; width: 0px; background: none;
    }}
    """
    return css


def _native_containers(c, dark) -> str:
    """The macOS app's sheet: only the containers, and nothing that would take a control off the native style.

    One surface for the whole window - the palette's base colour, the white (or near-black) the plots are drawn on, through ``palette(base)`` so that it follows the system's light and dark - so there are no grey bands between panels. Cards are hairline outlines with no fill. No rule may match a scroll area, tab widget, button or field: Qt then stops drawing it (and, for a scroll area, its scroll bars) natively.
    """
    line = "rgba(255,255,255,0.13)" if dark else "rgba(0,0,0,0.10)"
    return f"""
    QMainWindow, QStatusBar, QSplitter, QWidget#controlsPage, QWidget#controlsBody {{ background: palette(base); }}
    QFrame#card {{
        background: transparent;
        border: 1px solid {line};
        border-radius: 12px;
    }}
    QLabel#sectionTitle {{
        color: {c['muted']}; background: transparent;
        font-size: 11px; font-weight: 600; letter-spacing: 0.8px;
        padding: 8px 2px 0px 2px;
    }}
    QLabel#hint {{ color: {c['muted']}; background: transparent; font-size: 11px; }}
    QLabel#footer {{ color: {c['muted']}; background: transparent; border-top: 1px solid {line};
                    padding: 3px 6px; font-size: 10px; }}
    QPlainTextEdit#output {{
        border: 1px solid {line}; border-radius: 10px; padding: 8px;
    }}
    """


#: Whether the last sheet applied keeps a glass window see-through.
_GLASS = False


def apply(app=None, glass=None) -> str:
    """Install the style sheet on the application (and, for the web style, the palette's placeholder colour), and follow the system when it switches between light and dark. Returns the sheet."""
    global _GLASS
    app = app or QtWidgets.QApplication.instance()
    if app is None:
        return ""
    if glass is not None:
        _GLASS = bool(glass)
    css = stylesheet(glass=_GLASS)
    app.setStyleSheet(css)
    if not getattr(app, "_difmapy_follows_scheme", False):
        try:
            app.styleHints().colorSchemeChanged.connect(lambda *_: apply(app))
            app._difmapy_follows_scheme = True
        except AttributeError:  # Qt < 6.5: no live switching
            pass
    if not native_controls():
        # The palette too, so that whatever the sheet does not reach (a
        # tab bar's strip, a message box) has the design's colours.
        c = palette()
        pal = app.palette()
        R = QtGui.QPalette.ColorRole
        for role, key in ((R.Window, "window"), (R.WindowText, "text"), (R.Base, "field"),
                          (R.AlternateBase, "hover"), (R.Text, "text"), (R.Button, "raised"),
                          (R.ButtonText, "text"), (R.ToolTipBase, "raised"), (R.ToolTipText, "text"),
                          (R.PlaceholderText, "faint"), (R.Highlight, "accent"),
                          (R.HighlightedText, "accent_text")):
            pal.setColor(role, QtGui.QColor(c[key]))
        app.setPalette(pal)
    return css


def add_shadow(widget, dark=None):
    """A soft drop shadow under a card (web style only: native controls get none)."""
    if native_controls():
        return None
    effect = QtWidgets.QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(22)
    effect.setOffset(0, 3)
    effect.setColor(palette(dark)["shadow"])
    widget.setGraphicsEffect(effect)
    return effect
