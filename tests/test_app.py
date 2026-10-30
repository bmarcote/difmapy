"""difmapy.app's Python side: the Terminal command it installs on first open, and the window glass."""

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from difmapy import app_main


@pytest.fixture()
def bundle(tmp_path, monkeypatch):
    """A fake difmapy.app with its command script, and a link location we may write to."""
    app = tmp_path / "difmapy.app"
    cmd = app / "Contents" / "Resources" / "bin" / "difmapy"
    cmd.parent.mkdir(parents=True)
    cmd.write_text("#!/bin/sh\n")
    monkeypatch.setenv("DIFMAPY_APP_BUNDLE", str(app))
    monkeypatch.setattr(app_main, "COMMAND_LINK", tmp_path / "bin" / "difmapy")
    monkeypatch.setattr(app_main, "_settings_path", lambda: tmp_path / "settings" / "app.json")
    return app, cmd


def test_install_command_links_into_the_app(bundle):
    app, cmd = bundle
    assert not app_main.command_installed()
    ok, msg = app_main.install_command()
    assert ok, msg
    assert app_main.COMMAND_LINK.resolve() == cmd.resolve()
    assert app_main.command_installed()
    # Again (a moved app, an old link): replaced, not refused.
    ok, _ = app_main.install_command()
    assert ok


def test_install_command_outside_the_app(monkeypatch):
    monkeypatch.delenv("DIFMAPY_APP_BUNDLE", raising=False)
    ok, msg = app_main.install_command()
    assert not ok and "not running from difmapy.app" in msg


def test_the_question_is_asked_once(bundle, monkeypatch):
    """The first launch asks; whatever the answer, the next does not."""
    from pyqtgraph.Qt import QtWidgets

    from difmapy.plots.base import ensure_app

    ensure_app()
    asked = []

    def fake_exec(box):
        asked.append(box.text())
        return 0

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", fake_exec)
    app_main._offer_command()
    app_main._offer_command()
    assert len(asked) == 1 and "Terminal" in asked[0]
    assert app_main._settings()["command_offered"]
    assert not app_main.command_installed()          # "Not Now" by default here


def test_glass_is_off_where_it_cannot_work(monkeypatch):
    from difmapy.plots import macglass

    monkeypatch.delenv("DIFMAPY_GLASS", raising=False)
    assert not macglass.available()                   # off unless asked for
    monkeypatch.setenv("DIFMAPY_GLASS", "0")
    assert not macglass.available()
    monkeypatch.setenv("DIFMAPY_GLASS", "1")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    assert not macglass.available()                   # nothing to draw on
    if sys.platform != "darwin":
        monkeypatch.delenv("QT_QPA_PLATFORM")
        assert not macglass.available()


def test_web_style_everywhere_but_the_mac_app(monkeypatch):
    """The terminal and GNU/Linux get the web-like style sheet; only the
    macOS app keeps the native controls. $DIFMAPY_STYLE overrides."""
    from difmapy.plots import style

    monkeypatch.delenv("DIFMAPY_STYLE", raising=False)
    monkeypatch.delenv("DIFMAPY_APP_BUNDLE", raising=False)
    assert not style.native_controls()
    monkeypatch.setenv("DIFMAPY_APP_BUNDLE", "/Applications/difmapy.app")
    assert style.native_controls() == (sys.platform == "darwin")
    monkeypatch.setenv("DIFMAPY_STYLE", "web")
    assert not style.native_controls()
    monkeypatch.setenv("DIFMAPY_STYLE", "native")
    assert style.native_controls()


def test_web_style_sheet(monkeypatch):
    """Fields, buttons, check boxes and tabs are drawn by the sheet, in a
    chosen typeface; field captions are lighter than the text."""
    from difmapy.plots import style
    from difmapy.plots.base import ensure_app

    ensure_app()
    web = style.stylesheet(dark=False, native=False)
    for rule in ("QLineEdit", "QPushButton#action", "QCheckBox::indicator:checked",
                 "QTabBar::tab:selected", "font-family"):
        assert rule in web
    c = style.palette(dark=False)
    assert f"QLabel#hint {{ color: {c['muted']}" in web
    assert style.palette(False)["muted"] != style.palette(False)["text"]
    native = style.stylesheet(dark=False, native=True)
    assert "QPushButton" not in native and "QFrame#card" in native
    monkeypatch.setenv("DIFMAPY_APPEARANCE", "dark")
    assert style.is_dark()
    monkeypatch.setenv("DIFMAPY_APPEARANCE", "light")
    assert not style.is_dark()
