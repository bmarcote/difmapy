"""Plots without a screen: the inline mode notebooks and pipelines use,
image output, and the choice between windows and inline output."""

import os

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_PLATFORMTHEME"] = ""

pg = pytest.importorskip("pyqtgraph")

import difmapy
from difmapy.plots import base

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture()
def obs(uvfits_file):
    o = difmapy.load(uvfits_file)
    o.mapsize(256, 0.25)
    return o


@pytest.fixture(autouse=True)
def restore_mode():
    old = base.set_mode("auto")
    yield
    base.set_mode(old)


def test_mode_resolution(monkeypatch):
    monkeypatch.setattr(base, "in_jupyter", lambda: False)
    monkeypatch.setattr(base, "has_display", lambda: True)
    assert base.get_mode() == "window"
    monkeypatch.setattr(base, "has_display", lambda: False)
    assert base.get_mode() == "inline"   # nothing to put a window on
    monkeypatch.setattr(base, "has_display", lambda: True)
    monkeypatch.setattr(base, "in_jupyter", lambda: True)
    assert base.get_mode() == "inline"   # notebooks draw inline by default
    assert base.set_mode("window") == "auto"
    assert base.get_mode() == "window"
    with pytest.raises(ValueError, match="unknown plot mode"):
        base.set_mode("popup")


def test_every_plot_renders_off_screen(obs, tmp_path):
    base.set_mode("inline")
    obs.addcmp(2.5, 4.0, -2.5)
    obs.selfcal(phase=True, quiet=True)
    plots = [obs.radplot(), obs.projplot(30), obs.uvplot(), obs.vplot(2),
             obs.tplot(), obs.cpplot(), obs.corplot(), obs.fplot(),
             obs.specplot(), obs.mapplot(quiet=True)]
    for i, p in enumerate(plots):
        assert not p.isVisible(), p          # drawn, never put on screen
        assert p.to_png().startswith(PNG), p
        assert p._repr_png_().startswith(PNG), p
        out = p.savefig(tmp_path / f"plot{i}.png")
        assert os.path.getsize(out) > 1000, p
        p.close()


def test_savefig_writes_every_page(obs, tmp_path):
    base.set_mode("inline")
    v = obs.vplot(3)                         # 10 baselines, 3 to a page
    assert v.npages == 4
    files = v.savefig(tmp_path / "pages" / "vplot_{page}.png")
    assert [os.path.basename(f) for f in files] == [
        f"vplot_{i}.png" for i in range(1, 5)]
    assert all(os.path.getsize(f) > 1000 for f in files)
    assert v.page == 0                       # left where it was
    v.close()
    with pytest.raises(OSError, match="could not write"):
        obs.radplot().savefig(tmp_path / "radplot.nosuchformat")


def test_inline_plots_show_in_the_notebook_once(obs, monkeypatch):
    """Under Jupyter an inline plot goes into the cell output as it is
    made. Returning it as the cell's value must not show it again - but
    once it has changed, it shows."""
    import IPython.display

    shown = []
    monkeypatch.setattr(base, "in_jupyter", lambda: True)
    monkeypatch.setattr(base, "_cell_number", lambda: 7)
    monkeypatch.setattr(IPython.display, "display", shown.append)
    p = obs.radplot()
    assert len(shown) == 1 and shown[0].data.startswith(PNG)
    assert p._repr_png_() is None
    p.cycle_antenna(1)
    assert p._repr_png_().startswith(PNG)
    assert repr(p) == "<difmapy radplot>"
    p.close()


def test_window_mode_hooks_the_event_loop_instead_of_blocking(obs, monkeypatch):
    base.set_mode("window")
    monkeypatch.setattr(base, "_hook_qt_loop", lambda: True)
    monkeypatch.setattr(pg, "exec", lambda: pytest.fail("blocked on the Qt loop"))
    p = obs.radplot()
    assert p.isVisible()
    p.close()

    # With no session to pump events a script still blocks, unless told not to.
    calls = []
    monkeypatch.setattr(base, "_hook_qt_loop", lambda: False)
    monkeypatch.setattr(pg, "exec", lambda: calls.append(1))
    obs.radplot().close()
    obs.radplot(block=False).close()
    assert calls == [1]
