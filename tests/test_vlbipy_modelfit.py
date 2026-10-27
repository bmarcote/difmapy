"""Native synthetic-data checks for vlbipy's optional modelfit-only pipeline integration."""
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

pytest.importorskip("vlbipy")
from vlbipy.backends import difmap
import difmapy


@pytest.mark.parametrize("workflow", ["target", "joint"])
def test_pipeline_modelfit_never_cleans(uvfits_file, tmp_path, monkeypatch, workflow):
    """Fit and restore a known synthetic point source without invoking any CLEAN implementation."""
    obs = difmapy.load(uvfits_file, freqavg="all")
    obs.select("I")
    obs.mapsize(256, 0.5)

    def forbid_clean(*args, **kwargs):
        raise AssertionError("modelfit-only pipeline invoked CLEAN")

    monkeypatch.setattr(type(obs), "clean", forbid_clean)
    monkeypatch.setattr(difmap, "load", lambda *a, **kw: obs)
    prefix = str(tmp_path / workflow)
    if workflow == "target":
        report = difmap.image_source(uvfits_file, prefix, mapsize=256, robust_values=(-2, 0, 2))
    else:
        monkeypatch.setattr(difmap, "phase_selfcal", lambda observation, *a, **kw: (observation, []))
        report = difmap.joint_model(uvfits_file, prefix, mapsize=256, robust_values=(-2, 0, 2))
        assert Path(report["model"]).is_file()
    model = report["image_model"]
    assert model["method"] == "modelfit" and model["converged"]
    assert len(model["components"]) == 1
    component = model["components"][0]
    assert component["flux"] == pytest.approx(2.5, rel=0.01)
    assert component["x"] == pytest.approx(4.0, abs=0.05)
    assert component["y"] == pytest.approx(-2.5, abs=0.05)
    for image in report["images"].values():
        assert image["method"] == "modelfit" and image["ncomponents"] == 1
        with fits.open(image["fits"]) as hdul:
            assert np.squeeze(hdul[0].data).shape == (128, 128)
            assert np.isfinite(hdul[0].data).all()
