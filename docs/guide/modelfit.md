# Model fitting

```python
obs.addcmp(1.5, 2.0, -1.0, type="gauss", major=3.0, ratio=0.7, phi=30,
           free=["flux", "pos", "shape"])
res = obs.modelfit()            # iterate until it converges
print(res["rchisq"], res["converged"], obs.model, res["errors"])
```

Positions and sizes are in mas, angles in degrees. Component types are
`delta`, `gauss`, `disk`, `ellipse`, `ring`, `rect` and `sz`, as in
Difmap.

## What varies

`free` marks which parameters of a component the fit may change:
`"flux"`, `"pos"`, `"major"`, `"ratio"`, `"phi"`, `"spcind"`, or
`"shape"` for the three shape parameters together.

Components keep their free parameters, so a fit can simply be run again
to continue. `modelfit(free=...)` overrides them: a list gives one spec
per model component; a single spec sets which parameters vary on the
components that already have free parameters (or on every component, if
none has).

As in Difmap, components without free parameters - CLEAN components
above all - are held fixed, and their visibilities are subtracted before
the fit, so that it does not absorb their flux a second time.

## Starting from nothing

With no model at all, `modelfit` seeds itself with a circular Gaussian
of zero width at the peak of the residual map (`obs.seed_model()`), so
`obs.modelfit()` on a freshly loaded dataset already does something
sensible.

## Iterations and errors

`niter=-1` (the default) iterates until successive Levenberg-Marquardt
steps stop improving the reduced chi-squared; a positive `niter` runs a
fixed number of steps. The result reports `niter` and `converged`.

!!! note "A local optimizer"

    Like Difmap's, the fit converges from a sensible starting guess but
    can settle in a local minimum from a far-off one - start from
    something like the map peak. The reported `errors` are first-order
    estimates from the inverse Hessian and ignore parameter covariances.

## The model

What `addcmp`, `clean`, `rmodel` and `modelfit` produce is part of the
model straight away: `obs.model` is always the whole model, and there is
no separate "keep" step.

```python
obs.model               # a list of dicts, in mas and degrees
obs.model_flux
obs.wmodel("src.mod")   # Difmap .mod files
obs.rmodel("src.mod")
obs.clearmodel()        # drop every component
```

## From the map window

Press ++m++ in [the map display](mapplot.md) to place a component with
the mouse, and ++f++ to fit it to the UV data - no need to go back to
the prompt.
