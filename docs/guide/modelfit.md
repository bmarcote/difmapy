# Model fitting

```python
obs.addcmp(1.5, 2.0, -1.0, type="gauss", major=3.0, ratio=0.7, phi=30,
           free=["flux", "pos", "shape"])
res = obs.modelfit(quiet=False)   # iterate until it converges
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

## The report

`modelfit` prints the fit statistics and then each fitted component,
with its 1-sigma errors:

```text
modelfit: converged after 14 iterations; reduced chi-squared 1.0235 (25918 visibilities, 7 free parameters)
  fitted 2 of 2 components; 1.514 Jy in the model

Component 0 (gauss):
  Flux density:      1.5021 ± 0.0034 Jy
  Centroid position: 2.0031 ± 0.0042 mas East, -1.0012 ± 0.0051 mas North
  Position (ICRS):   RA 16:42:58.81014044 ± 0.0042 mas, Dec +39:48:36.9929848 ± 0.0051 mas
  Size:              3.012 ± 0.021 mas FWHM, axial ratio 0.702 ± 0.008, PA 30.14 ± 0.83 deg

Component 1 (delta):
  Flux density:      12.30 ± 0.40 mJy
  Centroid position: -8 mas East, 3 mas North (fixed)
  Position (ICRS):   RA 16:42:58.809272, Dec +39:48:36.99699 (fixed)
  Size:              point source
```

The component number is its index in `res["components"]` and
`res["errors"]`. The centroid is the offset from the phase centre; the
position under it is the absolute one, the phase centre plus that
offset (deprojected exactly, and independent of any `shift`), with the
offsets' errors as the uncertainties of RA and Dec - arcs on the sky,
in mas - and the digits rounded to them. `res["positions"]` has the
same for every fitted component: `ra` and `dec` as text, `ra_deg` and
`dec_deg`, `ra_err_mas`, `dec_err_mas`, and the `frame` (ICRS, as the
FITS images also say) and `equinox`. They
are the fit's formal errors only: the uncertainty of the phase centre
itself (a phase-reference source's position) is not in them. A flux density is given in whichever of Jy, mJy and µJy
puts the number between 0.1 and 100; an error is given to two
significant figures and its value to the same decimal place; and a
parameter that was not fitted has no error and is marked `(fixed)`. The
heading is in bold in a terminal or a notebook. `verbose=False` prints
nothing, `quiet=False` returns the result as a dict, and `outfile=`
writes that dict to a JSON
file, [as for every command](imaging.md#the-numbers-each-command-returns).

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
