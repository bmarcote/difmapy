# Provenance and licensing notice

**Read this before distributing difmapy publicly.**

difmapy is a re-implementation of the numerical core of **Difmap**,
written by Martin C. Shepherd, Copyright (c) 1992-2010 by the
California Institute of Technology.

The difmap distribution states:

> Permission to copy the difmap distribution is granted for
> non-commercial purposes on the understanding that neither the author,
> nor Caltech shall be held responsible for any adverse consequences
> resulting from its use. **Difmap must not be re-distributed in a
> modified form without the author's explicit permission.**

and individual source files (e.g. `newfft.c`) add:

> This file is copyrighted but available free for non-commercial
> purposes ... It can also be modified under the understanding that
> this copyright notice not be removed, and with the proviso that the
> author of subsequent modifications appends note to this notice to say
> that the file has been modified by them.

## Why this matters here

difmapy was written by porting difmap's algorithms closely, including
its numerical constants, conventions and, in places, the structure of
individual routines. The modules below are direct ports of the named
difmap C sources:

| difmapy module | ported from (difmap_src/) |
|---|---|
| `grid.rs` | `uvinvert.c`, `uvtrans.c`, `costran.c`, `newfft.c` (FFT replaced by rustfft) |
| `clean.rs` | `mapclean.c`, `mapres.c` |
| `model.rs` | `modvis.c`, `besj.c` |
| `modelfit.rs` | `modfit.c`, `lmfit.c` |
| `selfcal.rs` | `slfcal.c` |
| `stream.rs` | `obutil.c` (`ob_select`), `obpol.c` |
| `edit.rs` | `obedit.c` |
| `geom.rs` | `obshift.c`, `resoff.c` |
| `closure.rs` | closure-phase handling (`clphs.c`) |

No difmap source code is included in this repository or in the
distributed package, and none of it is compiled into difmapy. But
whether a close port counts as a "modified form" of difmap, or as an
independent implementation of published algorithms, is a legal question
that the copyright holder should answer.

**Recommended action before any public release: contact Martin
Shepherd (mcs@astro.caltech.edu) and/or Caltech, describe difmapy, and
ask for explicit permission and their preferred attribution.** That is
the clean path, and such requests are commonly granted for open
re-implementations. Until then, treat this repository as
source-available for non-commercial use, mirroring difmap's own terms,
and do not publish it to PyPI under a permissive licence.

## Underlying published algorithms

The methods themselves are from the literature and are not specific to
difmap:

* CLEAN: Högbom, J. A. 1974, A&AS, 15, 417
* Self-calibration: Cornwell, T. & Fomalont, E. B. 1989, in *Synthesis
  Imaging in Radio Astronomy*, ASP, chapter 9 (eq. 9.5)
* Gaussian convolution of elliptical Gaussians: Wild, J. P. 1970,
  Aust. J. Phys., 23, 113
* Levenberg-Marquardt least squares: Levenberg 1944; Marquardt 1963
* Bessel function rational approximations: Abramowitz & Stegun,
  *Handbook of Mathematical Functions*, §9.4 (public domain; these are
  the coefficients also reproduced in Numerical Recipes)
* Gridding-convolution imaging: Thompson, Moran & Swenson,
  *Interferometry and Synthesis in Radio Astronomy*

## Third-party runtime dependencies

The distributed wheel links only permissively licensed Rust crates
(rustfft, realfft, rayon, ndarray, num-complex, pyo3, numpy - MIT
and/or Apache-2.0). Python dependencies (numpy, astropy) are BSD-like.
The optional extras pull in casatools (Measurement Set support) and
PySide6 (plots); **PySide6 is LGPL**, which is why plotting is an
optional extra rather than a hard dependency.
