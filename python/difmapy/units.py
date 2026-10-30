"""Parsing of unit-bearing arguments.

Every difmapy argument that carries a time accepts a string with its unit - ``"30s"``, ``"30 s"``, ``"1min"``, ``"1.5 hours"`` - as well as a plain number in the argument's own base unit. Solution intervals also accept ``"scan"`` and ``"Nscan"``, which bin by whole scans rather than by a fixed duration.

Plain numbers keep the unit each argument has always had (minutes for `selfcal`'s solint, seconds for time ranges), so existing code and scripts are unaffected; a string is how you say something else.

The other direction - a number with its uncertainty and unit, as a report prints it - is `format_value` and `format_flux`.
"""

from __future__ import annotations

import math
import re

__all__ = ["TIME_UNITS", "FLUX_UNITS", "UV_UNITS", "parse_time", "parse_interval",
           "format_interval", "parse_uvdist", "format_uvdist", "flux_unit", "format_value",
           "format_flux"]

#: Recognised time units and their length in seconds.
TIME_UNITS = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "m": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "d": 86400.0, "day": 86400.0, "days": 86400.0,
}

#: Flux-density units a report may use, and their size in Jy.
FLUX_UNITS = (("Jy", 1.0), ("mJy", 1e-3), ("µJy", 1e-6))

#: Spellings of the scan "unit" (a scan is not a fixed duration).
SCAN_UNITS = frozenset({"scan", "scans"})

# A number (optional, for a bare "scan") followed by a unit.
_SPEC = re.compile(
    r"^\s*(?P<num>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)?\s*"
    r"(?P<unit>[a-zA-Z]*)\s*$"
)


def _split(text):
    """(number or None, lowercased unit) from a spec string."""
    m = _SPEC.match(text)
    if m is None:
        raise ValueError(_bad(text))
    num, unit = m.group("num"), m.group("unit").lower()
    return (float(num) if num is not None else None), unit


def _bad(text, scans=False):
    units = ", ".join(sorted(set(TIME_UNITS) | (SCAN_UNITS if scans else set())))
    return (f"cannot read {text!r} as a time; write a number with a unit, "
            f"e.g. '30s', '1min', '2 hours'" + (", 'scan' or '2scan'" if scans
                                                else "") + f" (units: {units})")


#: UV-distance prefixes and the number of wavelengths each stands for. A trailing "l", "lambda" or "λ" is optional ("50M" = "50 Mλ" = "50 Mlambda"); case does not matter, so "m" is mega, not milli - a milli-wavelength is no UV distance.
UV_UNITS = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9}

# A number followed by an optional prefix (k, M, G) and an optional
# "l"/"lambda"/"λ".
_UV_SPEC = re.compile(
    r"^\s*(?P<num>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*"
    r"(?P<prefix>[kKmMgG]?)\s*(?P<lam>l|lambda|λ|wl|wavelengths?)?\s*$",
    re.IGNORECASE,
)


def parse_uvdist(value, default="M") -> float:
    """A UV distance in wavelengths, from a number or a string with a unit.

    A number is taken to be in `default` - mega-wavelengths unless told otherwise (``"k"`` for kilo-, ``""`` or ``"lambda"`` for wavelengths). A string carries its own unit: a prefix k, M or G, optionally followed by "l", "lambda" or "λ"; a bare number in a string means `default` again. None gives None.

        parse_uvdist(50)            -> 5e7
        parse_uvdist("50M")         -> 5e7
        parse_uvdist("500 klambda") -> 5e5
        parse_uvdist("2e6 lambda")  -> 2e6
        parse_uvdist(5e7, "")       -> 5e7
    """
    if value is None:
        return None
    unit = str(default).lower().rstrip("λ").replace("lambda", "")
    if unit not in UV_UNITS:
        raise ValueError(f"unknown default UV unit {default!r}; use 'k', 'M', 'G' or ''")
    if isinstance(value, str):
        m = _UV_SPEC.match(value)
        if m is None:
            raise ValueError(f"cannot read {value!r} as a UV distance; write a number with an optional unit, "
                             "e.g. '50', '50M', '500k', '2e6 lambda' (a bare number is in M-lambda)")
        num = float(m.group("num"))
        if m.group("prefix"):
            return num * UV_UNITS[m.group("prefix").lower()]
        # A bare "lambda" means wavelengths; a bare number, the default.
        return num * (1.0 if m.group("lam") else UV_UNITS[unit])
    return float(value) * UV_UNITS[unit]


def format_uvdist(wavelengths) -> str:
    """A UV distance as a short string in M-lambda, the unit `parse_uvdist` reads by default (``5e7`` -> ``"50"``)."""
    return f"{float(wavelengths) / 1e6:g}"


def parse_time(value, default="s") -> float:
    """`value` in seconds, from a number or a string with a unit.

    A number is taken to be in `default` (seconds unless told otherwise), which is the unit that argument has always had; a string must carry its own unit, or be a bare number meaning `default` again.

        parse_time(30)         -> 30.0        seconds
        parse_time("30s")      -> 30.0
        parse_time("1 min")    -> 60.0
        parse_time(1, "min")   -> 60.0
        parse_time("1.5h")     -> 5400.0
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return float(value) * TIME_UNITS[default]
    num, unit = _split(value)
    if not unit:
        if num is None:
            raise ValueError(_bad(value))
        return num * TIME_UNITS[default]
    if unit not in TIME_UNITS:
        raise ValueError(_bad(value))
    if num is None:
        num = 1.0
    return num * TIME_UNITS[unit]


def parse_interval(value, default="s"):
    """A solution interval, as ``(seconds, nscan)``.

    Exactly one of the two is set: `nscan` counts whole scans for the scan-based forms ("scan", "2scan"), where the interval is not a duration at all, and is 0 otherwise.

        parse_interval(0, "min")        -> (0.0, 0)   per integration
        parse_interval(30, "min")       -> (1800.0, 0)
        parse_interval("30s", "min")    -> (30.0, 0)
        parse_interval("scan", "min")   -> (0.0, 1)
        parse_interval("2scan", "min")  -> (0.0, 2)
    """
    if isinstance(value, str):
        num, unit = _split(value)
        if unit in SCAN_UNITS:
            n = 1.0 if num is None else num
            if n != int(n) or n < 1:
                raise ValueError(
                    f"a scan-based interval must be a whole number of "
                    f"scans, not {value!r}"
                )
            return 0.0, int(n)
        if unit and unit not in TIME_UNITS:
            raise ValueError(_bad(value, scans=True))
    return parse_time(value, default), 0


def format_interval(seconds, nscan) -> str:
    """How a `parse_interval` result reads in a report."""
    if nscan:
        return "per scan" if nscan == 1 else f"every {nscan} scans"
    if seconds <= 0:
        return "per integration"
    # Whichever unit says it without a fraction, so the report echoes
    # what was asked for: 90 s stays 90 s rather than becoming 1.5 min.
    if seconds % 60:
        return f"every {seconds:g} s"
    if seconds % 3600:
        return f"every {seconds / 60:g} min"
    return f"every {seconds / 3600:g} h"


def format_value(value, err=None, unit="") -> str:
    """``value ± err unit``, as a report prints a measurement.

    The uncertainty is given to two significant figures and the value to the same decimal place. With no uncertainty (None, or the NaN of a parameter that was not fitted) the value stands alone, to four significant figures.

        format_value(12.3456, 0.0421, "mas")  -> "12.346 ± 0.042 mas"
        format_value(-41.27, 3.6, "deg")      -> "-41.3 ± 3.6 deg"
        format_value(12.3456, unit="mas")     -> "12.35 mas"
    """
    value = float(value)
    tail = f" {unit}" if unit else ""
    if err is None or not math.isfinite(err) or not math.isfinite(value):
        return f"{value:.4g}{tail}"
    err = abs(float(err))
    # Decimal places that show two significant figures of the error.
    places = max(0, 1 - math.floor(math.log10(err))) if err > 0 else 0
    if err == 0 or places > 6:
        # No error to set the precision by, or one so far below the
        # value (noiseless test data) that fixed notation is all zeros.
        return f"{value:.7g} ± {err:.2g}{tail}"
    value = round(value, places) + 0.0  # no "-0.00"
    return f"{value:.{places}f} ± {err:.{places}f}{tail}"


def flux_unit(jy):
    """The unit a flux density of `jy` Jy reads best in, as ``(name, size in Jy)``: whichever of Jy, mJy and µJy puts it between 0.1 and
    100. Outside what those cover it is the nearest one - Jy above
    100 Jy, µJy below 0.1 µJy - and Jy for zero.
    """
    v = abs(float(jy))
    if not v > 0:  # zero, or NaN
        return FLUX_UNITS[0]
    for name, size in FLUX_UNITS:
        if v >= 0.1 * size:
            return name, size
    return FLUX_UNITS[-1]


def format_flux(jy, err=None) -> str:
    """A flux density in Jy, with its uncertainty if there is one, in the unit `flux_unit` picks for it; both are put in that unit.

        format_flux(1.5021, 0.0034)   -> "1.5021 ± 0.0034 Jy"
        format_flux(0.0123, 0.0004)   -> "12.30 ± 0.40 mJy"
        format_flux(3.2e-5)           -> "32 µJy"
    """
    scale = abs(float(jy))
    if err is not None and math.isfinite(err):
        # A flux consistent with zero is read on the scale of its
        # error: 0.002 ± 3.0 mJy, not 2 ± 3000 µJy.
        scale = max(scale, abs(float(err)))
    name, size = flux_unit(scale)
    return format_value(jy / size, None if err is None else err / size, name)
