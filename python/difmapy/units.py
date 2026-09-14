"""Parsing of unit-bearing arguments.

Every difmapy argument that carries a time accepts a string with its
unit - ``"30s"``, ``"30 s"``, ``"1min"``, ``"1.5 hours"`` - as well as a
plain number in the argument's own base unit. Solution intervals also
accept ``"scan"`` and ``"Nscan"``, which bin by whole scans rather than
by a fixed duration.

Plain numbers keep the unit each argument has always had (minutes for
`selfcal`'s solint, seconds for time ranges), so existing code and
scripts are unaffected; a string is how you say something else.
"""

from __future__ import annotations

import re

__all__ = ["TIME_UNITS", "parse_time", "parse_interval", "format_interval"]

#: Recognised time units and their length in seconds.
TIME_UNITS = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "m": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "d": 86400.0, "day": 86400.0, "days": 86400.0,
}

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


def parse_time(value, default="s") -> float:
    """`value` in seconds, from a number or a string with a unit.

    A number is taken to be in `default` (seconds unless told
    otherwise), which is the unit that argument has always had; a
    string must carry its own unit, or be a bare number meaning
    `default` again.

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

    Exactly one of the two is set: `nscan` counts whole scans for the
    scan-based forms ("scan", "2scan"), where the interval is not a
    duration at all, and is 0 otherwise.

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
