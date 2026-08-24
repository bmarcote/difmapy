"""The ``difmapy`` command: an interactive IPython session for VLBI
imaging, in the spirit of difmap's own prompt.

    difmapy                                   # empty session
    difmapy myfile.uvfits                     # load and select Stokes I
    difmapy --obs data.ms --stokes I          # same, explicit
    difmapy data.ms --mapsize 2048 --cell 0.5 # set up imaging too
    difmapy data.ms -c "clean(200, 0.03)"     # run commands on startup
    difmapy data.ms --batch -c "wmap('m.fits')"   # no prompt, just run

Inside the session the loaded observation is ``obs``, and the commands
you would type in difmap (``select``, ``invert``, ``clean``, ``selfcal``,
``mapplot``, ...) are available as bare functions bound to it.
"""

from __future__ import annotations

import argparse
import os
import sys

# The observation methods exposed as bare, difmap-like commands.
COMMANDS = (
    # selection and data
    "select", "header", "flag", "unflag", "save_flags", "uvaver",
    "savecaltable", "gain_snapshot",
    # imaging setup
    "mapsize", "uvweight", "uvtaper", "uvrange", "uvzero",
    # imaging
    "invert", "clean", "keep", "clrmod", "restore", "imstat", "peak_offset",
    "add_window", "clear_windows",
    # calibration
    "selfcal", "gscale", "uncalib", "selfant", "startmod", "resoff", "clroff",
    # models
    "addcmp", "modelfit", "rmodel", "wmodel",
    # geometry
    "shift", "unshift",
    # plots
    "mapplot", "radplot", "projplot", "uvplot", "vplot",
    "cpplot", "tplot", "corplot", "specplot",
    # output
    "wobs", "wmap", "wdmap", "wbeam", "wwins", "rwins", "save",
    # data access
    "closure_phases", "spectrum",
)


def _parse_channels(spec):
    """Parse "0-31", "0:31", "4", or "0-31,64-95" into inclusive ranges."""
    if not spec:
        return None
    ranges = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        sep = "-" if "-" in part else (":" if ":" in part else None)
        if sep:
            a, b = part.split(sep, 1)
            ranges.append((int(a), int(b)))
        else:
            ranges.append((int(part), int(part)))
    return ranges


def build_parser():
    p = argparse.ArgumentParser(
        prog="difmapy",
        description="Interactive VLBI imaging and self-calibration.",
        epilog="With no --batch, an IPython session is started; the loaded "
               "observation is 'obs' and difmap-style commands are bound to it.",
    )
    p.add_argument("file", nargs="?", help="UVFITS file or Measurement Set to load")
    p.add_argument("-o", "--obs", dest="obs_file",
                   help="same as the positional argument")
    p.add_argument("-s", "--stokes", default="I",
                   help="polarization to select on load (default: I; "
                        "'pi' is a legacy alias of I). Use 'none' to skip.")
    p.add_argument("--channels", help="channel ranges to select, e.g. 0-31 or 0-3,8-11")
    p.add_argument("--mapsize", type=int, help="map size in pixels (power of 2)")
    p.add_argument("--cell", type=float, help="pixel size in mas")
    p.add_argument("--uvweight", nargs=2, type=float, metavar=("BINWID", "ERRPOW"),
                   help="gridding weights, e.g. --uvweight 0 -1 for natural")
    p.add_argument("-c", "--command", action="append", default=[], metavar="CODE",
                   help="Python to run after loading (repeatable)")
    p.add_argument("--batch", action="store_true",
                   help="run the --command code and exit, without a prompt")
    p.add_argument("--no-gui", action="store_true",
                   help="do not enable the Qt event loop (plots would block)")
    p.add_argument("--version", action="store_true", help="print the version and exit")
    return p


def _qt_available() -> bool:
    from importlib.util import find_spec

    if find_spec("pyqtgraph") is None:
        return False
    return any(find_spec(m) is not None for m in ("PySide6", "PyQt6", "PySide2", "PyQt5"))


def _load(path, stokes, channels, mapsize, cell, uvweight):
    """Load an observation and apply the startup options."""
    import difmapy

    obs = difmapy.load(path)
    if stokes and str(stokes).lower() != "none":
        try:
            obs.select(stokes, channels=channels)
        except ValueError as exc:
            print(f"warning: could not select {stokes}: {exc}", file=sys.stderr)
    if mapsize or cell:
        obs.mapsize(mapsize or 256, cell or 1.0)
    if uvweight:
        obs.uvweight(uvweight[0], uvweight[1])
    # Printed after the selection so the summary reflects it.
    print(obs.header())
    return obs


def make_namespace(obs=None):
    """The session namespace: difmapy, numpy, obs, and bound commands."""
    import numpy as np

    import difmapy

    ns = {"difmapy": difmapy, "np": np, "numpy": np, "obs": obs}

    def load(path, **kwargs):
        """Load a new observation and rebind the commands to it."""
        new = difmapy.load(path, **kwargs)
        ns["obs"] = new
        ns.update(bind_commands(new))
        print(new.header())
        return new

    ns["load"] = load
    if obs is not None:
        ns.update(bind_commands(obs))
    return ns


def bind_commands(obs):
    """Map difmap-style command names onto the observation's methods."""
    return {name: getattr(obs, name) for name in COMMANDS if hasattr(obs, name)}


def banner(obs) -> str:
    import difmapy

    lines = [
        f"difmapy {difmapy.__version__} - interactive VLBI imaging",
    ]
    if obs is None:
        lines.append("No data loaded. Use load('file.uvfits') or difmapy.load(...).")
    else:
        sel = ""
        try:
            s = obs._core.selection()
            sel = f", {s['stokes']} selected"
        except RuntimeError:
            sel = ", nothing selected yet (use select('I'))"
        lines.append(f"Loaded {obs.source}: {obs._core.nrow} rows, "
                     f"{obs.nif} IFs{sel}. The observation is 'obs'.")
    lines.append("Commands: " + ", ".join(COMMANDS[:14]) + ", ...")
    lines.append("Values live on obs (obs.model_flux, obs.dmap, obs.windows, ...)")
    lines.append("Help: obs? / clean? / difmapy.Observation?   Quit: exit")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.version:
        import difmapy

        print(f"difmapy {difmapy.__version__}")
        return 0

    path = args.obs_file or args.file
    obs = None
    if path:
        if not os.path.exists(path):
            print(f"difmapy: no such file: {path}", file=sys.stderr)
            return 2
        try:
            obs = _load(path, args.stokes, _parse_channels(args.channels),
                        args.mapsize, args.cell, args.uvweight)
        except Exception as exc:  # a bad file should not show a traceback
            print(f"difmapy: could not load {path}: {exc}", file=sys.stderr)
            return 1

    ns = make_namespace(obs)

    # Startup commands run in the session namespace, so --batch can drive
    # a whole reduction from a shell script.
    for code in args.command:
        try:
            exec(code, ns)  # noqa: S102 - this is the point of the flag
        except Exception as exc:
            print(f"difmapy: error in -c {code!r}: {exc}", file=sys.stderr)
            return 1

    if args.batch:
        return 0
    return _start_shell(ns, obs, use_gui=not args.no_gui)


def _start_shell(ns, obs, use_gui=True) -> int:
    text = banner(obs)
    try:
        from IPython import start_ipython
        from traitlets.config import Config
    except ImportError:
        # Fall back to the plain interpreter rather than failing outright.
        import code

        print(text)
        print("(install IPython for a nicer session: pip install ipython)")
        code.interact(local=ns, banner="")
        return 0

    # Print the banner ourselves: IPython's display_banner=False (which
    # we want, to hide its own banner) would suppress banner1 as well.
    print(text)
    cfg = Config()
    cfg.TerminalIPythonApp.display_banner = False
    if use_gui and _qt_available():
        # Drive the Qt loop from IPython so plot windows stay live and
        # the prompt keeps working (difmapy's plots detect this).
        cfg.InteractiveShellApp.gui = "qt"
    start_ipython(argv=[], user_ns=ns, config=cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
