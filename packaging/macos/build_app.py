"""Build difmapy.app and a drag-to-Applications disk image (macOS).

    python packaging/macos/build_app.py                 # dist/difmapy.app + dist/difmapy-<version>.dmg
    python packaging/macos/build_app.py --no-ms         # without casatools (no Measurement Sets)
    python packaging/macos/build_app.py --sign "Developer ID Application: ..." --notarize <profile>

The app carries its own Python: a copy of a relocatable CPython (python-build-standalone, which is what uv installs - by default the one running this script) with difmapy and its dependencies installed into it, so it needs nothing on the Mac it is copied to. Its main executable is a small launcher (``launcher.c``) compiled here against the current macOS SDK, which embeds that Python: macOS gives an app the Liquid Glass look of its SDK by its main executable, and the Python binary itself is built against an older one. The optional window glass background (``$DIFMAPY_GLASS=1``) is ``glass.m`` (``_glass.dylib``).

On first open the app offers to link ``/usr/local/bin/difmapy`` to the command inside it (``difmapy.app_main``), so the Terminal session runs on the same Python.

Signing: ad hoc by default, which runs on this Mac; a Mac that downloads the disk image will refuse it (Gatekeeper) until it is signed with a Developer ID certificate (``--sign``) and notarized (``--notarize``, a ``notarytool`` keychain profile).
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import build_native  # noqa: E402

#: What the app installs into its Python. PySide6-Essentials rather than
#: PySide6: the add-ons (WebEngine, 3D, ...) are most of its gigabyte and
#: difmapy uses none of them.
REQUIREMENTS = ["numpy>=1.26", "astropy>=6.0", "ipython>=8.0", "pyqtgraph>=0.13",
                "PySide6-Essentials", "matplotlib>=3.7"]
MS_REQUIREMENTS = ["casatools"]
BUNDLE_ID = "io.github.bmarcote.difmapy"


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def project_version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def python_root(explicit=None) -> Path:
    """The relocatable CPython to bundle: its root directory (bin/, lib/, include/)."""
    root = Path(explicit) if explicit else Path(sys.base_prefix)
    lib = root / "lib" / f"libpython{sys.version_info.major}.{sys.version_info.minor}.dylib"
    if not lib.exists():
        sys.exit(f"{root} is not a relocatable (python-build-standalone) CPython: no {lib.name}. "
                 "Pass --python with one, e.g. from `uv python find`.")
    return root


def copy_python(src: Path, dest: Path):
    """The interpreter, without what an app never needs (the test suite, IDLE, caches)."""
    ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    ignore_std = shutil.ignore_patterns("__pycache__", "test", "tests", "idlelib", "turtledemo",
                                        "site-packages", "*.pyc")
    shutil.copytree(src, dest, symlinks=True, ignore=ignore_std)
    (dest / "lib" / ver / "site-packages").mkdir(exist_ok=True)
    # The standalone build refuses pip installs into it; the bundle is ours.
    (dest / "lib" / ver / "EXTERNALLY-MANAGED").unlink(missing_ok=True)
    lib = dest / "lib" / f"lib{ver}.dylib"
    # Absolute install name in the source: make it relative to whoever loads it.
    run(["install_name_tool", "-id", f"@rpath/lib{ver}.dylib", lib])
    return dest / "lib" / ver / "site-packages"


def install_packages(site: Path, python: Path, with_ms: bool):
    """The dependencies, resolved for the bundled interpreter itself (its own platform tags: casatools has macOS wheels only for macOS 15 and later)."""
    reqs = REQUIREMENTS + (MS_REQUIREMENTS if with_ms else [])
    run(["uv", "pip", "install", "--target", site, "--python", python / "bin" / "python3", *reqs])


def install_difmapy(site: Path, version: str):
    """difmapy itself: the package with its freshly built extension, and the metadata `difmapy.__version__` reads."""
    build_native.build()
    build_native.build_glass()
    pkg = ROOT / "python" / "difmapy"
    shutil.copytree(pkg, site / "difmapy", ignore=shutil.ignore_patterns("__pycache__", "_core-build-*",
                                                                         "_glass-build-*"))
    info = site / f"difmapy-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: difmapy\nVersion: {version}\n")
    (info / "INSTALLER").write_text("difmapy-app\n")
    (info / "entry_points.txt").write_text("[console_scripts]\ndifmapy = difmapy.cli:main\n")


def build_launcher(contents: Path, python: Path):
    ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    out = contents / "MacOS" / "difmapy"
    run(["clang", "-O2", "-mmacosx-version-min=12.0", str(HERE / "launcher.c"),
         "-I", python / "include" / ver, "-L", python / "lib", f"-l{ver}",
         "-Wl,-rpath,@executable_path/../Resources/python/lib", "-o", out])
    return out


def write_command(contents: Path):
    """``Contents/Resources/bin/difmapy``: the Terminal command the app links into /usr/local/bin."""
    path = contents / "Resources" / "bin" / "difmapy"
    path.parent.mkdir(parents=True)
    path.write_text("""#!/bin/sh
# difmapy's Terminal command, run with the Python inside difmapy.app.
# (Linked from /usr/local/bin by the app on its first launch.)
here="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
exec "$here/../python/bin/python3" -m difmapy.cli "$@"
""")
    path.chmod(0o755)
    return path


def make_icon(dest: Path):
    """The app icon (`difmapy.plots.icon`), at every size an .icns holds, packed with iconutil."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtGui

    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([])  # noqa: F841
    iconset = Path(tempfile.mkdtemp()) / "difmapy.iconset"
    iconset.mkdir()
    sys.path.insert(0, str(ROOT / "python"))
    from difmapy.plots.icon import icon_image

    for size in (16, 32, 64, 128, 256, 512, 1024):
        img = icon_image(size)
        for scale, name in ((1, f"icon_{size}x{size}.png"), (2, f"icon_{size // 2}x{size // 2}@2x.png")):
            if scale == 2 and size >= 32:
                img.save(str(iconset / name))
            elif scale == 1 and size <= 512:
                img.save(str(iconset / name))
    run(["iconutil", "-c", "icns", iconset, "-o", dest])
    return dest


#: The app icon, made in Icon Composer (layers in packaging/macos/icon/).
ICON = HERE / "icon" / "macos-icon.icon"


def compile_icon(resources: Path) -> dict:
    """Compile the Icon Composer icon (`ICON`) with actool into ``Assets.car`` (the Liquid Glass icon of macOS 26+) and an ``.icns`` for older systems. Returns the Info.plist keys naming it, or {} if there is no icon or no actool (then `make_icon` draws one)."""
    if not ICON.exists() or shutil.which("xcrun") is None:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        partial = Path(tmp) / "partial.plist"
        r = subprocess.run(["xcrun", "actool", str(ICON), "--compile", str(resources),
                            "--app-icon", ICON.stem, "--platform", "macosx", "--target-device", "mac",
                            "--minimum-deployment-target", "12.0",
                            "--output-partial-info-plist", str(partial)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not partial.exists():
            print(f"actool could not compile {ICON.name}; drawing the icon instead\n{r.stdout}{r.stderr}")
            return {}
        with open(partial, "rb") as f:
            return plistlib.load(f)


def write_plist(contents: Path, version: str, min_macos="13.0", icon_keys=None):
    plist = {
        "CFBundleName": "difmapy",
        "CFBundleDisplayName": "difmapy",
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": "difmapy",
        "CFBundleIconFile": "difmapy.icns",
        **(icon_keys or {}),
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": min_macos,
        "LSApplicationCategoryType": "public.app-category.education",
        "NSHighResolutionCapable": True,
        "NSPrincipalClass": "NSApplication",
        "NSHumanReadableCopyright": "difmapy, AGPL-3.0",
        "CFBundleDocumentTypes": [{
            "CFBundleTypeName": "UVFITS visibility data",
            "CFBundleTypeRole": "Viewer",
            "LSHandlerRank": "Alternate",
            "CFBundleTypeExtensions": ["uvf", "uvfits", "fits", "UVF", "UVFITS", "FITS"],
        }],
    }
    with open(contents / "Info.plist", "wb") as f:
        plistlib.dump(plist, f)


def sign(app: Path, identity: str):
    ent = HERE / "entitlements.plist"
    opts = [] if identity == "-" else ["--options", "runtime", "--timestamp", "--entitlements", ent]
    # Inside out: every library and executable, then the bundle.
    for path in sorted(app.rglob("*"), key=lambda p: -len(p.parts)):
        if path.is_file() and not path.is_symlink() and (
                path.suffix in (".so", ".dylib") or os.access(path, os.X_OK) and path.parent.name in ("bin", "MacOS")):
            subprocess.run(["codesign", "--force", "--sign", identity, *opts, str(path)],
                           check=False, capture_output=True)
    run(["codesign", "--force", "--sign", identity, *opts, app])


def make_dmg(app: Path, out: Path, volname="difmapy"):
    """A compressed disk image holding the app beside an Applications link: drag one onto the other."""
    stage = Path(tempfile.mkdtemp()) / volname
    stage.mkdir()
    run(["ditto", app, stage / app.name])
    (stage / "Applications").symlink_to("/Applications")
    out.unlink(missing_ok=True)
    run(["hdiutil", "create", "-volname", volname, "-srcfolder", stage, "-ov", "-format", "UDZO", out])
    shutil.rmtree(stage.parent)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--python", help="root of the relocatable CPython to bundle (default: the one running this)")
    ap.add_argument("--out", default=str(ROOT / "dist"), help="output folder (default: dist/)")
    ap.add_argument("--no-ms", action="store_true", help="leave casatools out (no Measurement Set support)")
    ap.add_argument("--sign", default="-", metavar="IDENTITY",
                    help='codesign identity (default "-": ad hoc, for this Mac only)')
    ap.add_argument("--notarize", metavar="PROFILE", help="notarytool keychain profile: notarize and staple the DMG")
    ap.add_argument("--no-dmg", action="store_true", help="build the .app only")
    ap.add_argument("--selftest", action="store_true", help="start the built app for a few seconds to check it runs")
    args = ap.parse_args(argv)
    if sys.platform != "darwin":
        sys.exit("difmapy.app can only be built on macOS")

    version = project_version()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    app = out / "difmapy.app"
    if app.exists():
        shutil.rmtree(app)
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    (contents / "Resources").mkdir()

    python = contents / "Resources" / "python"
    site = copy_python(python_root(args.python), python)
    install_packages(site, python, with_ms=not args.no_ms)
    install_difmapy(site, version)
    build_launcher(contents, python)
    write_command(contents)
    icon_keys = compile_icon(contents / "Resources")
    if not icon_keys:
        make_icon(contents / "Resources" / "difmapy.icns")
    # casatools' macOS wheels need macOS 15; without it, PySide6's 13.
    write_plist(contents, version, min_macos="13.0" if args.no_ms else "15.0", icon_keys=icon_keys)
    sign(app, args.sign)
    size = subprocess.run(["du", "-sh", str(app)], capture_output=True, text=True).stdout.split()[0]
    print(f"built {app} ({size})")

    if args.selftest:
        env = dict(os.environ, DIFMAPY_APP_SELFTEST="3")
        env.pop("QT_QPA_PLATFORM", None)
        run([contents / "MacOS" / "difmapy"], env=env, timeout=120)
        run([contents / "Resources" / "bin" / "difmapy", "--version"])

    if not args.no_dmg:
        dmg = make_dmg(app, out / f"difmapy-{version}.dmg")
        if args.sign != "-":
            run(["codesign", "--force", "--sign", args.sign, "--timestamp", dmg])
        if args.notarize:
            run(["xcrun", "notarytool", "submit", dmg, "--keychain-profile", args.notarize, "--wait"])
            run(["xcrun", "stapler", "staple", dmg])
        size = subprocess.run(["du", "-sh", str(dmg)], capture_output=True, text=True).stdout.split()[0]
        print(f"built {dmg} ({size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
