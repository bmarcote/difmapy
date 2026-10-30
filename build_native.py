"""Build and atomically publish the editable native extension without changing live processes."""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def build() -> Path:
    """Compile in release mode and replace the extension inode, leaving existing mappings intact."""
    root = Path(__file__).resolve().parent
    env = dict(os.environ, PYO3_PYTHON=sys.executable)
    command = ["cargo", "rustc", "--release", "-p", "difmap-py"]
    if sys.platform == "darwin":
        command += ["--", "-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]
    subprocess.run(command, cwd=root, env=env, check=True)
    if sys.platform == "darwin":
        artifact, suffix = "lib_core.dylib", ".abi3.so"
    elif sys.platform == "win32":
        artifact, suffix = "_core.dll", ".pyd"
    else:
        artifact, suffix = "lib_core.so", ".abi3.so"
    source = Path(env.get("CARGO_TARGET_DIR", root / "target")) / "release" / artifact
    target = root / "python" / "difmapy" / ("_core" + suffix)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix="_core-build-", delete=False) as handle:
        staged = Path(handle.name)
    try:
        shutil.copy2(source, staged)
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    return target


def build_glass(out=None) -> Path | None:
    """macOS only: compile the window-glass helper (packaging/macos/glass.m) into python/difmapy/_glass.dylib, which difmapy.plots.macglass loads. Returns its path, or None elsewhere or without a compiler."""
    if sys.platform != "darwin" or shutil.which("clang") is None:
        return None
    root = Path(__file__).resolve().parent
    target = Path(out) if out else root / "python" / "difmapy" / "_glass.dylib"
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix="_glass-build-", delete=False) as handle:
        staged = Path(handle.name)
    try:
        subprocess.run(["clang", "-fobjc-arc", "-dynamiclib", "-O2", "-mmacosx-version-min=11.0",
                        "-framework", "AppKit", str(root / "packaging" / "macos" / "glass.m"),
                        "-o", str(staged)], check=True)
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    return target


if __name__ == "__main__":
    print(build())
    glass = build_glass()
    if glass:
        print(glass)
