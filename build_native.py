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


if __name__ == "__main__":
    print(build())
