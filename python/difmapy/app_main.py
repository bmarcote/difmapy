"""Support for difmapy.app, the macOS application: the ``difmapy`` command it offers to install for the Terminal.

The app itself is a SwiftUI program (``packaging/macos/app/``) whose Python side is `difmapy.macapp`. On its first launch it asks (`command_offer_pending`, `answer_command_offer`) whether to install the ``difmapy`` command for the Terminal: a link in ``/usr/local/bin`` to ``Contents/Resources/bin/difmapy`` inside the app, which starts the usual IPython session with the app's own Python. Whatever the answer, it is not asked again; ``difmapy.app_main.install_command()`` (or the app's Settings window) does it later.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

__all__ = ["install_command", "command_installed", "command_target", "command_offer_pending",
           "answer_command_offer", "COMMAND_LINK"]

#: Where the Terminal command is linked: on the default PATH of every macOS shell.
COMMAND_LINK = Path("/usr/local/bin/difmapy")


def _bundle():
    """The .app the code runs from, or None outside one."""
    app = os.environ.get("DIFMAPY_APP_BUNDLE")
    return Path(app) if app else None


def _settings_path() -> Path:
    return Path.home() / "Library" / "Application Support" / "difmapy" / "app.json"


def _settings() -> dict:
    try:
        return json.loads(_settings_path().read_text())
    except (OSError, ValueError):
        return {}


def _save_settings(data):
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))


def command_target(bundle=None) -> Path | None:
    """The command script inside the app bundle that the Terminal link points to."""
    bundle = bundle or _bundle()
    return None if bundle is None else Path(bundle) / "Contents" / "Resources" / "bin" / "difmapy"


def command_installed(bundle=None) -> bool:
    """True if `COMMAND_LINK` already points into this app."""
    target = command_target(bundle)
    try:
        return target is not None and COMMAND_LINK.resolve() == target.resolve()
    except OSError:
        return False


def install_command(bundle=None) -> tuple[bool, str]:
    """Link ``/usr/local/bin/difmapy`` to the app's command, asking for an administrator password if the folder needs one.

    Returns
    -------
    (bool, str)
        Whether it worked, and a message saying what happened.
    """
    target = command_target(bundle)
    if target is None or not target.exists():
        return False, "this is not running from difmapy.app"
    link = COMMAND_LINK
    try:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
        return True, f"installed: {link} -> {target}"
    except OSError:
        pass
    # /usr/local/bin is root's on a fresh Mac: ask macOS for the password.
    cmd = (f"mkdir -p {shlex.quote(str(link.parent))} && "
           f"ln -sfn {shlex.quote(str(target))} {shlex.quote(str(link))}")
    script = f'do shell script "{cmd}" with administrator privileges'
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if r.returncode == 0:
        return True, f"installed: {link} -> {target}"
    return False, (r.stderr.strip() or "cancelled")


def command_offer_pending(bundle=None) -> bool:
    """True when the app should ask whether to install the Terminal command: it runs from difmapy.app (whose command exists), the question was never answered, and the command is not installed already."""
    target = command_target(bundle)
    if target is None or not target.exists():
        return False
    return not _settings().get("command_offered") and not command_installed(bundle)


def answer_command_offer(install: bool, bundle=None) -> tuple[bool, str]:
    """Record the answer to the first-launch question, so that it is not asked again, and install the command if `install`.

    Returns
    -------
    (bool, str)
        Whether the command was installed, and a message saying what happened.
    """
    settings = _settings()
    settings["command_offered"] = True
    _save_settings(settings)
    if not install:
        return False, "not installed; difmapy.app_main.install_command() does it later"
    return install_command(bundle)
