"""The entry point of difmapy.app, the macOS application: the dashboard as a desktop app.

The app's launcher (``packaging/macos/launcher.c``) runs this module in the bundled Python. It opens the dashboard - empty, or on the files the app was opened with (dropped on its icon, "Open With") - and on the first launch offers to install the ``difmapy`` command for the Terminal: a link in ``/usr/local/bin`` to ``Contents/Resources/bin/difmapy`` inside the app, which starts the usual IPython session with the app's own Python. Whatever the answer, it is not asked again; ``difmapy.app_main.install_command()`` does it later.

``$DIFMAPY_APP_SELFTEST=<seconds>`` skips the question, prints what the app found (glass, Python, the command) and quits after that many seconds - the build script's check that a bundle starts; with ``$DIFMAPY_APP_SNAPSHOT=<file.png>`` it also saves a picture of the window just before.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

__all__ = ["main", "install_command", "COMMAND_LINK"]

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


def _offer_command(parent=None):
    """First launch: ask once whether to install the Terminal command."""
    from pyqtgraph.Qt import QtWidgets

    settings = _settings()
    if settings.get("command_offered") or command_installed():
        return
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Icon.Question)
    box.setWindowTitle("difmapy")
    box.setText("Install the difmapy command for the Terminal?")
    box.setInformativeText(
        f"This links {COMMAND_LINK} to this app, so that typing difmapy in a Terminal starts the "
        "interactive session (macOS may ask for your password). You can do it later with "
        "difmapy.app_main.install_command().")
    install = box.addButton("Install", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Not Now", QtWidgets.QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(install)
    box.exec()
    settings["command_offered"] = True
    _save_settings(settings)
    if box.clickedButton() is install:
        ok, msg = install_command()
        print(f"difmapy: {msg}")
        if not ok:
            QtWidgets.QMessageBox.warning(parent, "difmapy", f"The command was not installed: {msg}")


def main(argv=None) -> int:
    """Start the app: the dashboard, on the files given (if any)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # Finder adds a process serial number argument on old systems.
    files = [a for a in argv if not a.startswith("-psn_")]
    from pyqtgraph.Qt import QtCore, QtWidgets

    from difmapy.plots.base import ensure_app, set_mode

    app = ensure_app()
    app.setApplicationName("difmapy")
    app.setApplicationDisplayName("difmapy")
    set_mode("window")
    from difmapy.plots.dashboard import Dashboard

    dash = Dashboard(None)
    dash.show()

    class _Opener(QtCore.QObject):
        """Files opened with the app (dropped on its icon, "Open With")."""

        def eventFilter(self, obj, ev):
            if ev.type() == QtCore.QEvent.Type.FileOpen:
                dash.load(ev.file())
                return True
            return False

    opener = _Opener()
    app.installEventFilter(opener)
    if files:
        dash.load(files[0])

    selftest = os.environ.get("DIFMAPY_APP_SELFTEST")
    if selftest:
        import difmapy
        from difmapy.plots import style

        print(f"difmapy {difmapy.__version__} app self-test: python {sys.version.split()[0]} "
              f"at {sys.prefix}; stdout {sys.stdout.encoding} (± λ); bundle {_bundle()}; glass {dash.glass}; "
              f"native controls {style.native_controls()} (style {app.property('difmapy_base_style')}); "
              f"command {command_target()}")
        snapshot = os.environ.get("DIFMAPY_APP_SNAPSHOT")
        if snapshot:
            # What the app draws (native controls included), for checking
            # its look without screen-recording permission.
            QtCore.QTimer.singleShot(int(float(selftest) * 1000) - 300,
                                     lambda: dash.grab().save(snapshot))
        QtCore.QTimer.singleShot(int(float(selftest) * 1000), app.quit)
    else:
        QtCore.QTimer.singleShot(300, lambda: _offer_command(dash))
    code = app.exec()
    app.removeEventFilter(opener)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
