/*
 * The embedded Python of difmapy.app, as the Swift code sees it: four plain
 * C functions, so that Swift never has to import Python.h.
 */
#pragma once

/* What difmapy.macapp.emit sends: a kind ("output", "changed") and a JSON
 * payload. Called on whichever thread printed; both strings are only valid
 * during the call. */
typedef void (*dpy_notify_fn)(const char *kind, const char *payload);

/* Start Python with `home` as its prefix and `executable` as sys.executable,
 * add the ':'-separated `extra_sites` (may be empty) as site directories,
 * import `module`, hand it `notify` (module.connect) and release the GIL.
 * Returns 0, or -1 with a malloc'd message in *error. */
int dpy_start(const char *home, const char *executable, const char *extra_sites, const char *module,
              dpy_notify_fn notify, char **error);

/* module.dispatch(function, json_args) with the GIL taken, from any thread.
 * Returns a malloc'd UTF-8 JSON string, to be released with dpy_free. */
char *dpy_call(const char *function, const char *json_args);

void dpy_free(char *s);
