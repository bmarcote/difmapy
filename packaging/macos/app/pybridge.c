/*
 * The embedded Python of difmapy.app (see pybridge.h).
 *
 * The app's main executable is the SwiftUI program, built against the
 * current macOS SDK, and Python runs inside it: the bundled interpreter in
 * Contents/Resources/python, or - in a development build - the one named
 * by Contents/Resources/python-dev.json.
 */
#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pybridge.h"

static PyObject *g_module;

/* The pending Python exception as a malloc'd string (and cleared). */
static char *take_error(void) {
    PyObject *type = NULL, *value = NULL, *tb = NULL;
    PyErr_Fetch(&type, &value, &tb);
    PyErr_NormalizeException(&type, &value, &tb);
    PyObject *text = value ? PyObject_Str(value) : NULL;
    const char *utf8 = text ? PyUnicode_AsUTF8(text) : NULL;
    char *out = strdup(utf8 ? utf8 : "unknown Python error");
    Py_XDECREF(text);
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(tb);
    PyErr_Clear();
    return out;
}

static int status_failed(PyStatus status, char **error) {
    if (!PyStatus_Exception(status)) {
        return 0;
    }
    *error = strdup(status.err_msg ? status.err_msg : "Python could not start");
    return 1;
}

int dpy_start(const char *home, const char *executable, const char *extra_sites, const char *module,
              dpy_notify_fn notify, char **error) {
    /* UTF-8 mode: started from Finder there is no locale, and the isolated
     * configuration would then leave stdout ASCII - and every report with a
     * "±" or "λ" in it would fail to print. */
    PyPreConfig preconfig;
    PyPreConfig_InitIsolatedConfig(&preconfig);
    preconfig.utf8_mode = 1;
    if (status_failed(Py_PreInitialize(&preconfig), error)) {
        return -1;
    }
    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);
    config.site_import = 1;
    config.user_site_directory = 0;
    /* The app owns the signals (Ctrl+C means nothing to a GUI). */
    config.install_signal_handlers = 0;
    /* Unbuffered, as `python -u`: what the app prints reaches its log at once. */
    config.buffered_stdio = 0;
    char *argv[] = {(char *)executable};
    PyStatus status = PyConfig_SetBytesString(&config, &config.home, home);
    if (!PyStatus_Exception(status)) {
        /* sys.executable: what subprocesses (and multiprocessing) run. */
        status = PyConfig_SetBytesString(&config, &config.executable, executable);
    }
    if (!PyStatus_Exception(status)) {
        status = PyConfig_SetBytesArgv(&config, 1, argv);
    }
    if (!PyStatus_Exception(status)) {
        status = Py_InitializeFromConfig(&config);
    }
    PyConfig_Clear(&config);
    if (status_failed(status, error)) {
        return -1;
    }
    if (extra_sites && *extra_sites) {
        PyObject *site = PyImport_ImportModule("site");
        if (!site) {
            *error = take_error();
            return -1;
        }
        char *dirs = strdup(extra_sites), *save = NULL;
        for (char *dir = strtok_r(dirs, ":", &save); dir; dir = strtok_r(NULL, ":", &save)) {
            PyObject *r = PyObject_CallMethod(site, "addsitedir", "s", dir);
            if (!r) {
                free(dirs);
                Py_DECREF(site);
                *error = take_error();
                return -1;
            }
            Py_DECREF(r);
        }
        free(dirs);
        Py_DECREF(site);
    }
    g_module = PyImport_ImportModule(module);
    if (!g_module) {
        *error = take_error();
        return -1;
    }
    PyObject *r = PyObject_CallMethod(g_module, "connect", "n", (Py_ssize_t)(intptr_t)notify);
    if (!r) {
        *error = take_error();
        return -1;
    }
    Py_DECREF(r);
    /* Let go of the GIL: every call takes it again (dpy_call), from the
     * main thread or the worker. */
    PyEval_SaveThread();
    return 0;
}

char *dpy_call(const char *function, const char *json_args) {
    if (!g_module) {
        return strdup("{\"error\": \"Python is not running\"}");
    }
    PyGILState_STATE gil = PyGILState_Ensure();
    char *out;
    PyObject *r = PyObject_CallMethod(g_module, "dispatch", "ss", function, json_args ? json_args : "{}");
    if (!r) {
        /* dispatch catches everything; this is a broken bridge. */
        char *msg = take_error();
        PyObject *text = PyUnicode_FromFormat("%s: %s", function, msg);
        PyObject *payload = text ? Py_BuildValue("{s:O}", "error", text) : NULL;
        PyObject *json = PyImport_ImportModule("json");
        PyObject *dumped = (json && payload) ? PyObject_CallMethod(json, "dumps", "O", payload) : NULL;
        const char *utf8 = dumped ? PyUnicode_AsUTF8(dumped) : NULL;
        out = strdup(utf8 ? utf8 : "{\"error\": \"Python bridge failure\"}");
        Py_XDECREF(dumped);
        Py_XDECREF(json);
        Py_XDECREF(payload);
        Py_XDECREF(text);
        PyErr_Clear();
        free(msg);
    } else {
        const char *utf8 = PyUnicode_AsUTF8(r);
        out = strdup(utf8 ? utf8 : "{\"error\": \"Python returned no text\"}");
        Py_DECREF(r);
        PyErr_Clear();
    }
    PyGILState_Release(gil);
    return out;
}

void dpy_free(char *s) { free(s); }
