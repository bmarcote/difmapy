/*
 * The main executable of difmapy.app: it runs the Python bundled in
 * Contents/Resources/python *in this process*, starting difmapy.app_main.
 *
 * Why not a script that execs python? macOS decides how an app looks -
 * Liquid Glass controls or the old look - from the SDK its main executable
 * was built against. The standalone Python binary is built against an older
 * SDK; this launcher is compiled with the current one by build_app.py, and
 * embedding Python keeps it the main executable.
 */
#include <Python.h>
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(PyStatus status) {
    if (PyStatus_Exception(status)) {
        Py_ExitStatusException(status);
    }
}

int main(int argc, char **argv) {
    char exe[PATH_MAX], real[PATH_MAX], contents[PATH_MAX], home[PATH_MAX], app[PATH_MAX];
    uint32_t size = sizeof(exe);
    if (_NSGetExecutablePath(exe, &size) != 0 || realpath(exe, real) == NULL) {
        fprintf(stderr, "difmapy: cannot find the application bundle\n");
        return 1;
    }
    /* .../difmapy.app/Contents/MacOS/difmapy -> .../difmapy.app/Contents */
    char tmp[PATH_MAX];
    strncpy(tmp, real, sizeof(tmp));
    strncpy(contents, dirname(dirname(tmp)), sizeof(contents));
    snprintf(home, sizeof(home), "%s/Resources/python", contents);
    strncpy(tmp, contents, sizeof(tmp));
    strncpy(app, dirname(tmp), sizeof(app));
    /* For difmapy.app_main: where the bundle is (the terminal command
     * links into it) and that it is running as the app. */
    setenv("DIFMAPY_APP_BUNDLE", app, 1);
    unsetenv("PYTHONHOME");
    unsetenv("PYTHONPATH");

    /* UTF-8 mode: started from Finder there is no locale, and the
     * isolated configuration would then leave stdout ASCII - and every
     * report with a "±" or "λ" in it would fail to print. */
    PyPreConfig preconfig;
    PyPreConfig_InitIsolatedConfig(&preconfig);
    preconfig.utf8_mode = 1;
    fail(Py_PreInitialize(&preconfig));

    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);
    config.site_import = 1;
    config.user_site_directory = 0;
    fail(PyConfig_SetBytesString(&config, &config.home, home));
    char python[PATH_MAX];
    snprintf(python, sizeof(python), "%s/bin/python3", home);
    /* sys.executable: what subprocesses (and multiprocessing) should run. */
    fail(PyConfig_SetBytesString(&config, &config.executable, python));
    fail(PyConfig_SetBytesArgv(&config, argc, argv));
    fail(PyConfig_SetString(&config, &config.run_module, L"difmapy.app_main"));
    PyStatus status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    fail(status);
    return Py_RunMain();
}
