# The dashboard

```python
obs.dashboard()          # every view of the session in one window
difmapy.dashboard()      # the same, empty: load a file from its Main tab
```

At the `difmapy` prompt it is `dashboard()`, and `difmapy --dashboard
file.ms` opens it straight away.

The window holds four panels:

| panel | shows |
| --- | --- |
| top left | `vplot`, `tplot` and `possm`, one tab each |
| top right | `radplot`, `projplot` and `corplot`, one tab each |
| bottom left | `mapplot` |
| bottom right | the **Main** and **Output** tabs |

The plots are the ordinary interactive ones, embedded: click into one
and its keys, mouse flagging and status line work as in its own window
(++h++ there lists them; ++q++ does not close an embedded plot).

## The panels follow each other

Whatever changes the data or the model in one panel - a flag edit in
`vplot` or `radplot`, a CLEAN (++c++) or a fit (++f++) in the map -
reloads the plots on show in the others, as ++r++ would in each. A plot
in a tab that is not shown is reloaded when its tab is next opened, so
the cost is only paid for what is on the screen. The calibration
buttons do the same, and **Reload all** (or ++r++ on the dashboard
itself) catches up with changes made from the prompt.

## Main

| control | does |
| --- | --- |
| **File...**, path | the file to load; the button opens the platform's own dialog, for a UVFITS file or a Measurement Set (entering the MS folder picks it, as does its `table.dat`) |
| timeavg, freqavg | optional averaging on load, as `load(timeavg=, freqavg=)` takes it: a time with a unit (`10s`, `2min`), and a channel count or `all` |
| **Load** | loads the file and rebuilds every plot for it; at the prompt the new data becomes `obs` |
| **Selfcal** | `selfcal` with phase and/or amplitude, as ticked (phase by default), at the solution interval typed: minutes as a bare number, a string with a unit (`30s`), `scan`, or empty for one solution per integration. With **Converge with modelfit** (on by default) it alternates with `modelfit` until the fit stops improving, as `selfcal(converge=True)`; a model of CLEAN components alone has nothing to fit, and gets one selfcal round |
| **Gscale** | `gscale`, with `float_scale` as ticked (on by default) |
| **Bayes** | `bayes_gscale` over the source models ticked: `clean`, `gauss1`, `gauss2`, `gauss3` (the Gaussians by default) |
| **UV range** | `uvrange` between min and max (an empty max is no limit), and re-image |
| **Taper** | `uvtaper` with value and radius (an empty value is no taper), and re-image |

UV distances are in M-lambda, or a number with a unit (`500k`). Enter in
a field applies its row too. The fields always show what is in effect,
also when it was set in the map panel or at the prompt.

Selfcal, Gscale and Bayes refuse to run before there is a model -
self-calibrating against nothing would fill the gain table with NaN.
CLEAN or fit one in the map first.

## Output

Everything the session prints - the reports of `clean`, `modelfit`,
`selfcal`, `gscale`, `bayes_gscale`, the dashboard's own messages, and
whatever is typed at the prompt meanwhile - also goes to this tab.
Under **Save:**, its buttons save, each to a file it asks for with the
platform's own dialog:

| button | writes |
| --- | --- |
| Restored map | `wmap`: the restored CLEAN map, FITS |
| Dirty map | `wdmap`: the residual map, FITS |
| Dirty beam | `wbeam`, FITS |
| Model | `wmodel`: a Difmap `.mod` file |
| Visibilities | `wobs`: the calibrated, edited data, UVFITS |
| Calibration table | `savecaltable`, in the data's own format (CASA or AIPS) |
| Flags | `wflags`: the flags added in the session, in the data's own format |
| Whole session | `save`: everything above under one prefix, including the restored map in every weighting |

## Look

The controls are arranged in cards - one per command, its options under
its button - and follow the system's light or dark appearance
(`$DIFMAPY_APPEARANCE=dark` or `light` fixes it).

From a terminal - on GNU/Linux and on macOS alike - every difmapy window
is drawn in a web-like style: a clean sans-serif typeface (Inter, or the
first of Cantarell, Ubuntu, Noto Sans... that is installed), white
rounded fields with a blue focus ring, solid blue action buttons and
outlined secondary ones, custom check boxes, underlined tabs, field
captions in a lighter grey, cards with a soft shadow and a dark console
for the output. The macOS app keeps the native controls instead -
Liquid Glass buttons and fields, the segmented tabs and overlay scroll
bars of macOS - on one uniform surface, with the actions in the system
accent colour (`$DIFMAPY_GLASS=1` puts the window on a glass
background). `$DIFMAPY_STYLE=web` or `native` overrides the choice. The
UV range and taper are set in the Main tab only: the map panel goes
without the fields a standalone `mapplot()` has above its image.

## The macOS app

```sh
python packaging/macos/build_app.py            # dist/difmapy.app and dist/difmapy-<version>.dmg
```

The disk image holds `difmapy.app` beside a link to Applications: drag
one onto the other. The app opens the dashboard, carries its own Python
(nothing else needs installing) and, on its first launch, offers to link
`/usr/local/bin/difmapy` to the `difmapy` command inside it, so the
Terminal session runs on the same Python; `difmapy.app_main.install_command()`
does it later. Files dropped on its icon are loaded.

Built as above the app is signed ad hoc, which is enough on the Mac that
built it. For other Macs, sign it with a Developer ID certificate and
notarize it: `--sign "Developer ID Application: <name> (<team>)"
--notarize <notarytool keychain profile>`. `--no-ms` leaves casatools
out, which also lowers the minimum macOS from 15 (casatools' wheels) to 13.
