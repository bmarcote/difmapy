# File formats

The readers and writers behind `load`, `save`, `wobs`, `savecaltable`,
`wflags` and `save_flags`. They are normally reached through those
methods; they are documented here because their docstrings say exactly
what goes into each file.

## UVFITS

::: difmapy.io.uvfits.load_uvfits
    options:
      heading_level: 3

::: difmapy.io.uvfits.save_uvfits
    options:
      heading_level: 3

## Measurement Sets

::: difmapy.io.ms.load_ms
    options:
      heading_level: 3

::: difmapy.io.ms.save_ms
    options:
      heading_level: 3

::: difmapy.io.ms.save_averaged_ms
    options:
      heading_level: 3

::: difmapy.io.ms.save_flags
    options:
      heading_level: 3

## Calibration tables

::: difmapy.io.solutions
    options:
      heading_level: 3
      members:
        - GainSolutions
        - from_gain_table
        - constant

::: difmapy.io.caltable.save_caltable
    options:
      heading_level: 3

::: difmapy.io.aips.save_sntable
    options:
      heading_level: 3

## Flag tables

::: difmapy.io.flags
    options:
      heading_level: 3
      members:
        - flag_entries
        - save_flagcmds

::: difmapy.io.aips.save_fgtable
    options:
      heading_level: 3
