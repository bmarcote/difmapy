# Installation

difmapy is a Python package around a compiled Rust core. Building it
needs a Rust toolchain; *using* a built wheel does not.

!!! note "Licence"

    difmapy is licensed under AGPL-3.0-only. Read `LICENSE` before redistributing it; the upstream Difmap sources
    retain their own terms.

## From a checkout

```sh
# Rust, once: https://rustup.rs
git clone https://github.com/bmarcote/difmapy
cd difmapy

pip install maturin
maturin develop --release --extras plot,ms
```

`--release` matters: the debug build is about ten times slower, which
shows on large maps. The extras are optional:

| extra | what it adds |
|---|---|
| `plot` | matplotlib, for the `bayes_gscale` figure |
| `ms` | Measurement Set support (casatools) |
| `dev` | pytest and maturin |

!!! tip "Editable install"

    `maturin develop` leaves an *editable* install that follows the
    sources in the checkout. A plain `pip install .` afterwards copies
    them into site-packages instead, and edits to the checkout then
    change nothing. If the package seems not to pick up a change, check
    where it is imported from:

    ```sh
    python -c "import difmapy; print(difmapy.__file__)"
    ```

## Check that it works

```sh
difmapy --version
python -c "import difmapy; print(difmapy.__version__)"
```

## Requirements

- Python 3.11 or newer; numpy, astropy, IPython, pyqtgraph and PySide6
  are installed with it. Without a display - on a server, in CI, in a
  notebook - plots are drawn off-screen instead; see
  [Plots](guide/plots.md#notebooks-and-pipelines).
- **casatools** to read and write Measurement Sets and CASA calibration
  tables. UVFITS needs nothing extra.
- **matplotlib** (the `plot` extra) for the `bayes_gscale` figure.

Building redistributable wheels and publishing are described in
`INSTALL.md` in the repository.
