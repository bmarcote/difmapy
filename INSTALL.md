# Installing difmapy

## 1. For your own use / manual testing

You need a Rust toolchain only to *build* difmapy; once built, it is a
normal Python package.

```sh
# Rust (once): https://rustup.rs
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# In the difmapy checkout, inside the environment you want to use:
pip install maturin
maturin develop --release            # builds the Rust core, installs editable
pip install ".[plot]"                # interactive plots (pyqtgraph + PySide6)
pip install ".[ms]"                  # Measurement Set support (casatools)
```

`maturin develop` installs difmapy in editable mode: edits to the Python
files take effect immediately, while changes to the Rust code need
`maturin develop --release` again.

Leave off `--release` for faster compiles while developing, but expect
the numerics to run roughly 10-30x slower.

### Verify the installation

```sh
python -c "import difmapy; print(difmapy.__version__)"
pytest tests -q          # real-data tests skip if the data files are absent
cargo test               # the Rust core's own tests
```

## 2. Installing a built wheel elsewhere (no Rust needed)

```sh
maturin build --release -o dist
pip install dist/difmapy-*.whl
```

The wheel is **abi3**, so a single wheel works on Python 3.11 and every
later version. Note that a wheel built this way is tagged with *your*
machine's glibc (`manylinux_2_35` on this one), so it will refuse to
install on older systems. For portable wheels use the CI workflow (see
below) or build in a manylinux container:

```sh
docker run --rm -v $PWD:/io -w /io ghcr.io/pyo3/maturin:latest \
    build --release --manylinux 2014 -o dist
```

`manylinux2014` needs only glibc >= 2.17, which covers RHEL/Rocky 7+ and
Ubuntu 16.04+ - worth having, since many observatory clusters are old.

## 3. Publishing to PyPI

> **Settle the licensing question first - see `NOTICE.md`.** difmapy is a
> close port of Difmap, whose terms forbid redistribution in modified
> form without the author's permission. Get that permission (and agree
> attribution) before making a public release, then declare a licence in
> `pyproject.toml`.

The packaging itself is ready: `pyproject.toml` carries the metadata,
the sdist builds from scratch in ~25 s, and the wheel contains only the
Python package plus one `_core.abi3.so` (no difmap sources, no test
data).

### One-off manual upload

```sh
pip install twine
maturin sdist -o dist                       # source distribution
docker run --rm -v $PWD:/io -w /io ghcr.io/pyo3/maturin:latest \
    build --release --manylinux 2014 -o dist  # portable linux wheel
twine check dist/*
twine upload --repository testpypi dist/*   # rehearse on TestPyPI first
twine upload dist/*
```

Then check the result in a clean environment:

```sh
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple difmapy
```

### Automated releases (recommended)

`.github/workflows/wheels.yml` builds Linux (x86_64 + aarch64, glibc
2.17), macOS (Intel + Apple Silicon), Windows and the sdist, smoke-tests
a wheel, and publishes on a `v*` tag using PyPI **trusted publishing**
(no API token stored anywhere). To enable it:

1. Push the repository to GitHub.
2. On PyPI: *Your projects -> Publishing -> Add a pending publisher*,
   naming the repository, the workflow file `wheels.yml`, and the
   environment `pypi`.
3. Create the matching GitHub environment named `pypi`.
4. Tag a release: `git tag v0.1.0 && git push --tags`.

### Before the first release

- [ ] Resolve licensing with the Difmap author/Caltech (`NOTICE.md`),
      then set `license` and `license-files` in `pyproject.toml`.
- [ ] Set the real repository URL in `pyproject.toml` and `Cargo.toml`
      (they currently point at `github.com/bmarcote/difmapy`).
- [ ] Keep the version in `pyproject.toml` and `Cargo.toml` in step -
      they are two separate strings today.
- [ ] Decide the support policy for `casatools`: it has no wheels for
      some platforms/Python versions, so `pip install difmapy[ms]` can
      fail even though plain `difmapy` installs fine.
