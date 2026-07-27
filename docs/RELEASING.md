# Release process

TorchGAMLSS uses PEP 440 versions and currently publishes only private GitHub
pre-releases. Publishing to TestPyPI or PyPI requires a separate decision and
credentials.

## Version source

The only configured version source is:

```text
src/torchgamlss/__init__.py
```

Hatchling reads `__version__` through `[tool.hatch.version]` in
`pyproject.toml`. Package metadata and runtime imports must therefore agree
without copying the version into the build configuration.

## Local verification

Run the complete Python and R checks:

```powershell
python -m pip install -e ".[dev,release]"
python -m ruff check .
python -m pytest
python -m pip check
python -m compileall -q src/torchgamlss
Rscript tools/generate_r_references.R --check
```

Build both supported distribution formats into a fresh output directory:

```powershell
python -m build --outdir work/release-dist
python -m twine check --strict work/release-dist/*
```

Create a clean virtual environment and install the wheel rather than the
source checkout:

```powershell
python -m venv work/release-smoke
work/release-smoke/Scripts/python -m pip install `
  (Get-ChildItem work/release-dist/*.whl).FullName
Push-Location $env:TEMP
& "<repository>\work\release-smoke\Scripts\python.exe" `
  "<repository>\tools\smoke_test_install.py"
Pop-Location
```

The smoke test checks installed metadata, imports, formula fitting,
prediction, and the public `plot()`, `wp()`, and `bp()` APIs.

## Continuous integration

Every push and pull request runs:

- the complete suite on Python 3.10, 3.11, 3.12, and 3.13 under both Linux and
  Windows;
- lint, dependency, and bytecode-compilation checks;
- wheel and source-distribution construction;
- strict Twine metadata/README validation;
- installation and smoke testing of the built wheel in a clean environment;
- regeneration and comparison of all focused R parity fixtures;
- the complete Normal location-scale analysis in both R and Python, with its
  numerical report and visualization retained as the
  `normal-location-scale-parity` artifact;
- the Poisson-versus-NBI model comparison in both languages, retained as the
  `count-model-comparison-parity` artifact.

Successful distribution files are retained as the
`torchgamlss-distributions` workflow artifact.

## Cutting a private pre-release

After the branch workflow is green:

1. Confirm that `CHANGELOG.md` has the version and date.
2. Confirm that the runtime and built metadata report the same version.
3. Create an annotated tag named `v<version>`.
4. Push the tag and wait for its workflow to pass.
5. Create a private GitHub pre-release from the tag.
6. Attach the wheel and source archive produced by the tag workflow.
7. Install the attached wheel once more before announcing the release.

Do not publish a final release from an untagged branch build. Do not reuse a
version after its artifacts have been distributed; increment the alpha,
beta, or release-candidate suffix instead.
