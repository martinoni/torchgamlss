# Release process

TorchGAMLSS uses PEP 440 versions and publishes GitHub pre-releases. The
package-index rollout is TestPyPI-first and uses OpenID Connect trusted
publishing, without a long-lived API token. Production PyPI publication
remains a separate release decision.

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

## Cutting a GitHub pre-release

After the branch workflow is green:

1. Confirm that `CHANGELOG.md` has the version and date.
2. Confirm that the runtime and built metadata report the same version.
3. Create an annotated tag named `v<version>`.
4. Push the tag and wait for its workflow to pass.
5. Create a GitHub pre-release from the tag.
6. Attach the wheel and source archive produced by the tag workflow.
7. Install the attached wheel once more before announcing the release.

Do not publish a final release from an untagged branch build. Do not reuse a
version after its artifacts have been distributed; increment the alpha,
beta, or release-candidate suffix instead.

## Publishing to TestPyPI

The manually dispatched `.github/workflows/release.yml` workflow publishes
only to TestPyPI. Its build job checks out the requested tag, requires the tag
to be annotated, verifies that the tag and package versions agree, builds the
wheel and source archive, and runs strict Twine validation. A separate job
downloads exactly those artifacts and publishes them from the `testpypi`
GitHub environment with a short-lived OpenID Connect credential.

The first trusted publication, `0.1.0a1`, completed on 2026-07-29 UTC and is
available at <https://test.pypi.org/project/torchgamlss/0.1.0a1/>. Its
[release workflow run](https://github.com/martinoni/torchgamlss/actions/runs/30412935839)
and isolated wheel smoke test both passed.

When bootstrapping a replacement TestPyPI project, register a pending GitHub
Actions publisher at <https://test.pypi.org/manage/account/publishing/> with
these exact values:

| Field | Value |
| --- | --- |
| PyPI project name | `torchgamlss` |
| Owner | `martinoni` |
| Repository name | `torchgamlss` |
| Workflow name | `release.yml` |
| Environment name | `testpypi` |

TestPyPI accounts are separate from PyPI accounts. No repository secret or
API token should be created for this workflow.

Once the pending publisher exists, publish an annotated tag with:

```powershell
gh workflow run release.yml --ref main -f tag=v0.1.0a1
$runId = gh run list --workflow release.yml --limit 1 `
  --json databaseId --jq '.[0].databaseId'
gh run watch $runId --exit-status
```

After the run succeeds, download only TorchGAMLSS from TestPyPI, verify its
digest against the release asset, and install the local wheel. This avoids
using a secondary package index to resolve dependencies:

```powershell
$version = "0.1.0a1"
$download = "work/testpypi-download"
python -m venv work/testpypi-smoke
python -m pip download --no-deps `
  --index-url https://test.pypi.org/simple/ `
  --dest $download `
  "torchgamlss==$version"
$wheel = Get-ChildItem $download -Filter "torchgamlss-$version-*.whl"
Get-FileHash $wheel -Algorithm SHA256
work/testpypi-smoke/Scripts/python -m pip install $wheel
```

Run the smoke test stored in the release tag from outside the repository. The
development-branch test may require APIs added after an older tag:

```powershell
Push-Location work
try {
  git -C .. show "v${version}:tools/smoke_test_install.py" |
    & testpypi-smoke/Scripts/python -
} finally {
  Pop-Location
}
```

Production PyPI should receive its own `pypi` environment with manual
approval and a separately registered trusted publisher only after the
TestPyPI artifact has passed this smoke test.
