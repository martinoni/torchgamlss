# Contributing to TorchGAMLSS

Thank you for helping improve TorchGAMLSS. The project welcomes bug reports,
documentation improvements, numerical parity cases, and focused
implementations.

## Before opening an issue

- Search existing issues and the roadmap for related work.
- Reduce numerical discrepancies to a reproducible example when possible.
- Include the TorchGAMLSS version, Python and Torch versions, operating
  system, dtype, device, and the corresponding R result when reporting a
  parity problem.
- Do not disclose security-sensitive information in an issue. Follow
  `SECURITY.md` instead.

## Development setup

From a clone of the repository:

```bash
python -m venv .venv
python -m pip install -e ".[dev,release]"
python -m pytest
python -m ruff check .
```

R parity development additionally requires R 4.6.1 and the local package
library:

```bash
Rscript tools/install_r_dependencies.R
Rscript tools/generate_r_references.R --check
Rscript tools/generate_truncated_references.R
Rscript tools/generate_censored_references.R --check
Rscript tools/generate_inflated_references.R --check
```

## Pull requests

Keep pull requests small enough to review and give them one clear purpose.
New response families or fitting behavior should normally include:

- the source R package, version, functions, and papers used as references;
- parameterization, link, support, and derivative documentation;
- committed R-generated fixtures with explicit tolerances;
- autograd and response-support tests;
- at least one fitted-model test where the R implementation supports it;
- CPU coverage and CUDA coverage when the implementation has an on-device
  Torch path;
- roadmap, changelog, and user-documentation updates.

Run the complete test suite and lint checks before requesting review. Do not
commit generated build directories, local R libraries, credentials, or
proprietary data.

## Licensing

By contributing, you agree that your contribution is distributed under the
repository's GPL-3.0-only license. Translations or derivative implementations
must preserve the provenance and licensing obligations of their sources.
