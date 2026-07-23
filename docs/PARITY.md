# R parity protocol

TorchGAMLSS treats the R implementations as executable references. A family is
not considered supported until its parameterization, links, log likelihood,
derivatives, and at least one fitted model have numerical parity tests.

## Normal family (`NO`)

The first reference slice targets `gamlss.dist` 6.1-1 and `gamlss` 5.5-0. In
the R `NO` parameterization, `mu` is the mean and `sigma` is the standard
deviation. Their default links are identity and log, respectively.

The committed fixtures cover:

- `dNO(..., log = TRUE)`;
- the default link functions;
- `dldm` and `dldd`, the parameter-scale score functions;
- `d2ldm2`, `d2ldd2`, and `d2ldmdd`, the expected second derivatives supplied
  to the GAMLSS fitting algorithm;
- a joint fit of `mu ~ x` and `sigma ~ 1` using `gamlss(..., family = NO())`.
- a weighted RS fit with parameter-specific offsets, `mu ~ x` and
  `sigma ~ z`, including coefficients, iteration count, log likelihood, and
  global deviance.

The expected sigma-sigma derivative supplied by `NO()` is Fisher-scoring
information, not the observation-wise second derivative of the normal log
density. The Python API names this distinction explicitly.

## Reproducing the fixtures

From the repository root:

```powershell
Rscript tools/install_r_dependencies.R
Rscript tools/generate_no_reference.R
Rscript tools/generate_no_reference.R --check
python -m pytest
```

The generator reads the small input datasets in `tests/reference/` and writes
the R results back to that directory. `--check` recomputes the results without
changing files and compares them with explicit tolerances.

## Sources

- `gamlss.dist` 6.1-1, `R/NO.r`, distributed by CRAN under GPL-2 or GPL-3:
  <https://cran.r-project.org/package=gamlss.dist>
- `gamlss` 5.5-0, distributed by CRAN under GPL-2 or GPL-3:
  <https://cran.r-project.org/package=gamlss>
- Rigby and Stasinopoulos (2005),
  <https://doi.org/10.1111/j.1467-9876.2005.00510.x>

The linear RS implementation follows the working-response and Fisher-weight
updates in `gamlss` 5.5-0, `R/gamlss-5.R`. See [`RS.md`](RS.md) for the
equations and current scope.
