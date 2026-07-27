# Normal location-scale parity example

This example fits the same weighted Normal GAMLSS in R and Python:

```text
mu:    y ~ x + offset(mu_offset)
sigma:   ~ z + offset(sigma_offset)
family:  NO / Normal
method:  RS
```

Both response location and scale vary with covariates. Integer frequency
weights and parameter-specific offsets exercise the same model lifecycle that
is commonly used in a translated analysis.

## Run both implementations

From the repository root:

```powershell
python tools/run_parity.py `
  examples/normal_location_scale/parity.json `
  --output-dir work/parity/normal_location_scale
```

The harness discovers `Rscript` from `PATH`, `RSCRIPT`, or a standard Windows
R installation. It runs both scripts without a shell and writes:

- R and Python result tables;
- a machine-readable `report.json`;
- maximum absolute and relative errors for every compared column;
- the first mismatching key and value when a tolerance fails;
- `python/location_scale_fit.png`, a compact location, scale, and residual
  visualization.

To validate Python against the committed R result without installing R:

```powershell
python tools/run_parity.py `
  examples/normal_location_scale/parity.json `
  --r-reference examples/normal_location_scale/reference/r `
  --output-dir work/parity/normal_location_scale-python
```

## Compared outputs

The declarative manifest aligns and checks:

- convergence, global deviance, negative log likelihood, EDF, effective
  observation count, and AIC;
- every linear coefficient by distribution parameter and term name;
- fitted `mu` and `sigma` for every observation;
- the 3rd, 50th, and 97th conditional response centiles;
- continuous normal quantile residuals.

Outer iteration counts are exported for inspection but deliberately excluded
from the acceptance gate. Numerical parity is based on the fitted statistical
quantities rather than requiring identical optimizer paths.
