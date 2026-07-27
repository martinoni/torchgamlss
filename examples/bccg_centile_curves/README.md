# BCCG fetal-growth centile curves

This example fits smooth response-centile curves to fetal abdominal
circumference measurements:

```text
mu:       y ~ pb(age, lambda=10)
sigma:      ~ pb(age, lambda=10)
nu:         ~ pb(age, lambda=10)
family:   BCCG
method:   RS
```

The Box-Cox Cole-Green distribution allows location, scale, and skewness to
change with gestational age. All three predictors use the same fixed
smoothing parameter so that the parity gate isolates the P-spline fit,
distributional parameters, and response centiles rather than differences
between smoothing-parameter optimizers.

The output grid contains the 0.4th, 2nd, 9th, 25th, 50th, 75th, 91st, 98th,
and 99.6th centiles. These symmetric normal-score probabilities make changes
in both tails visible while retaining the median as the central reference
curve.

## Run both implementations

From the repository root:

```powershell
python tools/run_parity.py `
  examples/bccg_centile_curves/parity.json `
  --output-dir work/parity/bccg-centile-curves
```

The case executes R `gamlss` and TorchGAMLSS and compares:

- convergence, likelihood, deviance, effective degrees of freedom, and AIC;
- every linear coefficient and the effective degrees of freedom of all three
  P-splines;
- fitted `mu`, `sigma`, and `nu` for all 610 observations;
- nine response-centile curves on a common 121-point age grid;
- continuous quantile residuals.

`python/bccg_centile_curves.png` shows the observations, all centile curves,
and the fitted location, scale, and shape predictors. `report.json` records
the maximum absolute and relative error for each compared quantity.

## Run without R

The committed R tables let a Python-only environment rerun the complete
TorchGAMLSS analysis:

```powershell
python tools/run_parity.py `
  examples/bccg_centile_curves/parity.json `
  --r-reference examples/bccg_centile_curves/reference/r `
  --output-dir work/parity/bccg-centile-curves-reference
```

## Data provenance

`data.csv` is the `abdom` dataset from `gamlss.data` 6.0-7. It contains 610
ultrasound measurements of fetal abdominal circumference at gestational ages
between 12 and 42 weeks. The package identifies Dr. Eileen M. Wright as the
data source and cites:

- Chitty, Altman, Henderson, and Campbell (1994), *Charts of fetal size: 3,
  abdominal measurement*, British Journal of Obstetrics and Gynaecology
  101:125-131;
- Wright and Royston (1997), *A comparison of statistical methods for
  age-related reference intervals*, Journal of the Royal Statistical Society
  A 160:47-69.

`gamlss.data` is distributed under GPL-2 or GPL-3; this repository distributes
the copied data under its GPL-3.0-only license. The example is a numerical
compatibility demonstration, not a current clinical reference chart.
