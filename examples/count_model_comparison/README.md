# Poisson versus negative-binomial count regression

This example fits two weighted count regressions to the same overdispersed
dataset:

```text
Poisson:
  mu:       y ~ x + offset(log_exposure)

NBI:
  mu:       y ~ x + offset(log_exposure)
  sigma:      ~ z + offset(sigma_offset)

Var_PO(Y)  = mu
Var_NBI(Y) = mu + sigma * mu^2
```

The common mean predictor makes the comparison interpretable: NBI adds a
distributional predictor for quadratic overdispersion. The data are
deterministic and contain non-integer case weights, a log-exposure offset, and
a dispersion offset.

## Run both implementations

From the repository root:

```powershell
python tools/run_parity.py `
  examples/count_model_comparison/parity.json `
  --output-dir work/parity/count-model-comparison
```

The case executes R `gamlss` and TorchGAMLSS and compares:

- convergence, likelihood, deviance, effective degrees of freedom, AIC, AICc,
  BIC, and Pearson dispersion;
- the AIC ranking, delta, and normalized Akaike weights;
- every mean and dispersion coefficient;
- fitted means, NBI dispersion, and conditional variances;
- conditional 5th, 50th, and 95th response quantiles on a representative
  observation grid;
- randomized Dunn-Smyth residuals using identical explicit uniforms.

`python/count_model_comparison.png` shows the fitted means, predictive
intervals, residuals, and Akaike weights. `report.json` records maximum
absolute and relative errors for every compared numeric column.

## Run without R

The committed R tables let ordinary Python environments rerun the full Python
analysis:

```powershell
python tools/run_parity.py `
  examples/count_model_comparison/parity.json `
  --r-reference examples/count_model_comparison/reference/r `
  --output-dir work/parity/count-model-comparison-reference
```

Iteration counts are exported for inspection but are not an acceptance
criterion. The parity gate targets the fitted statistical quantities and the
resulting model comparison.
