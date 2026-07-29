# Generalized-gamma censored-survival parity example

This example fits a three-parameter generalized-gamma regression to 96
synthetic event times. There are 54 observed events and 42 right-censored
rows. The model is

```text
log(mu) = beta_0 + beta_1 x
log(sigma) = gamma_0
nu = delta_0
```

The R side uses `gamlss.cens::cens("GG", type = "right")` and RS. The Python
side uses the same censored log likelihood with Torch L-BFGS. Comparing two
optimizers makes the parity gate target the fitted statistical model rather
than iteration-by-iteration implementation details.

From the repository root on Windows PowerShell:

```powershell
python tools/run_parity.py `
  examples/generalized_gamma_survival/parity.json `
  --output-dir work/parity/generalized-gamma-survival
```

The run compares:

- global deviance and negative log likelihood;
- all four regression coefficients;
- fitted `mu`, `sigma`, and `nu` for every row;
- survival, hazard, and cumulative-hazard curves for three covariate profiles;
- latent event-time quantiles despite the censored observation mechanism.

The Python output also includes `generalized_gamma_survival.png`, showing
events, censoring marks, fitted quantiles, and survival curves.

To compare only Python against the committed R reference:

```powershell
python tools/run_parity.py `
  examples/generalized_gamma_survival/parity.json `
  --r-reference examples/generalized_gamma_survival/reference/r `
  --output-dir work/parity/generalized-gamma-survival-reference
```
