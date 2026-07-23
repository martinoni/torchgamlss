# RS fitting

TorchGAMLSS includes an initial translation of the Rigby-Stasinopoulos (RS)
algorithm used by the R `gamlss` package. It fits linear terms and fixed-lambda
or automatically selected P-splines through additive backfitting.

## Working iteration

For distribution parameter `theta_k`, link predictor `eta_k`, offset `o_k`,
score `u_k`, and expected second log-likelihood derivative `h_k`, define

```text
d_k = d eta_k / d theta_k
w_k = -h_k / d_k^2
z_k = (eta_k - o_k) + u_k / (d_k w_k)
```

Each inner iteration fits `z_k` using weights `w_k` multiplied by the
observation likelihood weights. Without smooth terms this is weighted least
squares. With smooth terms, backfitting alternates the parametric fit and each
penalized smoother using partial residuals. The fitted predictor is combined
with the offset, transformed through the inverse link, and used to recompute
the global deviance. The outer cycle updates `mu`, `sigma`, and any future
parameters sequentially.

The working weights are clipped to `[1e-10, 1e10]`, matching the safeguards in
the R source. Automatic step halving is used when an inner update increases the
global deviance.

## Controls

`RSControl` exposes separate outer, inner, backfitting, ML smoothing-update,
target-EDF root, and GAIC/GCV minimization tolerances and iteration limits, a
step length, automatic step halving, and an allowed deviance increase. Its
defaults match `gamlss.control()`, `glim.control()`, and `pb()` where
applicable.

## Verified scope

The parity fixture fits a heteroscedastic normal model with:

```text
mu ~ x + offset(mu_offset)
log(sigma) ~ z + offset(sigma_offset)
```

It uses nonuniform likelihood weights. TorchGAMLSS matches R `gamlss` 5.5-0
in all four coefficients, global deviance, log likelihood, convergence status,
and number of outer RS cycles.

A second fixture fits `mu ~ pb(x, lambda=12)` and `sigma ~ 1`. TorchGAMLSS
matches the R parametric and spline coefficients, fitted parameters, global
deviance, outer iteration count, fixed smoothing parameter, and effective
degrees of freedom. See [`SMOOTHS.md`](SMOOTHS.md) for the Python API and the
current smoother scope.

A third fixture omits `lambda` and uses the variance-component ML update from
`pb()`. TorchGAMLSS matches the estimated smoothing parameter, effective
degrees of freedom, coefficients, fitted values, global deviance, and four
outer RS cycles.

A fourth fixture requests `pb(x, df=3)`. Following the R convention, this
means three nonlinear degrees of freedom and a target total EDF of five. The
final smoothing parameter, EDF, coefficients, fitted values, and deviance
match R. TorchGAMLSS solves the EDF root more stably, so its number of outer
RS cycles can differ from R for very tight convergence tolerances.

Fifth and sixth fixtures select `lambda` using local GAIC and GCV with
`k=2`. TorchGAMLSS matches the R smoothing parameters, EDF, coefficients,
fitted values, and deviances. The bounded Brent optimizer and R's `nlminb()`
can take different optimization and outer-RS paths, so parity is defined by
the final numerical fit rather than identical iteration counts.

The Gamma fixtures verify that the same RS core works beyond the Normal
family. They cover a weighted heteroscedastic `GA` model with offsets in both
predictors and an additive fit with a fixed-lambda P-spline for `mu`.
Coefficients, fitted parameters, effective degrees of freedom, deviances, and
outer iteration counts match R.
