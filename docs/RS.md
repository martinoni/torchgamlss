# Linear RS fitting

TorchGAMLSS includes an initial translation of the Rigby-Stasinopoulos (RS)
algorithm used by the R `gamlss` package. This implementation currently fits
unpenalized linear design matrices. It does not yet implement additive
backfitting, smoothers, or smoothing-parameter estimation.

## Working iteration

For distribution parameter `theta_k`, link predictor `eta_k`, offset `o_k`,
score `u_k`, and expected second log-likelihood derivative `h_k`, define

```text
d_k = d eta_k / d theta_k
w_k = -h_k / d_k^2
z_k = (eta_k - o_k) + u_k / (d_k w_k)
```

Each inner iteration solves weighted least squares for `z_k` using weights
`w_k` multiplied by the observation likelihood weights. The fitted predictor
is combined with the offset, transformed through the inverse link, and used to
recompute the global deviance. The outer cycle updates `mu`, `sigma`, and any
future parameters sequentially.

The working weights are clipped to `[1e-10, 1e10]`, matching the safeguards in
the R source. Automatic step halving is used when an inner update increases the
global deviance.

## Controls

`RSControl` exposes separate inner and outer tolerances and iteration limits,
a step length, automatic step halving, and an allowed deviance increase. Its
defaults match `gamlss.control()` and `glim.control()` where applicable.

## Verified scope

The parity fixture fits a heteroscedastic normal model with:

```text
mu ~ x + offset(mu_offset)
log(sigma) ~ z + offset(sigma_offset)
```

It uses nonuniform likelihood weights. TorchGAMLSS matches R `gamlss` 5.5-0
in all four coefficients, global deviance, log likelihood, convergence status,
and number of outer RS cycles.
