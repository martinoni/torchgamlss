# Cole-Green fitting

TorchGAMLSS provides a linear and additive implementation of the Cole-Green
(`CG`) algorithm from R `gamlss`. Unlike RS, which completes a separate inner
fit for each distribution parameter, CG uses the expected cross derivatives
to update all parameter predictors in a joint inner cycle.

For parameter `theta_k`, predictor `eta_k`, score `u_k`, and expected second
derivatives `h_kj`, define:

```text
d_k    = d eta_k / d theta_k
w_kk   = -h_kk / d_k^2
w_kj   = -h_kj / (d_k d_j)
z_k    = eta_k(old) - offset_k + step_k u_k / (d_k w_kk)

z*_k   = z_k - sum[j != k] w_kj (eta_j - eta_j(old)) / w_kk.
```

Without smooth terms, each inner pass fits `z*_k` by weighted least squares.
With smooth terms, it performs one `additive.fit()`-style backfitting pass and
retains the smooth state for the next inner pass. The new predictor is
immediately available to the following parameter. This is the Gauss-Seidel
order used by `gamlss(..., method=CG())`. After the inner loop converges, the
scores and complete working-weight matrix are recomputed in the next outer
iteration.

## Formula API

```python
from torchgamlss import CGControl, GAMLSS, Beta

model = GAMLSS.from_formula(
    Beta(),
    {
        "mu": "y ~ pb(x) + offset(mu_offset)",
        "sigma": "~ z + offset(sigma_offset)",
    },
    data,
)

fit = model.fit_cg_data(
    data,
    weights="weight",
    control=CGControl(
        outer_tolerance=1e-7,
        inner_tolerance=1e-7,
        max_outer_iterations=200,
        max_inner_iterations=200,
    ),
)
```

The low-level equivalent is `model.fit_cg(response, design_matrices, ...)`.
Both interfaces accept likelihood weights, parameter-specific offsets, and
parameter-scale starting values. `CGControl` also exposes separate `mu`,
`sigma`, `nu`, and `tau` step lengths, automatic step halving, and an allowed
deviance increase. Backfitting, smoothing updates, target-EDF root finding,
and GAIC/GCV optimization have independent numerical tolerances and iteration
limits.

`CGFitResult` reports the final deviance, convergence status, outer deviance
history, the number of inner and additive passes, fitted smoothing parameters,
smooth and parameter effective degrees of freedom, and smoothing-selection
iteration counts.

## Numerical behavior

CG is more sensitive to starting values than RS because the cross-parameter
working system is updated jointly. This is particularly visible for Box-Cox
families whose R-compatible identity link for `mu` does not prevent an
intermediate negative value. Supply suitable starts, smaller parameter steps,
or an explicit positive link when necessary.

R's automatic step-halving code changes final link predictors without
rewriting the associated `lm.wfit` coefficient object. TorchGAMLSS halves the
linear and smooth coefficients as well, keeping the fitted model and reported
deviance consistent. The BCT parity fixture disables automatic halving in both
implementations to compare the CG equations without this ambiguous R edge
case. The Beta fixtures use the defaults and do not encounter it.

Target-EDF, GAIC, and GCV selection compare the final numerical result rather
than requiring identical cycle counts. R uses `uniroot()` or `nlminb()` on
`lambda`, while TorchGAMLSS uses log-scale root finding or bounded Brent
minimization. These paths can take different numbers of cycles while selecting
the same smoothing level within numerical tolerance.

When both parameter predictors contain smooths, R's separately stored linear
and smooth component objects can differ from the reconstructed final
predictor at its stopping tolerance. The committed fixture uses tighter
controls: fitted parameters and deviance agree at float64 precision, while
individual stored components agree within approximately `2e-6`.

## Verified scope

Committed R fixtures cover:

- a weighted two-parameter Beta model with formulas and offsets;
- a weighted four-parameter BCT model exercising all six cross-derivative
  blocks;
- a weighted three-parameter TF model exercising its complete expected
  derivative matrix;
- a weighted three-parameter PE model exercising its complete expected
  derivative matrix;
- a weighted Beta model combining offsets, cross derivatives, and a fixed
  `pb()` term;
- Normal additive models with fixed, ML-selected, target-EDF, GAIC-selected,
  and GCV-selected smoothing parameters;
- simultaneous `pb()` terms for Beta `mu` and `sigma`, with fixed lambdas and
  with independent ML selection;
- two separate `pb()` terms in the same `mu` predictor;
- linear and smooth coefficients, fitted parameter and smooth values,
  smoothing parameters, effective degrees of freedom, and global deviance;
- coefficients, global deviance, negative log likelihood, convergence, and
  outer iteration counts;
- one-parameter fitting and control validation.

CG supports the current one-dimensional `pb()` implementation, all
smoothing-selection modes described in [`SMOOTHS.md`](SMOOTHS.md), and
fixed-lambda analytic joint covariance for penalized terms. Broader smooth
families remain future work.
