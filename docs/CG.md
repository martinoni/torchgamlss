# Cole-Green fitting

TorchGAMLSS provides a parametric implementation of the Cole-Green (`CG`)
algorithm from R `gamlss`. Unlike RS, which completes a separate inner fit for
each distribution parameter, CG uses the expected cross derivatives to update
all parameter predictors in a joint inner cycle.

For parameter `theta_k`, predictor `eta_k`, score `u_k`, and expected second
derivatives `h_kj`, define:

```text
d_k    = d eta_k / d theta_k
w_kk   = -h_kk / d_k^2
w_kj   = -h_kj / (d_k d_j)
z_k    = eta_k(old) - offset_k + step_k u_k / (d_k w_kk)

z*_k   = z_k - sum[j != k] w_kj (eta_j - eta_j(old)) / w_kk.
```

Each inner pass fits `z*_k` by weighted least squares and immediately makes
the new predictor available to the following parameter. This is the
Gauss-Seidel order used by `gamlss(..., method=CG())`. After the inner loop
converges, the scores and complete working-weight matrix are recomputed in the
next outer iteration.

## Formula API

```python
from torchgamlss import CGControl, GAMLSS, Beta

model = GAMLSS.from_formula(
    Beta(),
    {
        "mu": "y ~ x + offset(mu_offset)",
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
deviance increase.

`CGFitResult` reports the final deviance, convergence status, outer deviance
history, the number of inner passes used by every outer iteration, and
parametric effective degrees of freedom.

## Numerical behavior

CG is more sensitive to starting values than RS because the cross-parameter
working system is updated jointly. This is particularly visible for Box-Cox
families whose R-compatible identity link for `mu` does not prevent an
intermediate negative value. Supply suitable starts, smaller parameter steps,
or an explicit positive link when necessary.

R's automatic step-halving code changes final link predictors without
rewriting the associated `lm.wfit` coefficient object. TorchGAMLSS halves the
coefficients as well, keeping the fitted model and reported deviance
consistent. The BCT parity fixture disables automatic halving in both
implementations to compare the CG equations without this ambiguous R edge
case. The Beta fixture uses the defaults and does not encounter it.

## Verified scope

Committed R fixtures cover:

- a weighted two-parameter Beta model with formulas and offsets;
- a weighted four-parameter BCT model exercising all six cross-derivative
  blocks;
- coefficients, global deviance, negative log likelihood, convergence, and
  outer iteration counts;
- one-parameter fitting, control validation, and explicit rejection of smooth
  terms.

CG currently supports parametric linear predictors only. `pb()` and other
smooth terms remain available through RS; CG joint backfitting and
smoothing-parameter updates are future work.
