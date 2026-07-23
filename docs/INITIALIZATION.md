# RS starting values

Every family defines R-compatible default starting values on the distribution
parameter scale. `fit_rs()` validates and expands them before the first RS
cycle.

Any subset can be overridden with scalars or one-dimensional tensors:

```python
result = model.fit_rs(
    response,
    design_matrices,
    initial_parameters={
        "mu": 2.0,
        "sigma": sigma_start,
    },
)
```

A scalar is repeated for every observation. A vector must contain exactly one
value per response. Unspecified parameters retain the family default.
TorchGAMLSS rejects unknown names, incompatible shapes, non-finite values,
values outside the link domain, and combinations outside the distribution's
parameter space.

Formula models can also name data columns:

```python
result = model.fit_rs_data(
    data,
    initial_parameters={
        "mu": "mu_start",
        "sigma": 0.5,
    },
)
```

An override replaces only that parameter's default. For example, a constant
Normal response has no default sample standard deviation, but supplying a
positive `sigma` still retains the automatic `mu`.

This protocol belongs to the classical RS fitter. Torch L-BFGS currently
starts from the model's coefficient tensors, which advanced callers can set
directly before `fit()` or `fit_data()`.
