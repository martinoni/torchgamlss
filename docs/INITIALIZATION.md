# Fitting starting values

Every family defines R-compatible default starting values on the distribution
parameter scale. `fit_rs()`, `fit_cg()`, and `fit_minibatch()` validate and
expand them before optimization.

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

The same overrides are accepted by `fit_cg()`, `fit_cg_data()`,
`fit_minibatch()`, and `fit_minibatch_data()`. Mini-batch initialization
requires the first column of every design to be an intercept of ones. It
centers the full predictor on the linked starts while respecting existing
offset, smooth, neural, shared, and non-intercept contributions.

`fit_minibatch_loader()` accepts a complete mapping of scalar starts, one for
every family parameter. Streaming inputs cannot use response-wide defaults or
observation-level start vectors without retaining the population. A resumed
loader fit takes its state from the checkpoint and does not accept a second
set of initial values.

Torch L-BFGS currently starts from the model's coefficient tensors, which
advanced callers can set directly before `fit()` or `fit_data()`.
