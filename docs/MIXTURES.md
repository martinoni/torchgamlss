# Finite mixtures

`FiniteMixture` (alias `MX`) combines two or more existing TorchGAMLSS
families. For observation \(i\),

```text
f(y_i) = sum_k pi_ik f_k(y_i | theta_ik)
```

The implementation evaluates the likelihood with `torch.logsumexp`, keeping
the component densities and weights on the log scale until normalization.
It supports homogeneous or heterogeneous scalar-event components, provided
they are either all continuous or all discrete and the observed response is
valid for every component.

## Parameter names and mixing weights

Component parameters receive explicit names:

```python
from torchgamlss import FiniteMixture, Normal

family = FiniteMixture([Normal(), Normal()])
family.parameter_names
# (
#   "component_1_mu",
#   "component_1_sigma",
#   "component_2_mu",
#   "component_2_sigma",
#   "mixing_1",
# )
```

For \(K\) components, `mixing_1` through `mixing_{K-1}` are unrestricted
log-odds against component \(K\):

```text
mixing_k = eta_mixing_k = log(pi_k / pi_K)
```

This is the reference-category multinomial parameterization used when
`gamlss.mx` models prior probabilities. It guarantees positive probabilities
that sum to one. Obtain the normalized values with:

```python
parameters = model.predict_data(data)
prior = family.component_weights(parameters)
log_prior = family.component_log_weights(parameters)
```

Each mixing predictor may have its own formula or design matrix, so component
membership can vary with covariates.

## Fitting

An intercept-only two-Normal mixture is:

```python
from torchgamlss import FiniteMixture, GAMLSS, MixtureControl, Normal

family = FiniteMixture([Normal(), Normal()])
model = GAMLSS.from_formula(
    family,
    {
        "component_1_mu": "y ~ 1",
        "component_1_sigma": "~ 1",
        "component_2_mu": "~ 1",
        "component_2_sigma": "~ 1",
        "mixing_1": "~ 1",
    },
    data,
)
fit = model.fit_mixture_data(
    data,
    control=MixtureControl(
        tolerance=1e-6,
        max_iterations=200,
    ),
)
```

`fit_mixture_data()` runs a generalized expectation-maximization algorithm.
The E-step calculates posterior component probabilities. Each M-step uses
Torch L-BFGS on the expected complete-data objective, which allows all
component parameters and prior probabilities to retain independent linear or
fixed-lambda smooth predictors.

The main result fields are:

```python
fit.global_deviance
fit.negative_log_likelihood
fit.iterations
fit.converged
fit.deviance_history
fit.posterior_probabilities
fit.effective_counts
fit.effective_proportions
```

The deviance is checked after every M-step and is not allowed to increase
beyond floating-point tolerance. A component whose effective count falls
below `MixtureControl.minimum_effective_count` is treated as collapsed rather
than silently retained.

The tensor equivalent is `model.fit_mixture(response, design_matrices, ...)`.
The standalone `fit_mixture(model, ...)` function exposes the same operation.

## Shared component parameters

Parameters may share one predictor across every component:

```python
family = FiniteMixture(
    [Normal(), Normal()],
    shared_parameters=("sigma",),
)
family.parameter_names
# ("component_1_mu", "sigma", "component_2_mu", "mixing_1")
```

The shared name must exist and use the same link in every component. Its
single predicted value is passed to each component. Other parameters remain
component-specific.

This sharing is a TorchGAMLSS extension. `gamlssMX()` fits separate GAMLSS
objects and does not expose the same cross-component parameter constraint.

## Initialization and label ordering

Mixture likelihoods are invariant to relabeling exchangeable components.
Starting all components at the same coefficients would also create a
stationary symmetric fit. TorchGAMLSS therefore uses deterministic starts:

1. sort the response;
2. split it into \(K\) contiguous groups;
3. obtain family-specific starts within each group;
4. assign the lowest-response group to component 1;
5. start the prior probabilities uniformly.

For exchangeable components, `ordering_parameter` defaults to unshared `mu`.
The family exposes:

```python
order = family.component_order(parameters)
canonical = family.canonicalize_parameters(parameters)
```

Canonicalization is available only when every component has the same family,
parameter names, and links. Heterogeneous components have intrinsic labels
and are not automatically reordered.

## Posterior probabilities and diagnostics

Posterior probabilities use Bayes' rule on the stable log scale:

```python
posterior = model.posterior_probabilities(
    response,
    design_matrices,
)
```

`posterior` has shape `n x K`. For formula models, materialize the stored
designs with `prepare_formula_data()` or use the posterior tensor retained on
the fit result.

Component diagnostics summarize classification and separation:

```python
diagnostics = model.component_diagnostics(
    response,
    design_matrices,
    weights=case_weights,
)

diagnostics.classification
diagnostics.effective_counts
diagnostics.effective_proportions
diagnostics.entropy
diagnostics.mean_entropy
diagnostics.mean_max_posterior
```

Class labels are zero-based Torch indices. Effective counts, entropy, and
averages honor optional case weights.

## Distribution operations

Finite mixtures provide differentiable log density or mass, CDF, posterior
probabilities, means, variances, and sampling. Continuous-mixture quantiles
bracket the result with component quantiles and invert the mixture CDF by
bisection. These operations work with observation-specific parameters and
remain on the parameter device. Discrete-mixture quantiles are not yet
implemented.

The likelihood, posterior, moments, and gradients use Torch operations and
run on CUDA when the installed PyTorch build has CUDA support. A component
whose diagnostic CDF internally uses SciPy may still incur the same CPU
round-trip documented for that base family.

## R parity and current limits

Committed references target `gamlss.mx` 6.0-1. They compare `dMX()` and
`pMX()` for Normal, Gamma, and Poisson mixtures with observation-specific
parameters and probabilities. A two-Normal intercept fit compares final
component means, scales, prior and posterior probabilities, and global
deviance against `gamlssMX()`.

The current EM fitter supports linear and fixed-lambda smooth predictors. It
does not perform RS/CG working-derivative cycles or automatic smoothing
selection inside the M-step. Neural mixture predictors remain available
through the joint `fit()` and mini-batch APIs, but not yet through
`fit_mixture()`.
