# Changelog

All notable changes to TorchGAMLSS are documented in this file.

The project follows semantic versioning and uses the Python packaging
pre-release notation described by PEP 440.

## [0.1.0a1] - 2026-07-26

First installable alpha release.

### Added

- Normal, Gamma, Poisson, negative-binomial type I, Beta, BCCG, BCT, and BCPE
  response families with R-generated numerical parity fixtures.
- Formula and tensor interfaces with independent predictors, links, weights,
  offsets, P-splines, neural predictors, and shared representations.
- Rigby-Stasinopoulos and Cole-Green fitting, plus a Torch-native L-BFGS
  baseline and streaming Adam optimization.
- Fixed, ML, target-EDF, GAIC, and GCV smoothing-parameter selection.
- CUDA mini-batch fitting, FP16/BF16 automatic mixed precision, checkpointing,
  resumption, and validation early stopping.
- Parametric and smooth inference, simultaneous bands, joint bootstrap
  covariance, derived smooth functionals, and response-scale quantile bands.
- Model-selection diagnostics, randomized quantile residuals, four-panel
  residual plots, worm plots, and bucket plots.
- Python 3.10-3.13 test matrix on Linux and Windows.
- Wheel and source-distribution validation with a clean-environment smoke
  test.

### Known limitations

- This is an alpha release and is not intended for production or high-stakes
  statistical analysis without independent validation.
- Only one-dimensional, equally spaced P-spline bases are implemented.
- The bucket-plot family-locus overlays stored in R-specific serialized assets
  are not bundled.
- Public package-index publication has not been enabled.
