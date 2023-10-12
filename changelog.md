
# scBoolSeq Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
<!--
Types of changes:
    * `Added` for new features.
    * `Changed` for changes in existing functionality.
    * `Deprecated` for soon-to-be removed features.
    * `Removed` for now removed features.
    * `Fixed` for any bug fixes.
    * `Security` in case of vulnerabilities.
-->


## [0.2.0] - 2023-10-12

### Added

- Compliance with scikit-learn Transformer API.
- Usage of scikit-learn's `set_output` API to output pandas.DataFrames (keep gene and observation names).

### Changed

- Coarse-graining of Zero-Inflated genes is now performed as zero-or-not instead of quantile.

### Removed 

- All R dependencies: scBoolSeq is now implemented in pure Python, built on top of pandas and scikit-learn.
- Temporarily, the CLI has been removed as it needs major refactoring to accomodate the API's changes.

## [0.1.0] - 2022-05-13

First version of scBoolSeq, buit as a wrapper around a series of R scripts.

