# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project is versionned in the `YY.MM` format corresponding to the month in which the model was first used in production.

## [Unreleased]

## [21.03] - March 2021
### Added
- Created the `predictsignauxfaibles` python package
- Created [`SFDataset`](/predictsignauxfaibles/data.py) to help query data from MongoDB
- Created [`OversampledSFDataset`](/predictsignauxfaibles/data.py) to oversample firms that are in default
- Created projet [config](/predictsignauxfaibles/config.py) based on [environment](https://12factor.net/config).
- Created [`is_random`](/predictsignauxfaibles/decorators.py) decorator to help with reproducibility
- Created [`make_sf_train_test_splits`](/predictsignauxfaibles/model_selection.py) to help us perform unbiased cros-validation
- Created modular [pipelines](/predictsignauxfaibles/pipelines.py) of [`Preprocessors`](/predictsignauxfaibles/preprocessors.py) to perform preprocessing tasks
- Created ML models [configuration files](/models/) written in python
- Created a [CLI](/predictsignauxfaibles/__main__.py) that consumes and runs models based on their conf files
- Created basic [unit tests](/tests/unit/)
- Created and enforced data science development [workflow](datascience_workflow.md)

### Changed
- Nothing, it's our first release :smile: :tada:

### Removed
- Nothing, it's our first release :smile: :tada:

### Fixed
- Nothing, it's our first release :smile: :tada:


## [21.09] - September 2021
### Added
- Created documentation using Sphinx
- Created synthetic data generation capabilities
- Created [explainability module](/predictsignauxfaibles/explain.py)
- Created a [CLI](/predictsignauxfaibles/__main__.py) to easily parameterize model training runs
- Created [model config files](/models)
- Created "redressements experts" capacities [starting with a URSSAF-based rule](/predictsignauxfaibles/redressements.py)
- Created [evaluation module](/predictsignauxfaibles/evaluate.py)

### Fixed
- fill missing fields after fetching data
- merging redundant operations on data.columns
- force siren and sirets to be strings + pad them to 9 characters
- downgrade jedi version to fix auto-complete bug with ipython.
