# Sampler Contract

Detailed contract for sampler sub-objects under data runtime.

## Purpose

Sampler objects define how datasets are partitioned into train/test/validation
splits for downstream model and scoring stages.

## Capabilities

- Implement deterministic or randomized split strategies.
- Support task-aware partitioning such as stratification.
- Emit split indices compatible with DataConfig runtime flows.

## Standards Followed

- Docstring standard: {doc}`docstrings`
- Data design: {doc}`data`

## Required Documentation

- Sampling intent and split semantics
- `Attributes:` for class-scoped controls (`test_size`, `n_splits`, etc.)
- Return shape contract for index payloads

## See Also
- {doc}`../api/sample`
