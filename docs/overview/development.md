# Core software documentation:

This page is intended as a contributor map. Use it to locate the module area
you need to modify, then use the linked API docs for detailed object-level
behavior.

## Contributor Workflow

Typical contributor loop:

1. Identify the pipeline stage affected (`data`, `model`, `attack`, `score`,
   or `experiment`).
2. Update implementation and associated declarations/config wiring.
3. Add or update tests for behavior changes.
4. Update notebooks/docs when the user-facing behavior changes.
5. Re-run focused workflows (tests, docs build, notebook stage) before merge.

## Core Modules

- {doc}`/api/data`
- {doc}`/api/model`
- {doc}`/api/attack`
- {doc}`/api/detector`
- {doc}`/api/experiment`
- {doc}`/api/score`
- {doc}`/api/plot`
- {doc}`/api/layers`
- {doc}`/api/file`
- {doc}`/api/utils`

Extension documentation:

- {doc}`/api/pytorch`
- {doc}`/api/anjana`
- {doc}`/api/lifelines`
- {doc}`/api/seaborn`
- {doc}`/api/yellowbrick`

## Development Guidelines

- Prefer shared utility helpers over duplicated conversion/normalization logic.
- Keep configuration behavior deterministic and explicit rather than relying on
	implicit fallback behavior.
- Keep metric naming stable, especially for multi-attack and extension metrics,
	to preserve downstream report compatibility.
- When modifying notebook-driven workflows, validate with the corresponding DVC
	stage.

## Documentation Responsibilities

Code changes that affect behavior should include doc updates in at least one of
the following:

- API page updates when signatures or semantics change.
- Notebook narrative updates when workflow interpretation changes.
- Overview updates when architecture or user entry points change.