# Mixin Contract

Detailed contract for public mixin classes (`*Mixin`).

## Purpose

Mixins isolate reusable behavior so multiple config/plugin families can share
logic without inheritance-heavy duplication.

## Capabilities

- Provide focused utility methods for data/model/score pipelines.
- Keep feature logic composable across optional extensions.
- Reduce coupling between orchestration and implementation details.

## Standards Followed

- Documentation standards: {doc}`/developers/contributor/documentation`
- Mixin rules: {doc}`/developers/extensions/plugins`
- Naming rules: {doc}`/developers/contributor/naming`

## Required Documentation

- One-line capability summary
- `Attributes:` section for class fields
- Public method docs with `Args:` and `Returns:` when applicable

## See also

- {doc}`/developers/extensions/plugins`
- {doc}`/developers/extensions/hooks`
- {doc}`/developers/design/orchestration`
