# Config Declaration Architecture

This document defines the refactor design for [Hydra](https://hydra.cc) config
declaration discovery and registration.

## Canonical Sources

Authoritative config declarations live in:

- `examples/sklearn/configs/`
- `examples/pytorch/configs/`
- External roots provided via `DECKARD_CONFIG_DIRS` (optional)

Plugin declaration families that commonly register through these roots include:

- [Fairlearn plugin API](/api/plugins/fairlearn)
- [Lifelines plugin API](/api/plugins/lifelines)
- [Seaborn plugin API](/api/plugins/seaborn)
- [Yellowbrick plugin API](/api/plugins/yellowbrick)
- [Anjana plugin API](/api/plugins/anjana)

The `deckard/` Python package should provide runtime registration logic, not
authoritative declaration content.

## Runtime Registration Model

`deckard/declarations.py` is the runtime registration entrypoint.

Expected responsibilities:

- discover config roots (built-in + `DECKARD_CONFIG_DIRS`)
- iterate YAML declaration files under those roots
- parse declarations into [Hydra](https://hydra.cc)-compatible registration metadata
- register declarations dynamically with {func}`deckard.utils.safe_store`
- gate framework/plugin registration on installed optional dependencies

## Hydra Organization

Use canonical [Hydra](https://hydra.cc) config groups with this structure:

- `<group>@<sub-group>/<plugin>_<alias-name>`

Examples:

- `model@defense=fairlearn_exponeniated-gradient.yaml`
- `attack/evasion_fgsm`

YAML declaration file names should use `modified_snake-case.yaml`.
Separate logical groups with `_`.
Use `-` in multi-word aliases.

- Prefer YAML declarations over Python hardcoded `ConfigStore` calls.
- Remove duplicated declarations in module-local declaration files once YAML
  equivalents exist.
- Keep registration deterministic and side-effect-safe for optional dependencies.
- Keep declaration loading extensible for downstream users via `DECKARD_CONFIG_DIRS`.

## Canonical Declaration Alignment

This section defines how runtime dispatch and config-group structure stay
aligned while preserving declaration trees.

### Goal

Keep declaration trees intact, but make runtime ownership and declaration
selection unambiguous at the component/sub-component boundary.

### Scope

- Runtime dispatch: attacks, defenses, frameworks, plugins.
- Declaration groups: `examples/sklearn/config/` and `examples/pytorch/config/`.
- Runtime registration: deckard/declarations.py and safe_store usage.

### Non-Goals

- No changes to experiment semantics or score definitions.
- No new compatibility shims.
- No broad plugin behavior redesign (documented in plugin consolidation guides).

### Problem Statement

Current declaration and runtime surfaces still include overlapping groups and
wrapper-style indirection that make ownership ambiguous. We do not want to
flatten declaration trees, but we do want to remove behavior-duplicate routing
and make component/sub-component selection explicit.

### Target State

Use a one-owner/one-tree model:

- Attack runtime owner: AttackConfig with canonical attack/... tree paths.
- Defense runtime owner: DefenseConfig and DefensePipelineConfig with one
  canonical defense/... tree.
- Framework runtime owner: framework-specific model/data components grouped
  under canonical `frameworks/...` trees.
- Plugin runtime owner: HookPlugin and plugin package declaration groups with
  one canonical plugin namespace per plugin family.

Config trees should be discoverable by component and sub-component, not by
historical aliases.

### Proposed Rules

1. Preserve declaration trees (`component/sub-component/name`) as the primary
   organization model.
2. Remove behavior-duplicate indirection entries that do not change runtime
   behavior.
3. Keep canonical runtime ownership in dispatch logic while allowing tree
   diversity for framework/plugin families.
4. Preserve optional dependency gating at registration time.
5. Keep examples and docs aligned in the same PR slice that changes groups.

### Proposed CLI Syntax for Declarations and Sub-Components

Use component/sub-component selection directly in CLI commands:

1. `deckard declarations list <component>`
2. `deckard declarations list <component>/<subcomponent>`
3. `deckard declarations show <component>/<subcomponent>/<name>`
4. `deckard declarations validate <component>/<subcomponent>/<name>`
5. `deckard declarations compose <component>/<subcomponent>/<name> [--set key=value ...]`

Examples:

- `deckard declarations list model`
- `deckard declarations list model/defense`
- `deckard declarations show attack/evasion/fgsm`
- `deckard declarations validate frameworks/pytorch/default_model`
- `deckard declarations compose model/defense/fairlearn_exponentiated-gradient --set defense_params.eps=0.1`

Optional flags:

- `--root sklearn|pytorch|external`
- `--format tree|json|yaml`
- `--resolve`
- `--strict`

### Config-Group Consolidation Plan

Rationalize by category without flattening trees:

- Attack declarations:
  - keep `attack/<subcomponent>/...` tree structure
  - remove parallel wrappers that only redirect without behavior changes
- Defense declarations:
  - keep `defense/<subcomponent>/...` tree structure
  - remove wrappers that point to identical runtime behavior
- Framework declarations:
  - keep canonical `frameworks/<framework>/...` trees
  - remove aliases that duplicate existing framework behavior
- Plugin declarations:
  - keep plugin-family trees and namespaces
  - remove alias entries that do not change runtime behavior

### Runtime Dispatch Alignment Plan

- Keep component/sub-component dispatch explicit in canonical runtime owners.
- Remove dispatch remapping that exists only for obsolete alias names.
- Keep plugin hooks stage-scoped while removing duplicate pass-through layers.

### Implementation Sequence

1. Inventory declaration tree nodes as canonical, behavior-duplicate, or alias-only.
2. Remove behavior-duplicate alias entries while preserving tree shape.
3. Update `deckard/declarations.py` registration and CLI lookup to component/sub-component paths.
4. Remove obsolete runtime dispatch remapping for deleted aliases.
5. Update docs and notebooks to canonical tree paths and CLI forms.
6. Validate via representative slices and update changelog acceptance notes.

### Acceptance Criteria

Declaration alignment is complete when all are true:

- Runtime dispatch for attacks/defenses/frameworks/plugins is aligned to
  canonical owner paths.
- Declaration trees remain intact and discoverable by component/sub-component.
- Behavior-duplicate declaration entries or indirection are removed.
- Docs and examples use canonical tree paths and CLI syntax in the same slice.

### Validation Plan

- Compose-level validation:
  - representative component/sub-component declarations compose without fallback
  - removed aliases fail fast with clear error context
- Runtime-level validation:
  - representative runtime slices reach expected owners and hook boundaries
- Docs validation:
  - developers and overview docs reference canonical component/sub-component paths

### Risks and Mitigations

- Risk: hidden downstream dependence on removed group aliases.
  - Mitigation: remove in one hard-cut, but document exact canonical
    replacements in changelog and migration notes.
- Risk: declaration removal causes optional-plugin compose regressions.
  - Mitigation: keep dependency-gated registration tests for each plugin family.
- Risk: notebooks/examples drift from runtime names.
  - Mitigation: require docs/example updates in the same PR as declaration changes.

### Deliverables

- Preserved but rationalized declaration trees in `examples/sklearn/config/` and
  `examples/pytorch/config/`.
- Updated runtime registration and declarations CLI lookup in `deckard/declarations.py`.
- Updated developer docs and examples to canonical component/sub-component paths.
- Test evidence from representative-slice composition and focused runtime suites.

## Test Strategy

- Compose representative slices: verify [Hydra](https://hydra.cc) composition
  using one representative declaration per component/sub-component family,
  rather than cross-product composition across all declaration combinations.

Use:

```bash
deckard optimize --cfg job
```

- Unit tests: consume canonical composed representative slices instead of
fixture-local declaration copies.

- Experiment tests: validate end-to-end execution from representative composed
config stacks that cover each runtime owner boundary.

- Use representative slices instead of cross-product validation.

______________________________________________________________________

**Related:** [Refactor Plan](../refactor_plan) | [Naming
Conventions](../naming) | {doc}`Mixin and Plugin Rules <../extensions/plugins>`
