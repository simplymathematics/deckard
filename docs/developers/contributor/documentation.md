# Documentation Standards and Build Guide

This page is the single source of truth for Deckard documentation standards.

It consolidates and supersedes distributed standards from:

- `scripts/repository_enforcement.py`
- `scripts/fix_docs_crosslinks.py`
- legacy docstring/build-docs notes previously split across developer pages

## Purpose and Scope

Deckard documentation is split by audience and runtime ownership:

- API docs explain what the subsystem does: usage, behavior, and configuration.
- Developer docs explain how and why the subsystem works: internals, contracts, and rationale.
- Runtime pages should keep API <-> developer parity; process/standards pages are allowed as explicit `N/A` parity exceptions.

## Information Architecture Standards

- Keep navigation separated into Core Modules, Framework Integrations, and Plugin Integrations.
- Keep framework/plugin extension details in integration pages, not core runtime pages.
- Keep sub-objects grouped under their owning runtime domain (for example data/model families).
- Keep terminology stable across docs: `stage`, `mode`, `hook`, `runtime owner`, `plugin`, and `framework adapter`.

## Authoring Standards

### API vs Developer Content Boundaries

- API pages are user-facing and should focus on runtime behavior and configuration.
- Developer pages are implementation-facing and should focus on contracts and internals.
- Preserve existing API YAML/Hydra examples when normalizing page templates.

### Cross-Folder Parity

- Maintain runtime-page parity between `docs/api` and `docs/developers`.
- Mark process/standards pages as `N/A` in parity maps when no runtime mirror is expected.

### Core vs Integration Boundaries

- Remove framework/plugin automodule content from core pages when integration pages exist.
- Add compact integration link sections from core pages to relevant framework/plugin docs.
- Integration pages should reference parent core pages and document only behavior deltas.

## Docstring Standard

All public Deckard docstrings must use MyST-native Google-style sections rendered by Sphinx/Napoleon.

Required sections:

- `Attributes:` for public classes with runtime/config fields (`*Config`, `*Mixin`, `*Plugin`, and runtime sub-objects such as samplers/pipelines/trainers/defenses/scorers)
- `Args:` when user-facing parameters are present
- `Returns:` when return value is non-`None`
- `Raises:` when exceptions are raised

Recommended sections:

- `Note:` for side effects and execution caveats
- `Example:` for canonical usage snippets

Syntax requirements:

- Do not use reStructuredText docstring markers (`:param`, `:type`, `:rtype:`, `.. code-block::`, `.. note::`).
- Use single-backtick inline code in docstrings.
- Use MyST roles for cross-references.
- Use fenced markdown blocks for examples.

## Enforcement Standards (`repository_enforcement.py`)

### Docs Markdown/Notebook Rules

- `DOCX001`: do not use legacy Sphinx `mod` role syntax in markdown/notebooks; use MyST roles.
- `DOCX002`: do not use legacy Sphinx `doc` role syntax in markdown/notebooks; use MyST roles.
- `DOCX003`: fix malformed legacy `mod`/`doc` role syntax in markdown/notebooks.
- `DOCX004`: inline-code references to public Deckard symbols must be linked.
- `DOCX005`: inline-code references to frameworks/plugins must be linked.

Docs enforcement scope defaults to:

- `docs` (all markdown pages and notebook markdown cells)

Override behavior:

- Use `--docs-scope <csv>` to validate selected docs trees.
- Use `--docs-scope none` to disable docs markdown/notebook checks.

### Docstring and Class Contract Rules

Default mode includes low-noise naming and shape checks and key docstring checks:

- `MIX004`, `MIX005`, `MIX006`, `MIX007` for mixin docstring coverage and no-RST policy
- plugin callable contract checks (`PLG001`, `PLG002`)

Strict mode (`--strict-docs-types`) adds:

- annotation requirements (`ANN001`, `ANN003`, `ANN004`)
- public method docstring requirements (`DOC001` to `DOC005`)

Optional class-level docstring field policy:

- `--require-attributes-sections` enforces `DOC006` for required class families.

### Canon Literal Exception Policy

`canon.py` literal token sets (mode/stage/alias/valid families and normalized variants) are exempted from plain inline-code link requirements to reduce false positives for canonical literals.

## Autofix Standards (`fix_docs_crosslinks.py`)

Use the fixer to bulk-convert inline-code references into links for markdown and notebook markdown cells.

Behavior:

- builds symbol/framework/plugin catalogs from source
- maps symbols to domain API pages using source-path-aware routing
- maps extension tokens to `docs/overview/extensions/*`
- skips fenced code blocks
- rewrites only inline-code tokens matching source-derived catalogs

Fallback behavior:

- unresolved mappings become `TODO-BROKEN-LINK` entries with an inline TODO comment
- TODO fallbacks are expected to be resolved to concrete docs targets in follow-up edits

## Build Docs Standard

Use this baseline docs build process for local validation:

1. Install docs dependencies.
2. Build from the `docs/` directory.
3. Resolve warnings before merge for affected pages.

Commands:

```bash
pip install -e .[docs]
cd docs
make html
```

Related CI/build contracts:

- Docs workflow: `compile-docs.yml`
- DVC-backed docs cache behavior: {doc}`actionscache`
- Workflow reference and troubleshooting: {doc}`workflows`

## Recommended Validation Sequence

Run this sequence before opening a docs-heavy PR:

```bash
python scripts/repository_enforcement.py --scope deckard --docs-scope docs/developers,docs/api
python scripts/fix_docs_crosslinks.py
python scripts/repository_enforcement.py --scope deckard --docs-scope docs/developers,docs/api
cd docs && make html
```

If the fixer introduces `TODO-BROKEN-LINK`, replace each placeholder with a concrete page or symbol link before final review.
