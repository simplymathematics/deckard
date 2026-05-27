# API Page Template

Use this template for all pages in `docs/api/`.

API pages describe what a subsystem does and how users configure and run it.
They should avoid implementation contracts and internal invariants.

## Required Sections

1. `# <Topic>`
1. `## Basic flow state`
1. `## Capabilities`
1. `## Outputs`
1. `## Configuration`
1. `## Usage Examples`
1. `## Integrations`
1. `## API Reference`
1. `## Operations and Troubleshooting`
1. `## See also`

## Section Rules

- `Basic flow state`: one lifecycle line with canonical runtime order.
- `Capabilities`: user-facing behavior only.
- `Outputs`: concrete payloads/files produced at runtime.
- `Configuration`: YAML/Hydra-oriented setup guidance and defaults.
- `Usage Examples`: runnable snippets and practical workflows.
- `Integrations`: links to framework/plugin pages; do not duplicate internals.
- `API Reference`: public classes/functions/modules only.
- `Operations and Troubleshooting`: runtime tips, common failures, and fixes.
- `See also`: always include the developer counterpart page.

## Example Preservation Rule

- Do not remove existing API, YAML, or Hydra examples from API pages.
- If examples are reorganized, keep all existing examples under either
	`Configuration` or `Usage Examples`.

## Out of Scope For API Pages

- Contract requirements
- Internal invariants
- Hook/plugin execution internals
- Migration guardrails
- Contributor implementation notes

Put those in `docs/developers/` pages.

## Developer Counterpart

The matching template for developer pages is in {doc}`../developers/template`.
