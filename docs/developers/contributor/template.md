# Developer Page Template

Use this template for all pages in `docs/developers/`.

Developer pages describe how a subsystem works internally and why design
choices exist. They define contracts, invariants, and extension rules.

## Required Sections

1. `# <Topic> Design and Contract`
1. `## Purpose and Rationale`
1. `## Internal Architecture`
1. `## Execution Model`
1. `## Contracts and Invariants`
1. `## Extension Points`
1. `## Validation and Guardrails`
1. `## Migration and Compatibility`
1. `## See also`

## Section Rules

- `Purpose and Rationale`: ownership boundary, tradeoffs, and design intent.
- `Internal Architecture`: core runtime objects and control flow.
- `Execution Model`: stage ordering and lifecycle semantics.
- `Contracts and Invariants`: non-negotiable behavior guarantees.
- `Extension Points`: supported plugin/framework customization points.
- `Validation and Guardrails`: failure modes, protections, and test paths.
- `Migration and Compatibility`: compatibility and deprecation behavior.
- `See also`: always include the API counterpart page.
- Cross links to other `docs/*` folders should move to the bottom.
- Write settled runtime behavior as canonical policy and avoid phase-scoped wording.

## Out of Scope For Developer Pages

- End-user onboarding/tutorial content
- Long beginner walkthroughs
- Repetition of API usage snippets unless needed for internal explanation

## API Counterpart

API pages describe what a subsystem does and how users configure and run it.
They should avoid implementation contracts and internal invariants.

## API Page Template

Use this template for all pages in `docs/api/`.

### Required Sections

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

### Section Rules

- `Basic flow state`: one lifecycle line with canonical runtime order.
- `Capabilities`: user-facing behavior only.
- `Outputs`: concrete payloads/files produced at runtime.
- `Configuration`: YAML/Hydra-oriented setup guidance and defaults.
- `Usage Examples`: runnable snippets and practical workflows.
- `Integrations`: links to framework/plugin pages; do not duplicate internals.
- `API Reference`: public classes/functions/modules only.
- `Operations and Troubleshooting`: runtime tips, common failures, and fixes.
- `See also`: always include the developer counterpart page.

### Example Preservation Rule

- Do not remove existing API, YAML, or Hydra examples from API pages.
- If examples are reorganized, keep all existing examples under either
	`Configuration` or `Usage Examples`.

### Out of Scope For API Pages

- Contract requirements
- Internal invariants
- Hook/plugin execution internals
- Migration guardrails
- Contributor implementation notes

Put those in `docs/developers/` pages.

### Developer Counterpart

Use this page as the source of truth for developer-side template structure.
