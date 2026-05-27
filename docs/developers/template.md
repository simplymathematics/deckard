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

## Out of Scope For Developer Pages

- End-user onboarding/tutorial content
- Long beginner walkthroughs
- Repetition of API usage snippets unless needed for internal explanation

## API Counterpart

The matching template for API pages is in {doc}`../api/template`.
