# Docs Refactor Checklist

This checklist tracks alignment work between developer docs and API docs.

## Audit Summary

Current state after audit:

- Core module pages exist for data/model/attack/detector/experiment/score/plot.
- Canon modules exist in source for core domains:
  - deckard/data/canon.py
  - deckard/model/canon.py
  - deckard/attack/canon.py
  - deckard/detector/canon.py
  - deckard/experiment/canon.py
  - deckard/score/canon.py
  - deckard/plot/canon.py
- API index now separates core modules, framework integrations, and plugin integrations.
- Several core API pages previously duplicated plugin/framework behavior details; this refactor is moving those to integration pages.

## Refactor Goals

- Developers docs: detailed architecture, capabilities, orchestration, and extension mechanics.
- API docs: user-facing runtime behavior, primary config entrypoints, and behavior changes only.
- Keep plugins/frameworks as separate integration pages, cross-linked from core pages.
- Mirror source layout and canon runtime contracts in docs structure and cross-links.

## Checklist

### 1) Information Architecture

- [x] Separate API navigation into Core Modules, Framework Integrations, Plugin Integrations.
- [x] Add matching high-level sections to developers index for Core, Frameworks, Plugins.
- [x] Add a short docs map in developers index describing where users vs developers should read first.

### 2) Template Enforcement and API/Developer Parity

Enforce one template per folder and one counterpart per page across folders.

Authoring rules:

- API pages describe what the subsystem does (usage, flow, configuration).
- Developer pages describe how and why the subsystem works (internals, contracts, rationale).
- Preserve existing API examples, including YAML and Hydra examples.

Template definition checklist:

- [x] Create API template page: {doc}`../api/template`.
- [x] Create developers template page: {doc}`template`.
- [x] Condense redundant sections in both templates while preserving required examples.

API template enforcement checklist (docs/api):

- [x] Require top intro triad in core pages: `Basic flow state`, `Capabilities`, `Outputs`.
- [x] Keep API content user-facing and remove contract-heavy sections from core API pages.
- [x] Ensure API template explicitly preserves API/YAML/Hydra examples.
- [x] Apply template uniformly to all API pages (`docs/api/*.md`).

Developer template enforcement checklist (docs/developers):

- [x] Define required sections for internals, contracts, validation, and compatibility.
- [x] Add missing core developer counterparts for API core pages (`detector`, `plot`).
- [x] Apply template uniformly to all developer pages (`docs/developers/*.md`).

Cross-folder parity checklist:

- [x] Create API -> developers parity map and track it in this checklist.
- [x] Create developers -> API parity map and track it in this checklist.
- [x] Add template/parity pages to index navigation.
- [x] Define parity rule as runtime-page parity, with process/standards pages explicitly marked `N/A` in parity maps.
- [x] Update parity tracking with matching `N/A` exceptions for process/standards pages.


Initial implementation status:

- [x] Began enforcement by updating API core pages (`data`, `model`, `score`) to API-focused scope.
- [x] Began enforcement by introducing developer counterparts for missing core topics.
- [x] Complete full-folder normalization pass (all API and developer pages).
- [x] Begin parity exception implementation for process/standards pages (`N/A` policy) in both parity maps.

Implementation summary:

- Runtime pages follow strict API <-> developer parity.
- Process/standards pages are intentionally mapped as `N/A` in parity maps.
- Next implementation step is runtime-only parity verification and cleanup of any non-runtime mirror artifacts.

Runtime-only cleanup implementation (started):

- [x] Established `N/A` exceptions for process/standards in both parity maps.
- [x] Populate stub files/sections in docs/api
- [x] Populate stub files/sections in docs/developers
- [x] Replace high-level placeholder prose in new `docs/developers/` pages with implementation-level details (modules, classes, execution contracts).

### 3) Core vs Integration Boundaries

- [x] Remove framework/plugin automodule blocks from core pages where equivalent integration pages exist.
- [x] Ensure each core page has a compact "Integrations" link section (frameworks/plugins only).
- [x] Ensure each integration page references its parent core module page and behavior deltas only.

### 4) Capability Consistency Across folders

- [x] For each domain (data/model/attack/detector/experiment/score/plot), verify "Purpose" and "Capabilities" language matches between developers and API docs.
- [x] Normalize terminology: "stage", "mode", "hook", "runtime owner", "plugin", "framework adapter".
- [x] Remove repeated inherited behavior text from API pages unless behavior differs from parent.

### 5) Plugin and Hook Positioning

- [x] Keep plugin rules/capabilities in developers/plugins.
- [x] Add developers/orchestration as the central execution model page.
- [x] Move "Extensions" (frameworks/plugins) out of core API and developer docs into dedicated extension sub-trees.


### 6) Sub-object Documentation

- [x] Data sub-objects: verify sampler/pipeline pages focus on callable behavior and link to developers/samplers and developers/pipelines.
- [x] Model sub-objects: verify train/defend pages focus on behavior deltas and link to developers/trainers and developers/defenses.
- [x] Clarify that ART retrainer defenses are distinct from trainer configuration objects in both API and developer docs.
- [x] Score sub-objects: ensure API and developers/scorers pages agree on screr-stage capability language.
- [x] Ensure sub-objects are grouped with their owning top-level object trees (data/model families) across overview/API/developers to parallel source layout.

### 7) Link and Warning Hygiene

- [x] Run docs build and capture warnings.
- [x] Fix `deckard.experiment.base.ExperimentConfig` docstring indentation error reported by Sphinx (`Unexpected indentation`).
- [x] Resolve duplicate object description warnings for DataConfig between data/pipeline pages.
- [x] Keep automodule directives valid and avoid malformed signatures/indentation.
- [x] Resolve ambiguous xrefs by using explicit MyST links when labels collide.

### 8) Final Validation

- [x] Verify docs build cleanly with nb execution off.
- [x] Spot-check 10 cross-links spanning core -> integration and API -> developers.
- [x] Confirm no stale references to removed/renamed pages.
