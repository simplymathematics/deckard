# Core, Framework, And Plugin Boundaries

This page documents the intended execution boundaries between the core layer,
framework layer, and plugin layer.

## Core Layer

Core modules under `deckard/data`, `deckard/model`, `deckard/attack`,
`deckard/detector`, `deckard/experiment`, and `deckard/score` provide:

- Public orchestration `*Config` APIs
- Deterministic lifecycle execution (`__post_init__`, `__call__`)
- Runtime composition boundaries
- Framework-agnostic defaults and shared behavior

Core packages should not eagerly import sibling plugin families during module
import. Optional plugin and framework symbols should be lazily resolved.

## Framework Layer

Framework modules under `deckard/frameworks/<framework>` provide:

- Concrete framework-specific implementations of shared contracts in
  `deckard/frameworks/core.py`
- Adapter bridge behavior from core configs to framework runtime objects
- Optional dependency logic local to each framework family

Framework implementations must remain isolated from sibling frameworks.

## Plugin Layer

Plugin modules under `deckard/plugins/<family>` provide:

- Optional domain integrations (for example fairlearn, anjana, lifelines)
- Family-specific extensions to data/model/score/experiment/plot behavior
- Runtime hooks and plugin-specific composition logic

Plugin families must remain isolated from each other and from unrelated
framework modules.

## Adapter Boundary Rule

Adapter mixins in `deckard/frameworks/adapters.py` are the only cross-boundary
translation layer. Adapter methods must:

- Access only public attributes or methods on target config objects
- Avoid reading or mutating underscore-prefixed target attributes
- Exchange runtime state through typed public fields/accessors

## Validation Coverage

Boundary assumptions are validated by:

- `test/test_package/test_core_package_isolation.py`
- `test/test_package/test_framework_isolation.py`
- `test/test_package/test_plugin_family_isolation.py`
- `test/test_package/test_frameworks_package.py`
- `test/test_frameworks/test_core_contracts.py`
