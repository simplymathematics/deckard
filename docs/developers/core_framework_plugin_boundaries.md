# Core, Framework, and Plugin Boundaries

This document summarizes the execution boundaries and responsibilities for each layer in Deckard.

## Core Layer
- Public orchestration `*Config` APIs
- Deterministic lifecycle execution (`__post_init__`, `__call__`)
- Framework-agnostic defaults and shared behavior

## Framework Layer
- Concrete implementations of shared contracts (see `deckard/frameworks/core.py`)
- Adapter bridge logic from core configs to framework runtime objects
- Optional dependency logic local to each framework

## Plugin Layer
- Optional domain integrations (e.g., fairlearn, anjana, lifelines)
- Family-specific extensions to data/model/score/experiment/plot
- Runtime hooks and plugin-specific composition logic

## Adapter Boundary Rule
- Adapter mixins in `deckard/frameworks/adapters.py` are the only cross-boundary translation layer
- Adapter methods must access only public attributes or methods on target config objects
- No reading or mutating underscore-prefixed target attributes
- All runtime state exchanged through typed public fields/accessors

---

**Related:** [Design Principles](design_principles.md) | [Adapter Contract](adapter_contract.md)
