# Core, Framework, and Plugin Boundaries

This document summarizes the execution boundaries and responsibilities for each layer in Deckard.

## Core Layer
- Public orchestration `*Config` APIs
- Deterministic lifecycle execution (`__post_init__`, `__call__`)
- Framework-agnostic defaults and shared behavior
- Reusable chunks exist as Mixins so that sampling/training/defense behavior can be composed at run-time

## Frameworks
- Concrete implementations of shared contracts (see `deckard/frameworks/core.py`)
- Adapter bridge logic from core configs to framework runtime objects
- Optional dependency logic local to each framework

## Plugins
- Optional domain integrations (e.g., fairlearn, anjana, lifelines)
- Family-specific extensions to data/model/score/experiment/plot
- Runtime hooks and plugin-specific composition logic


---

**Related:** [Design Principles](design_principles.md) | [Adapter Contract](adapter_contract.md)
