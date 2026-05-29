# Canon Runtime Execution Guide

This page centralizes runtime execution order and contract details derived from
core canon modules:

- deckard/data/canon.py
- deckard/model/canon.py
- deckard/attack/canon.py
- deckard/detector/canon.py
- deckard/experiment/canon.py
- deckard/score/canon.py
- deckard/plot/canon.py

The API pages focus on user-facing behavior; canon contract internals live here.

## Data

Execution order:

1. `pre-load`
2. `pre-sample`
3. `post-sample`
4. `post-pipeline`
5. score and persist

Key runtime capabilities:

- Normalized score stages and split modes.
- Canonical runtime fields for split payloads and labels.
- Canonical timing payload keys for load/sample/pipeline/score.

See also: {doc}`../../api/data`.

## Model

Execution order:

1. initialize model
2. train
3. score/evaluate
4. persist outputs

Defense stage order is canonicalized as:

1. `pre_art_defense`
2. `pre_fit`
3. `post_fit_pre_predict`

Key runtime capabilities:

- Canonical trainer alias normalization.
- Canonical model runtime/timing fields.
- Canonical score-mode normalization ([train](../../api/modules), `test`, `val`).

See also: {doc}`../../api/model`, {doc}`../../api/train`, {doc}`../../api/defend`.

## Attack

Execution order:

1. normalize mode and stage
2. generate attack payloads
3. produce attack predictions
4. score and persist attack artifacts

Key runtime capabilities:

- Canonical attack stage aliases (`pre-attack`, `post-attack`).
- Canonical split mode validation (`auto`, [train](../../api/modules), `test`, `val`).
- Canonical attack timing keys.

See also: {doc}`../../api/attack`.

## Detector

Execution order:

1. normalize detector stage
2. fit detector
3. detect payloads
4. score and persist detector artifacts

Key runtime capabilities:

- Canonical detector stage aliases (`pre-fit`, `post-fit`, `pre-detect`, `post-detect`).
- Canonical detector runtime/timing fields.

See also: {doc}`../../api/detector`.

## Experiment

Execution order (high-level):

1. [load](../../api/modules)
2. [sample](../../api/modules)
3. [train](../../api/modules)
4. `defense`
5. [attack](../../api/modules)
6. [score](../../api/modules)
7. `persist`

Expanded component-level stages are canonized in experiment runtime helpers for
fine-grained orchestration and hook routing.

Key runtime capabilities:

- Canonical stage-component mapping.
- Canonical run-mode aliases.
- Canonical experiment runtime schema/version/hash support.

See also: {doc}`../../api/experiment`, {doc}`../orchestration`.

## Score

Execution order:

1. normalize score mode
2. normalize stage tokens
3. execute scorer payload contract
4. emit canonical score output

Key runtime capabilities:

- Canonical scorer mode validation.
- Canonical stage token normalization.
- Typed scorer runtime contract payload.

See also: {doc}`../../api/score`, {doc}[score](../../api/modules).

## Plot

Execution order:

1. normalize backend
2. execute plotting pipeline
3. persist plot runtime payload

Key runtime capabilities:

- Canonical backend normalization ([seaborn](../../overview/extensions/index), [yellowbrick](../../overview/extensions/index)).
- Typed plot runtime contract for files/times/state.

See also: {doc}`../../api/plot`.

## Purpose and Rationale

Define ownership boundaries, design intent, and tradeoffs for this domain.

## Internal Architecture

Describe runtime components, data flow, and orchestration boundaries.

## Execution Model

Describe canonical stage ordering and lifecycle semantics.

## Contracts and Invariants

Define non-negotiable behavior guarantees and invariant runtime contracts.

## Extension Points

Describe framework/plugin extension surfaces and constraints.

## Validation and Guardrails

List failure modes, guardrails, and validating tests.

## Migration and Compatibility

Document migrations, aliases, and compatibility expectations.

## See also

- {doc}`../../api/modules`
