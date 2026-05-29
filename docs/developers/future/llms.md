# Experiment Plan: Mutation and Adversarial Robustness Evaluation for DEMI-MathAnalysis

## Objective

Evaluate whether LLMs trained or fine-tuned on DEMI-MathAnalysis perform genuine mathematical reasoning or rely on memorized proof templates. The study combines mutation testing (`mutmut`) with adversarial perturbation analysis to measure reasoning robustness, proof stability, and generalization. DEMI-MathAnalysis focuses on proof-oriented real analysis tasks spanning sequences, limits, series, continuity, differentiation, integration, and convexity. :contentReference[oaicite:0]{index=0}

## Experimental Conditions

### Models
- Baseline foundation model
- Fine-tuned DEMI-MathAnalysis model
- Fine-tuned + adversarially augmented model
- Fine-tuned + self-consistency / CoT variant

### Dataset Splits
- Training: `pretraining_data.csv`
- Evaluation: `benchmark_data.csv`
- Holdout adversarial benchmark (generated perturbations)

## Phase 0: Implementation


## Phase 1: Baseline Evaluation

Measure:
- Proof correctness (%)
- Theorem usage accuracy
- Logical consistency
- Hallucinated theorem rate
- Token efficiency
- Judge score (LLM + human subset)

Store all outputs for replay.

## Phase 2: Adversarial Perturbation Benchmark

Generate semantically equivalent perturbations:

### Surface Perturbations
- Variable renaming
- Symbol substitution
- Reordering assumptions
- Alternate mathematical notation

### Semantic Perturbations
- Equivalent theorem formulations
- Contrapositive statements
- Hidden intermediate lemmas
- Distractor assumptions

### Reasoning Traps
- Near-identical false premises
- Ambiguous quantifiers
- Boundary-condition modifications
- Missing regularity assumptions

Measure:

Robust Accuracy (RA):

RA = Correct(original ∩ perturbed) / Total

A large drop indicates shortcut learning rather than genuine reasoning.

## Phase 3: Proof Mutation Testing

Create a proof-evaluation package and run `mutmut`.

Mutation categories:

### Mathematical Mutations
- Reverse inequalities
- Change convergence/divergence claims
- Replace existence with uniqueness
- Alter quantifiers
- Remove proof steps

### Programmatic Mutations
- Corrupt theorem retrieval
- Modify prompt templates
- Alter chain-of-thought scaffolding
- Disable verification modules

For each mutant:

Mutation Score = Killed Mutants / Total Mutants

A mutant is "killed" when benchmark performance significantly degrades.

Target:
- >90% mutation score for evaluation pipeline
- >80% mutation score for retrieval/prompt modules

## Phase 4: Robustness Stress Testing

Evaluate:

### Distribution Shift
- New textbook sources
- Olympiad-style analysis problems
- Advanced undergraduate analysis

### Noise Robustness
- OCR corruption
- LaTeX corruption
- Incomplete hypotheses

### Multi-Turn Proof Refinement
- Initial proof
- Critique round
- Revision round

### ART Family Attacks
- Malware Adaptation

Track proof improvement rate and self-correction rate.

## Metrics

| Metric | Description |
|----------|-------------|
| Accuracy | Final proof correctness |
| Robust Accuracy | Accuracy after perturbation |
| Mutation Score | Fraction of killed mutants |
| Proof Stability | Output similarity across perturbations |
| Calibration Error | Confidence vs correctness |
| Hallucination Rate | Invalid theorem citations |
| Self-Correction Rate | Improvement after critique |

## Success Criteria

- Accuracy drop under perturbation <10%
- Mutation score >90%
- Hallucination rate <5%
- Self-correction rate >25%
- Robust accuracy significantly higher than baseline model

## Deliverables

1. Mutation score report
2. Adversarial benchmark dataset
3. Robustness leaderboard
4. Error taxonomy
5. Statistical significance analysis
6. Reproducible evaluation pipeline

## Expected Outcome

If DEMI-MathAnalysis produces genuine reasoning improvements, fine-tuned models should maintain proof correctness under semantic-preserving perturbations and exhibit high mutant-kill rates. Large performance degradation under minor reformulations would indicate reliance on memorized proof patterns rather than robust mathematical reasoning. This aligns with broader findings that mathematical benchmarks often overestimate true reasoning ability when adversarial variants are introduced. 