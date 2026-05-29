# Security Report (Developer)

Date: 2026-05-27  
Repository: simplymathematics/deckard  
Branch validated: refactor-squashed

## Summary

- Security workflow install path was updated to use a baseline constraints file at `constraints/security-baseline.txt`.
- The security workflow still fails at the Bandit scan step (expected while unresolved findings remain), but installation and tooling setup complete successfully.
- The base Deckard test workflow succeeds after the installation strategy changes.
- B324 is intentionally excluded from Bandit scans as accepted non-cryptographic hash usage noise.
- B615 was remediated by requiring explicit model revision pinning for non-local Hugging Face model loads.

## Current Verification Status

### 1. Security Workflow

Workflow: `security-scan.yml`  
Result: failed at Bandit stage, install stage passed

Observed outcome from local ACT validation:

- Install security tooling: success
- Run Bandit SAST (`-s B324`): failure (non-zero findings)
- Job status: failed

Key install evidence in the run shows constrained baselines resolving as intended:

- `setuptools` installed at `82.0.1`
- `gitpython` resolved to `3.1.50`
- `urllib3` resolved to `2.7.0`
- `idna` resolved to `3.15`

Evidence log: `build/workflow-runs/security-scan.yml.log`

### 2. Deckard Test Workflow

Workflow: `deckard-test.yml`  
Result: succeeded

Observed outcome from local ACT validation:

- Install dependencies: success
- Test suite execution: success
- Final result: `1206 passed, 11 skipped, 4 xfailed`

Evidence log: `build/workflow-runs/deckard-test.yml.log`

## Implemented Security/Packaging Changes

1. Baseline constraints strategy

- Added `constraints/security-baseline.txt` as a security floor for known vulnerable packages.
- Updated `security-scan.yml` install commands to use `-c constraints/security-baseline.txt`.

2. Cleaner direct dependency model

- Removed transitive-focused direct entries from `pyproject.toml` (`urllib3`, `gitpython`, `idna`, `mistune`, `starlette`) to keep the runtime dependency set cleaner.
- Kept the baseline floor in constraints and security workflow resolution.

3. Transformer supply-chain hardening (B615)

- `GenericFlexibleTransformer` now requires `model_revision` for non-local model names.
- `AutoConfig.from_pretrained(...)` and `AutoModel.from_pretrained(...)` now pass explicit `revision=`.

## Risk Notes

- B324 (weak MD5 hash) is tracked as accepted risk for non-cryptographic stable identifiers and cache keys.
- B301/B614 (unsafe loading) remain trust-boundary-sensitive and primarily depend on whether users load untrusted artifacts.
- Security scan remains fail-fast on unresolved Bandit findings, which is intentional for visibility.

## How to Generate a New Security Report

Use this process whenever security dependencies, workflow logic, or scan policy changes.

### 1. Run the security workflow locally

```bash
set -o pipefail
scripts/test_workflow.sh \
	--workflow security-scan.yml \
	--event workflow_dispatch \
	--ref refactor-squashed \
	--quiet \
	--platform ghcr.io/catthehacker/ubuntu:act-latest \
	| tee build/workflow-runs/security-scan.yml.log
```

### 2. Run the base Deckard test workflow

```bash
set -o pipefail
scripts/test_workflow.sh \
	--workflow deckard-test.yml \
	--event pull_request \
	--ref refactor-squashed \
	--quiet \
	--platform ghcr.io/catthehacker/ubuntu:act-latest \
	| tee build/workflow-runs/deckard-test.yml.log
```

### 3. Capture headline outcomes for the report

```bash
grep -nE "Job succeeded|Job failed|Failure - Main|Install security tooling|Run Bandit SAST" build/workflow-runs/security-scan.yml.log | tail -n 40
grep -nE "Job succeeded|Job failed|Install dependencies|Test base dependencies only|=+ [0-9]+ passed" build/workflow-runs/deckard-test.yml.log | tail -n 40
```

### 4. Refresh report content

Update this page (`docs/developers/security-report.md`) with:

- Date and branch validated
- Security workflow result and stage of failure/success
- Deckard test workflow result and test totals
- Any policy updates (for example, scan excludes or baseline constraints changes)
- Any remediation status changes (implemented, deferred, or accepted risk)

### 5. Keep evidence pointers current

Ensure these references remain valid in the report:

- `build/workflow-runs/security-scan.yml.log`
- `build/workflow-runs/deckard-test.yml.log`
- `constraints/security-baseline.txt`

## Next Actions

1. Continue triage/reduction of remaining Bandit findings in core runtime paths.
2. Add a scheduled baseline refresh workflow for `constraints/security-baseline.txt`.
3. Optionally publish a brief release-security note when baseline floors are updated.
