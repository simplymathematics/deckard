# GitHub Actions Workflows

This document provides a comprehensive overview of all GitHub Actions workflows in the Deckard project, their purposes, triggers, and current status.

## Workflow Categories

### Core Testing Workflows

#### `deckard-test.yml`
**Purpose:** Run base test suite on pull requests to main branch.

**Triggers:**
- Pull requests to `main` branch

**Jobs:**
- `base-tests`: Runs on Ubuntu latest with Python 3.10
  - Installs base dependencies (`.[test]`)
  - Executes pytest excluding optional-dependency tests
  - Skips: PyTorch, Fairness, Lifelines, Seaborn, Yellowbrick tests

**Status:**  Active

**Notes:**
- Runs only base dependencies to provide fast feedback
- Optional dependency tests are in separate workflows

---

#### `optional-dependency-test-reusable.yml`
**Purpose:** Reusable workflow template for testing optional dependencies.

**Type:** Reusable workflow (called by specific optional dependency workflows)

**Parameters:**
- `dependency-group`: Pip dependency group to install (e.g., `[fairlearn]`)
- `test-ignore-list`: Pytest ignore patterns for tests not using this dependency

**Status:**  Active

**Notes:**
- Centralizes optional dependency test logic
- Called by: fairlearn, lifelines, seaborn, torch, yellowbrick workflows

---

#### Optional Dependency Workflows
These workflows test the package with specific optional dependencies:

**Consolidated workflow (matrix-based):**
- `test-optional-dependencies.yml` - Tests all optional dependencies using matrix strategy
  - Jobs run in parallel with one per optional dependency
  - Covers fairlearn, lifelines, seaborn, torch, yellowbrick
  - Legacy optional workflows were removed after migration
  - Supports: fairlearn, lifelines, seaborn, torch, yellowbrick

**Triggers:** Pull requests to `main` branch

**Status:**  Active (single consolidated matrix workflow)

---

### Code Quality Workflows

#### `black.yml`
**Purpose:** Enforce code formatting standards using Black.

**Triggers:**
- Push to any branch
- Pull requests

**Jobs:**
- Runs Black formatter check on Python code

**Status:**  Active

**Notes:**
- Fast feedback on formatting violations
- Can be auto-fixed locally with `black .`

---

#### `repository-enforcement.yml`
**Purpose:** Enforce repository standards and best practices.

**Triggers:**
- Push events
- Manual workflow dispatch

**Jobs:**
- Validates file naming conventions
- Checks version consistency
- Enforces documentation standards

**Status:**  Active

**Notes:**
- Customizable enforcement rules
- See [Repository Enforcement Guide](../developers/development) for details

---

### Documentation Workflows

#### `compile-docs.yml`
**Purpose:** Build and validate Sphinx documentation with notebook execution.

**Triggers:**
- Push to `main` and `refactor-squashed` branches
- Manual workflow dispatch (supports cache override)

**Workflow Inputs (dispatch):**
- `cache_flush_token`: Optional cache namespace override (default: "stable")

**Jobs:**
- `docs`: Runs on Ubuntu latest with 60-minute timeout
  - Installs all optional dependencies (`[lifelines,anjana,fairlearn,seaborn,yellowbrick,docs,datasets]`)
  - Caches DVC artifacts and notebook outputs
  - Pulls DVC cache from remote
  - Executes all notebooks
  - Builds Sphinx docs with strict validation (`-n -W`)

**Cache Strategy:**
- Key: `dvc-cache-${OS}-${CACHE_FLUSH_TOKEN}-${HASH(dvc.lock, pyproject.toml)}`
- Allows cache invalidation by changing `cache_flush_token` in dispatch

**Status:**  Active

**Notes:**
- Long runtime due to notebook execution (10-15 min typical)
- Handles DVC cache failures gracefully
- Strict docs build prevents regressions

---

### Platform Build Workflows

#### Platform-Specific Builds (Legacy)
- `build_ubuntu.yml` - Ubuntu latest
- `build_macos.yml` - macOS latest
- `build_windows.yml` - Windows latest

#### Consolidated Platform Build (Matrix-Based)
- `platform-build.yml` - Builds package on all platforms using matrix strategy
  - Jobs run in parallel: ubuntu-latest, macos-latest, windows-latest
  - Single source of truth for build logic
  - Separate artifact per platform for easy debugging
  - 7-day retention on artifacts

**Purpose:** Verify package builds correctly on all supported platforms.

**Triggers:** 
- Pull requests to `main` branch
- Pushes to `main` or `plugins` branches
- Manual workflow dispatch

**Jobs:**
- Install dependencies
- Run `python -m build` (PEP 517 compliant)
- Upload platform-specific distribution artifacts

**Status:**  Active (both legacy and consolidated versions available)

**Recommendation:** Transition to `platform-build.yml` for faster parallel building and single maintenance point

---

### Docker Workflows

#### `docker-test.yml`
**Purpose:** Build and test Docker images for multiple variants (CPU, MPS, CUDA).

**Triggers:**
- Pull requests to `main` branch

**Matrix Variants:**
- `cpu`: Ubuntu 20.04 (no GPU support)
- `mps`: Ubuntu 20.04 (Apple Metal Performance Shaders)
- `cuda`: NVIDIA CUDA 12.0 on Ubuntu 20.04

**Environment Variables:**
- `DECKARD_APT_MIRROR_*`: Configure APT mirrors for builds
- `DECKARD_APT_*PROXY`: Proxy configuration for restricted networks

**Status:**  Active

**Notes:**
- Builds but doesn't push images
- Supports isolated network environments via proxy vars

---

#### `docker-push.yml`
**Purpose:** Build and push Docker images to container registry (GHCR).

**Triggers:**
- Manual workflow dispatch

**Matrix Variants:**
- Same as `docker-test.yml` (CPU, MPS, CUDA)

**Registry:** GitHub Container Registry (GHCR) at `ghcr.io/simplymathematics/deckard`

**Permissions:**
- `contents: read`
- `packages: write`

**Status:**  Manual trigger (requires explicit dispatch)

**Notes:**
- Only pushes on manual trigger to prevent accidental registry pollution
- Images tagged with branch name and commit SHA

---

### Release Workflows

#### `release-package.yml`
**Purpose:** Build and publish package to PyPI for releases.

**Triggers:**
- Manual workflow dispatch

**Jobs:**
- Build source distribution and wheels
- Publish to PyPI (uses trusted publishing)

**Status:**  Active

**Notes:**
- Requires PyPI trusted publisher configuration
- Used for official releases only

---

#### `release-package-tests.yml`
**Purpose:** Run comprehensive tests before release (all dependencies, all platforms).

**Triggers:**
- Manual workflow dispatch
- Can be run in isolation for pre-release validation

**Test Coverage:**
- All optional dependencies installed
- Runs full test suite
- Verifies all platforms (Ubuntu, macOS, Windows)

**Status:**  Active

**Notes:**
- More thorough than PR workflows
- Should be run before triggering `release-package.yml`

---

## Workflow Dependency Graph

```
Pull Request → main branch:
├── black.yml (quick, ~1 min)
├── deckard-test.yml (base tests + coverage, ~5 min)
├── test-optional-dependencies.yml (matrix, ~5-10 min)
├── platform-build.yml (matrix, ~2-3 min)
├── security-scan.yml (~5 min)
├── docker-test.yml (image build, ~10 min)
└── repository-enforcement.yml (standards, ~2 min)

Push → main/plugins branches:
├── compile-docs.yml (docs + notebooks, ~15 min)
└── docker-push.yml (build, scan, attest)

Manual Dispatch:
├── compile-docs.yml (with cache override)
├── docker-push.yml (build & push images)
├── release-package-tests.yml (pre-release validation)
└── release-package.yml (publish to PyPI)
```

## Workflow Timing & Performance

| Workflow | Typical Duration | Parallelizable | Timeout | Status |
|----------|------------------|----------------|---------|--------|
| black | ~1 min |  Yes | 10 min | Active |
| deckard-test | ~5 min |  Yes | 30 min | Active |
| test-optional-dependencies | 5-10 min |  Yes (5 parallel) | 35 min |  NEW |
| platform-build | 2-3 min |  Yes (3 parallel) | 20 min |  NEW |
| build_*-test (legacy) | 2-3 min |  Yes (3 parallel) | 20 min | Legacy |
| docker-test | ~10 min |  Sequential matrix | 60 min | Active |
| docker-push | ~15 min |  Sequential matrix | 90 min | Active |
| compile-docs | 10-15 min |  No | 60 min | Active |
| release-package | ~5 min |  Yes | 30 min | Active |
| repository-enforcement | ~2 min |  Yes | 10 min | Active |

**Total PR Check Time:** ~15-20 min (with parallelization)

**Key Improvements:**
-  All workflows have explicit `timeout-minutes` set
-  New consolidated workflows use matrix strategy for better parallelization
-  Artifacts have 7-day retention policy
-  Step timing/profiling added to key workflows

---

## Cache Strategy

### DVC Cache (Documentation)
- **Location:** `.dvc/cache/`, `docs/notebooks/build/`
- **Key Format:** `dvc-cache-${OS}-${CACHE_FLUSH_TOKEN}-${hash}`
- **TTL:** Default GitHub cache retention (~5-7 days)
- **Override:** Use `cache_flush_token` input in `compile-docs.yml` dispatch

### Pip Cache (via setup-python)
- **Location:** Managed by `actions/setup-python@v5`
- **Key:** Auto-generated from `requirements.txt` / `pyproject.toml`
- **TTL:** Default GitHub cache retention

---

## Workflow Dependencies & Orchestration

### Using `needs` Clauses

Workflows can specify dependencies using the `needs` clause to ensure proper sequencing and fail-fast behavior:

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check formatting
        run: black --check .

  tests:
    needs: lint  # Only runs if lint succeeds
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest
```

### Recommended Workflow Sequencing

For pull requests, recommend this execution order to fail fast:

```
1. Quick checks (format, lint): black.yml, repository-enforcement.yml
   ↓
2. Base tests: deckard-test.yml  
   ↓
3. Optional dependency tests (parallel): test-optional-dependencies.yml
   ↓
4. Platform builds (parallel): platform-build.yml
   ↓
5. Docker builds: docker-test.yml (optional, slower)
```

**Benefits:**
- Fast feedback on formatting/lint errors before expensive tests
- Platform builds only run if base tests pass
- Saves CI minutes by failing fast on preventable issues
- Better resource utilization

### Implementation Status

-  COMPLETED: Artifact retention policies (7-day for builds)
-  COMPLETED: Step timing/profiling (job duration reporting)
-  COMPLETED: Consolidated workflows created and active (`test-optional-dependencies.yml`, `platform-build.yml`)
-  COMPLETED: Security scanning and dependency auditing added (`security-scan.yml`, Dependabot)
-  COMPLETED: Failure notification and benchmark reporting added (`notify-failures.yml`, `workflow-benchmarks.yml`)
-  COMPLETED: Release gating and deployment control added (`deploy-release-gated.yml`)

---

## Permissions & Security

All workflows have been audited for least-privilege access:

- **Test workflows** (`black.yml`, `deckard-test.yml`, `test-optional-dependencies.yml`): `contents: read`
- **Build workflows** (build_*.yml): No explicit permissions (uses repository default)
- **Documentation** (compile-docs.yml): `contents: write` (needed for potential cache updates), `pages: write`, `id-token: write` (for deployment)
- **Docker workflows** (docker-test.yml): `contents: read, packages: read` (read-only)
- **Docker push** (docker-push.yml): `contents: read, packages: write` (only push capability)
- **Release** (release-package.yml, release-package-tests.yml): `contents: write` (for release creation)
- **Repository enforcement**: `contents: read` (audit only, no modifications)

**Security improvements implemented:**
-  All workflows have explicit `timeout-minutes` to prevent runaway jobs
-  Permissions minimized to necessary scopes only
-  Build workflows use read-only permissions where possible
-  Docker workflows limited to package registry access only

---

## Workflow Status Badges

You can display workflow status in your README or other documentation:

```markdown
### CI/CD Status

| Workflow | Badge |
|----------|-------|
| Tests | [![Python application](https://github.com/simplymathematics/deckard/actions/workflows/deckard-test.yml/badge.svg?branch=main)](https://github.com/simplymathematics/deckard/actions/workflows/deckard-test.yml) |
| Code Format | [![Lint](https://github.com/simplymathematics/deckard/actions/workflows/black.yml/badge.svg?branch=main)](https://github.com/simplymathematics/deckard/actions/workflows/black.yml) |
| Docs Build | [![Push Docs Check](https://github.com/simplymathematics/deckard/actions/workflows/compile-docs.yml/badge.svg?branch=main)](https://github.com/simplymathematics/deckard/actions/workflows/compile-docs.yml) |
| Docker | [![Test Docker Images](https://github.com/simplymathematics/deckard/actions/workflows/docker-test.yml/badge.svg?branch=main)](https://github.com/simplymathematics/deckard/actions/workflows/docker-test.yml) |
```

**Customizing badges:**
- Replace `simplymathematics/deckard` with your repository path
- Change `?branch=main` to any branch name (e.g., `?branch=plugins`)
- Omit `?branch=...` to show status for default branch

---

## Code Coverage & Reporting

### Adding Coverage Reports

To track code coverage over time, add coverage reporting to test workflows:

```yaml
- name: Run tests with coverage
  run: |
    pip install coverage
    python -m pytest test/ --cov=deckard --cov-report=xml --cov-report=term
    
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
    flags: unittests
    fail_ci_if_error: false  # Don't block CI if coverage fails
```

**Benefits:**
- Track coverage trends over time
- Identify under-tested code paths
- Optional: Block PRs if coverage drops below threshold
- Integrates with Codecov, Coveralls, or other services

**Implementation Status:**
-  COMPLETED: Coverage reporting added to `deckard-test.yml` and `test-optional-dependencies.yml`
- ⏳ TODO: Configure Codecov or similar service in repository settings

---

## Security & Dependency Scanning

### Adding Security Scanning (SAST)

Implement static analysis with Bandit:

```yaml
- name: Install Bandit
  run: pip install bandit
  
- name: Run security scan
  run: |
    bandit -r deckard -f json -o bandit-report.json
    bandit -r deckard -f screen  # Display results
```

**Benefits:**
- Detect common security issues (hardcoded secrets, SQL injection patterns, etc.)
- Prevent known vulnerability patterns
- Runtime: ~5 minutes

**Implementation Status:**
- ⏳ TODO: Create security-scanning.yml workflow
- ⏳ TODO: Add to repository enforcement checks

### Adding Dependency Scanning (Dependabot)

Enable GitHub's built-in Dependabot scanning:

**Steps:**
1. Go to repository Settings → Code security and analysis
2. Enable "Dependabot alerts", "Dependabot security updates", "Dependency graph"
3. GitHub will automatically create PRs for vulnerable dependencies

**Benefits:**
- Automatic CVE tracking and notifications
- Auto-create PRs for security patches
- Zero additional workflow setup needed
- No CI time cost

**Implementation Status:**
- ⏳ TODO: Enable Dependabot in repository settings

---

### Issue: Workflow Job Timeout
**Symptom:** Job fails with message "The job running on runner X has exceeded the maximum execution time of N minutes."

**Root Causes:**
- Dependency installation taking too long (network issues)
- Tests hanging indefinitely on a specific test case
- Large notebook execution timeout in docs build

**Solutions:**
1. Check workflow run logs for which step is slow
2. For test timeouts: Run locally to identify hanging test: `pytest test/ -v -s --timeout=30`
3. For build timeouts: Increase timeout in workflow (but prefer fixing the actual issue)
4. For docs: Check for notebooks with long execution time in `docs/notebooks/`
5. Use GitHub Actions runner groups if organizational network is congested

---

### Issue: DVC Cache Misses
**Symptom:** Workflow takes unexpectedly long, pulling from remote instead of cache.

**Root Causes:**
- `dvc.lock` or `pyproject.toml` changed unexpectedly
- Cache invalidated due to runner or cache cleanup
- DVC remote unreachable or missing artifacts

**Solutions:**
1. Verify `dvc.lock` and `pyproject.toml` haven't changed unexpectedly
2. Force cache refresh: Dispatch `compile-docs.yml` with new `cache_flush_token` value (e.g., "2026-05-refresh-20")
3. Check DVC remote configuration (see {doc}`gh_actions_cache`)
4. Verify DVC artifacts exist: `dvc status` and `dvc remote list`

---

### Issue: Permission Denied / Authentication Failures
**Symptom:** Workflow fails with "Permission denied", "authentication failed", or "403 Forbidden" errors.

**Root Causes:**
- Token/credentials not available in runner environment
- SSH keys not configured for GitHub Actions
- PyPI token expired or misconfigured
- Package registry credentials missing

**Solutions:**
1. For PyPI releases: Verify PyPI trusted publisher is configured
2. For private dependencies: Add GitHub token to pip install: `pip install --index-url https://token:${{ secrets.GITHUB_TOKEN }}@github.com/...`
3. For Git SSH: Generate and add SSH keys to GitHub Actions secrets
4. For Docker registry: Verify registry credentials in workflow permissions
5. Check GitHub Actions secret scopes: some secrets only available in PRs from same repo

---

### Issue: Workflow Fails on Specific Python Version
**Symptom:** Tests pass locally but fail in CI with different Python version.

**Root Causes:**
- CI uses different Python version than local environment
- Version-specific dependencies not handled correctly
- Syntax incompatibilities or behavior differences

**Solutions:**
1. Check CI workflow uses same Python as local: `python --version`
2. Test locally with CI's Python version: `pyenv install 3.10 && pyenv shell 3.10`
3. For version-specific tests: Use `sys.version_info` checks or use tox for multiple versions
4. Run `pip freeze` locally and compare with CI environment setup

---

### Issue: Flaky Tests in CI (Pass/Fail Randomly)
**Symptom:** Tests pass sometimes, fail other times, with no code changes.

**Root Causes:**
- Race conditions in concurrent tests
- Tests depending on execution order
- Random seed not set consistently
- Timing-dependent code (sleeps, timeouts)

**Solutions:**
1. Run tests locally multiple times: `for i in {1..5}; do pytest test/; done`
2. Run with fixed seed: `pytest test/ --randomly-seed=12345`
3. Check for tests modifying global state
4. Use pytest-timeout to catch hanging tests: `pip install pytest-timeout`
5. Run tests serially to rule out concurrency issues: `pytest test/ -n 0`

---
### Issue: Docker Build Fails in Restricted Networks
**Symptom:** `docker-test.yml` fails on APT package download.

**Solutions:**
1. Set organization variables:
   - `DECKARD_APT_MIRROR_PORTS`: Mirror port override
   - `DECKARD_APT_HTTP_PROXY`: HTTP proxy URL
   - `DECKARD_APT_HTTPS_PROXY`: HTTPS proxy URL
   - `DECKARD_APT_NO_PROXY`: Comma-separated no-proxy list

---

### Issue: Notebook Execution Timeouts
**Symptom:** `compile-docs.yml` fails with execution timeout.

**Solutions:**
1. Increase `nb_execution_timeout` in `docs/conf.py` (currently 1800 sec)
2. Optimize slow notebooks in `docs/notebooks/`
3. Increase GitHub Actions job timeout (currently unlimited)

---

## Related Documentation

- {doc}`Development Workflow <development>`
- {doc}`Repository Enforcement Guide <development>`
- {doc}`GH Actions Cache Setup <gh_actions_cache>`
- {doc}`Documentation Build Guide <../overview/build_docs>`
