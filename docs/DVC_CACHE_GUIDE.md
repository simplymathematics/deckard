# DVC Cache Guide


API documentation is generated from these docstrings and available in the [Core API](api/modules) section. For executable examples, see [Notebooks](notebooks/index).


This document describes how to use DVC caching for efficient local and CI notebook builds.

## Overview

The Deckard project uses DVC (Data Version Control) to manage notebook execution stages and cache their outputs. This reduces build times by:
- **Locally**: Caching notebook outputs so reruns only execute changed dependencies
- **In CI**: GitHub Actions cache stores `.dvc/cache` between runs, preventing redundant notebook executions

## Architecture

### DVC Pipeline Structure

- **Location**: `docs/notebooks/dvc.yaml`
- **Stages**: 12 notebook execution stages (seaborn, pytorch, fairlearn, etc.)
- **Outputs**: Generated plots, metrics, and model artifacts in `docs/notebooks/build/`
- **Dependencies**: Notebook files, source code modules, and auxiliary data

### Cache Strategy

1. **Local Development**:
   - DVC caches outputs in `.dvc/cache/` directory
   - When dependencies change, only affected stages rerun
   - Full cache is version-controlled in git (via `.gitignore`)

2. **CI Builds (GitHub Actions)**:
   - `actions/cache@v4` caches `.dvc/cache/` between workflow runs
   - Cache key: Hash of `dvc.lock` files (tracks when artifacts are valid)
   - Fallback: If cache misses, notebooks rerun and new outputs are cached

## Local Workflow

### Initial Setup

```bash
# Install project with all optional dependencies
pip install -e '.[lifelines,anjana,fairlearn,seaborn,yellowbrick,docs,datasets]'
pip install --index-url https://download.pytorch.org/whl/cpu --force-reinstall torch torchvision torchaudio

# Navigate to docs
cd docs
```

### Build Notebooks with Caching

```bash
# Execute all notebook stages with cache
make html

# Or run individual stage (e.g., notebook_pytorch)
make notebooks DVC_STAGE=notebook_pytorch

# Force rebuild all notebooks (ignore cache)
make html DVC_REPRO_ARGS="--force"

# Reproduce only changed stages
make notebooks
```

### DVC Cache Management

```bash
# View cache status
dvc cache dir          # Show cache location
dvc status             # See which stages need rerunning
dvc dag                # Visualize pipeline DAG

# Clear cache (forces full rebuild next time)
rm -rf .dvc/cache

# Check cache hits
dvc repro --dry       # Show what would run without --force

# View DVC metrics and plots
dvc plots show
dvc metrics show
```

## CI Workflow

### How GitHub Actions Caching Works

1. **Workflow Run**:
   - Checkout code
   - Set up Python
   - **Restore cache**: Download cached `.dvc/cache/` if available
   - Install dependencies
   - Run `make html` (calls `dvc repro`)
   - **Save cache**: Upload new/updated `.dvc/cache/` if changed

2. **Cache Key**:
   ```
   dvc-cache-${{ hashFiles('docs/notebooks/dvc.lock', 'dvc.lock') }}
   ```
   - Changes when `dvc.lock` files change (indicating pipeline changes)
   - Ensures stale cache isn't used after code updates

3. **Cache Paths**:
   - `docs/notebooks/build/`: Notebook outputs (plots, metrics, models)
   - `.dvc/cache/`: DVC's internal cache storage

### Viewing CI Cache

```bash
# On GitHub Actions tab:
1. Click on "Push Docs Check" workflow
2. Click a recent run
3. Scroll to "Set up cache" step to see cache size and hits
```

## Troubleshooting

### Cache isn't being used (slow CI)

1. Check `dvc.lock` hasn't changed unexpectedly
2. Verify cache wasn't evicted (GitHub keeps cache for 7 days of inactivity)
3. Check workflow for cache step errors

### Stale outputs after code change

1. **Clear local cache**: `rm -rf .dvc/cache`
2. **Force CI rebuild**: 
   - Run `make html DVC_REPRO_ARGS="--force"` locally
   - Commit updated `dvc.lock`
   - Or manually trigger workflow and watch for cache misses

### Large cache size

- DVC cache can grow with notebook outputs (models, plots, data)
- GitHub Actions cache limit: 5 GB per repository
- If cache exceeds limit, older entries are evicted
- Solution: Use `.dvcignore` to exclude unnecessary outputs

## Advanced: GitHub Releases for Persistent Storage

To maintain a persistent cache across all machines and indefinitely:

```bash
# (Optional future enhancement)
# Configure DVC remote to push to GitHub Releases:
dvc remote add -d releases 'remote://github-releases'
dvc push releases
```

This requires additional scripting; contact maintainers for setup.

## Performance Notes

- **Expected cache hit rate**: ~95% on routine builds (only changed stages rerun)
- **First build time**: ~30-45 minutes (all stages from scratch)
- **Cache hit build time**: ~2-5 minutes (Sphinx HTML generation only)
- **Cache eviction**: 7 days without access; recreated on next run

## See Also

- [DVC Documentation](https://dvc.org/doc)
- [GitHub Actions Cache](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
- [Project Makefile](https://github.com/simplymathematics/deckard/blob/main/docs/Makefile) for all available targets
- [DVC Pipeline](https://github.com/simplymathematics/deckard/blob/main/docs/notebooks/dvc.yaml) for stage definitions
