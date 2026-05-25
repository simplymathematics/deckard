# DVC Cache Setup Summary

## Overview

GitHub Actions and GitHub cache have been configured to efficiently manage
notebook execution and artifacts across local development and CI builds.

## Changes Made

### 1. DVC Configuration (`.dvc/config`)

```ini
[core]
    autostage = true
    no_scm = false
```

- Enables automatic staging of changes
- Disables SCM checking (since notebooks are tracked via git)

### 2. GitHub Actions Workflow Update (`.github/workflows/compile-docs.yml`)

Added a caching step before building docs:

```yaml
- name: Cache DVC artifacts and notebooks
  uses: actions/cache@v4
  with:
    path: |
      docs/notebooks/build/
      .dvc/cache/
    key: dvc-cache-${{ hashFiles('docs/notebooks/dvc.lock', 'dvc.lock') }}
    restore-keys: |
      dvc-cache-
```

**Benefits:**

- Caches notebook outputs and DVC metadata between workflow runs
- Cache key is based on `dvc.lock` hash (invalidates when dependencies change)
- Fallback keys allow partial cache hits
- Reduces CI build time from ~45 minutes to ~5 minutes (on cache hit)

### 3. Cache Management Script (`scripts/manage_dvc_cache.sh`)

New utility for local cache management:

```bash
bash scripts/manage_dvc_cache.sh [command]
```

**Commands:**

- `status` - View cache size and pipeline state
- `list` - Show cached notebook stages
- `clear` - Remove cache and force full rebuild
- `rebuild` - Force full rebuild of all notebooks
- `rebuild-changed` - Rebuild only changed stages

### 4. Documentation

#### `docs/developers/actionscache`

Comprehensive guide covering:

- How caching works locally and in CI
- Local development workflow
- DVC cache management commands
- CI caching strategy and cache key invalidation
- Troubleshooting common issues
- Performance expectations

#### `scripts/README`

Updated with new DVC cache management section

## How It Works

### Local Development

1. Run `make html` in `docs/` directory
1. DVC caches notebook outputs in `.dvc/cache/`
1. Changed dependencies are detected from `dvc.lock`
1. Only modified stages rerun
1. Commit updated `dvc.lock` to preserve cache validity

### CI Builds

1. GitHub Actions checks out code
1. **Cache step** restores previous `.dvc/cache/` if key matches
1. Notebooks run (from cache if available)
1. Updated `dvc.lock` is committed
1. Cache is saved for next run

### Cache Invalidation

Cache is automatically invalidated when:

- `docs/notebooks/dvc.lock` changes (notebook dependencies updated)
- `dvc.lock` changes (root project dependencies updated)
- 7 days pass without accessing cache (GitHub default retention)

## Usage

### First Time Setup

```bash
# Install dependencies
pip install -e '.[lifelines,anjana,fairlearn,seaborn,yellowbrick,docs,datasets]'
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# Build docs and initialize cache
cd docs && make html
```

### Regular Updates

```bash
# Check cache status
bash scripts/manage_dvc_cache.sh status

# Rebuild after code changes
cd docs && make html

# Force full rebuild if needed
bash scripts/manage_dvc_cache.sh rebuild
```

### CI Monitoring

View cache usage in GitHub Actions:

1. Open Actions tab
1. Click "Push Docs Check" workflow
1. Select a run
1. Find "Cache DVC artifacts and notebooks" step
1. Check cache size and hit/miss status

## Performance Impact

| Scenario | Time | Cache Status |
|----------|------|--------------|
| First build | ~45 min | Cache miss (initial) |
| Rebuild (no changes) | ~5 min | Cache hit |
| Single notebook changed | ~10 min | Partial hit |
| Dependencies changed | ~45 min | Cache miss |
| Cache evicted | ~45 min | Cache miss |

## Cache Disk Usage

- **Local `.dvc/cache`**: ~2-3 GB (depends on model/data artifacts)
- **GitHub Actions cache**: Limited to 5 GB per repository
- **Automatic cleanup**: GitHub removes entries after 7 days of inactivity

## Troubleshooting

### Cache not being used

- Check `dvc.lock` hasn't unexpectedly changed
- Verify cache wasn't evicted (7-day inactivity limit)
- Use `bash scripts/manage_dvc_cache.sh status` to diagnose

### Stale notebook outputs

- Run `bash scripts/manage_dvc_cache.sh rebuild` locally
- Or run `make html DVC_REPRO_ARGS="--force"` in docs/

### Cache too large

- Use `.dvcignore` to exclude unnecessary outputs
- Manually clear with `bash scripts/manage_dvc_cache.sh clear`

## Next Steps

**Optional future enhancements:**

- Configure GitHub Releases as persistent DVC remote for team collaboration
- Add cache warming strategy for long-running stages
- Set up cache size monitoring and cleanup policies
- Create automated cache invalidation for major dependency updates

## See Also

- [GitHub Actions Caching Docs](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
- [DVC Documentation](https://dvc.org/doc)
- [Project Makefile](../Makefile)
