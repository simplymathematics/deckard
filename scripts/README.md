# Scripts

This directory contains local helper scripts for CI and development workflows.

## Coverage Runner

Use [scripts/coverage.sh](scripts/coverage.sh) to run tests, collect coverage, and capture timing artifacts.

```bash
bash scripts/coverage.sh
```

Optional test directory argument:

```bash
bash scripts/coverage.sh test/test_score
```

Notes:

- The script resolves repository paths from its own location, so it can be invoked from any current working directory.
- It writes outputs to `build/` at repository root (`coverage.txt`, `timing.txt`, `runtime.txt`, and logs).
- It expects the project interpreter at `.venv/bin/python`.

## Test GitHub Workflows Locally

Use [scripts/test_workflow.sh](scripts/test_workflow.sh) to run any workflow with `act`.

### GPU mode selection

`test_workflow.sh` supports runtime mode selection for Docker-oriented workflows:

- `--gpu-mode auto|cpu|mps|cuda`
- `auto` resolves to:
	- `mps` on macOS
	- `cuda` when NVIDIA GPUs are detected via `nvidia-smi`
	- `cpu` otherwise

Examples:

```bash
./test_workflow.sh --workflow docker-test.yml --gpu-mode mps --dry-run
./test_workflow.sh --workflow docker-test.yml --gpu-mode cpu --dry-run
./test_workflow.sh --workflow docker-test.yml --gpu-mode cuda --dry-run
```

The runner exports these environment variables to `act`:

- `DECKARD_GPU_MODE`
- `DECKARD_DOCKER_IMAGE_TAG`
- `DECKARD_DOCKER_BUILD_ARGS`

### Generate a GitHub token for local runs

Use [scripts/generate_github_token.sh](scripts/generate_github_token.sh) to get a token from your authenticated GitHub CLI session.

```bash
# One-time login if needed
gh auth login

# Export in current shell
export GITHUB_TOKEN="$(./generate_github_token.sh --plain)"

# Then run your workflow test
./test_workflow.sh --workflow compile-docs.yml --job docs --ref refactor-squashed --verbose
```

You can also print an export command directly:

```bash
eval "$(./generate_github_token.sh --export)"
```

Or write to a local env file:

```bash
./generate_github_token.sh --write-env-file .act.env
```

### List available workflows

```bash
./test_workflow.sh --list
```

### Dry-run a workflow command

```bash
./test_workflow.sh --workflow compile-docs.yml --job docs --ref refactor-squashed --dry-run
```

### Run a workflow job

```bash
./test_workflow.sh --workflow compile-docs.yml --job docs --ref refactor-squashed
```

### Notes

- Requires `docker` and `act` for non-dry runs.
- Install `act` on macOS with `brew install act`.
- If `--ref` is omitted, the current git branch is used.