#!/usr/bin/env bash
set -euo pipefail

# Generic local runner for GitHub Actions workflows via act.
# Defaults are branch-aware so branch filters behave as expected.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKFLOW=""
JOB=""
EVENT_NAME="push"
REF_NAME=""
EVENT_FILE=""
ARCH="linux/amd64"
PLATFORM_IMAGE="ghcr.io/catthehacker/ubuntu:full-latest"
LIST_ONLY=0
DRY_RUN=0
VERBOSE=0
GITHUB_TOKEN_VALUE="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
GPU_MODE="auto"
RESOLVED_GPU_MODE=""

usage() {
  cat <<'EOF'
Usage: scripts/test_workflow.sh [options]

Options:
  -w, --workflow <name|path>  Workflow file name (e.g. compile-docs.yml) or path
  -j, --job <job_id>          Run a specific job id (optional)
  -e, --event <event_name>    Event type for act (default: push)
  -r, --ref <branch>          Branch name for event payload (default: current git branch)
      --event-file <path>     Use a custom event payload JSON
      --arch <value>          Container architecture (default: linux/amd64)
      --platform <image>      Image for ubuntu-latest
      --list                  List available workflow files and exit
      --verbose               Enable verbose act output (includes docker pull details)
      --github-token <token>  GitHub token for act (fallback: GITHUB_TOKEN or GH_TOKEN env)
      --gpu-mode <mode>       GPU mode: auto|cpu|cuda|mps (default: auto)
      --dry-run               Print the act command and exit
  -h, --help                  Show this help

Examples:
  scripts/test_workflow.sh --list
  scripts/test_workflow.sh --workflow compile-docs.yml --job docs
  GITHUB_TOKEN=ghp_xxx scripts/test_workflow.sh --workflow compile-docs.yml --job docs
  scripts/test_workflow.sh --workflow docker-test.yml --gpu-mode cpu
  scripts/test_workflow.sh -w .github/workflows/deckard-test.yml -e pull_request
EOF
}

list_workflows() {
  find "$ROOT_DIR/.github/workflows" -maxdepth 1 -type f \( -name "*.yml" -o -name "*.yaml" \) \
    -print | sed "s#^$ROOT_DIR/##" | sort
}

resolve_workflow_path() {
  local input="$1"

  if [[ -z "$input" ]]; then
    echo "Error: --workflow is required unless --list is used." >&2
    exit 2
  fi

  if [[ -f "$input" ]]; then
    echo "$input"
    return
  fi

  if [[ -f "$ROOT_DIR/$input" ]]; then
    echo "$ROOT_DIR/$input"
    return
  fi

  if [[ -f "$ROOT_DIR/.github/workflows/$input" ]]; then
    echo "$ROOT_DIR/.github/workflows/$input"
    return
  fi

  echo "Error: workflow file not found: $input" >&2
  exit 2
}

detect_gpu_mode() {
  case "$GPU_MODE" in
    auto)
      if [[ "$(uname -s)" == "Darwin" ]]; then
        RESOLVED_GPU_MODE="mps"
      elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        RESOLVED_GPU_MODE="cuda"
      else
        RESOLVED_GPU_MODE="cpu"
      fi
      ;;
    cpu|cuda|mps)
      RESOLVED_GPU_MODE="$GPU_MODE"
      ;;
    *)
      echo "Error: invalid --gpu-mode '$GPU_MODE'. Expected: auto|cpu|cuda|mps" >&2
      exit 2
      ;;
  esac

  if [[ "$RESOLVED_GPU_MODE" == "cuda" ]] && \
     ! (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1); then
    cat <<'EOF'
Warning: --gpu-mode cuda was selected but no NVIDIA GPU was detected via nvidia-smi.
Continuing with cuda mode as requested.
EOF
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--workflow)
      WORKFLOW="$2"
      shift 2
      ;;
    -j|--job)
      JOB="$2"
      shift 2
      ;;
    -e|--event)
      EVENT_NAME="$2"
      shift 2
      ;;
    -r|--ref)
      REF_NAME="$2"
      shift 2
      ;;
    --event-file)
      EVENT_FILE="$2"
      shift 2
      ;;
    --arch)
      ARCH="$2"
      shift 2
      ;;
    --platform)
      PLATFORM_IMAGE="$2"
      shift 2
      ;;
    --list)
      LIST_ONLY=1
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    --github-token)
      GITHUB_TOKEN_VALUE="$2"
      shift 2
      ;;
    --gpu-mode)
      GPU_MODE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$LIST_ONLY" -eq 1 ]]; then
  list_workflows
  exit 0
fi

WORKFLOW_PATH="$(resolve_workflow_path "$WORKFLOW")"
detect_gpu_mode

if [[ -z "$REF_NAME" ]]; then
  REF_NAME="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD)"
fi

cleanup_event_file() {
  if [[ -n "$EVENT_FILE_CREATED" && -f "$EVENT_FILE_CREATED" ]]; then
    rm -f "$EVENT_FILE_CREATED"
  fi
}

EVENT_FILE_CREATED=""
if [[ -z "$EVENT_FILE" ]]; then
  EVENT_FILE_CREATED="$(mktemp)"
  EVENT_FILE="$EVENT_FILE_CREATED"
  cat > "$EVENT_FILE" <<JSON
{
  "ref": "refs/heads/${REF_NAME}"
}
JSON
  trap cleanup_event_file EXIT
fi

if [[ -z "$GITHUB_TOKEN_VALUE" ]]; then
  if command -v gh >/dev/null 2>&1; then
    if gh auth status >/dev/null 2>&1; then
      GITHUB_TOKEN_VALUE="$(gh auth token 2>/dev/null || true)"
      if [[ -n "$GITHUB_TOKEN_VALUE" ]]; then
        echo "Info: using token from gh auth token"
      fi
    else
      cat <<'EOF'
Warning: gh is installed but not authenticated.
Run: gh auth login
EOF
    fi
  else
    cat <<'EOF'
Warning: gh is not installed.
Install with: brew install gh
EOF
  fi
fi

CMD=(
  act "$EVENT_NAME"
  -W "$WORKFLOW_PATH"
  -e "$EVENT_FILE"
  --container-architecture "$ARCH"
  -P "ubuntu-latest=$PLATFORM_IMAGE"
  --env "DECKARD_GPU_MODE=$RESOLVED_GPU_MODE"
)

if [[ "$RESOLVED_GPU_MODE" == "cuda" ]]; then
  CMD+=( --env "DECKARD_DOCKER_IMAGE_TAG=deckard:cuda" )
  CMD+=( --env "DECKARD_DOCKER_BUILD_ARGS=--build-arg ENABLE_CUDA=1 --build-arg BASE_IMAGE=nvidia/cuda:12.0.0-runtime-ubuntu20.04" )
elif [[ "$RESOLVED_GPU_MODE" == "mps" ]]; then
  CMD+=( --env "DECKARD_DOCKER_IMAGE_TAG=deckard:mps" )
  CMD+=( --env "DECKARD_DOCKER_BUILD_ARGS=--build-arg ENABLE_CUDA=0 --build-arg BASE_IMAGE=ubuntu:20.04" )
else
  CMD+=( --env "DECKARD_DOCKER_IMAGE_TAG=deckard:cpu" )
  CMD+=( --env "DECKARD_DOCKER_BUILD_ARGS=--build-arg ENABLE_CUDA=0 --build-arg BASE_IMAGE=ubuntu:20.04" )
fi

if [[ -n "$JOB" ]]; then
  CMD+=( -j "$JOB" )
fi

if [[ "$VERBOSE" -eq 1 ]]; then
  CMD+=( --verbose )
fi

if [[ -n "$GITHUB_TOKEN_VALUE" ]]; then
  CMD+=( -s "GITHUB_TOKEN=$GITHUB_TOKEN_VALUE" )
fi

PRINT_CMD=()
for arg in "${CMD[@]}"; do
  if [[ "$arg" == GITHUB_TOKEN=* ]]; then
    PRINT_CMD+=("GITHUB_TOKEN=***")
  else
    PRINT_CMD+=("$arg")
  fi
done

echo "==> Running workflow"
echo "Workflow: $WORKFLOW_PATH"
echo "Event:    $EVENT_NAME"
echo "Ref:      $REF_NAME"
echo "GPU mode: $RESOLVED_GPU_MODE (requested: $GPU_MODE)"
if [[ "$RESOLVED_GPU_MODE" == "cpu" ]]; then
  echo "Docker build hint: docker build -t deckard:cpu ."
elif [[ "$RESOLVED_GPU_MODE" == "mps" ]]; then
  echo "Docker build hint: docker build -t deckard:mps ."
else
  echo "Docker build hint: docker build --build-arg ENABLE_CUDA=1 --build-arg BASE_IMAGE=nvidia/cuda:12.0.0-runtime-ubuntu20.04 -t deckard:cuda ."
fi
if [[ -n "$JOB" ]]; then
  echo "Job:      $JOB"
fi

echo "Command:"
printf ' %q' "${PRINT_CMD[@]}"
printf '\n'

if [[ -z "$GITHUB_TOKEN_VALUE" ]]; then
  cat <<'EOF'
Warning: no GitHub token provided.
Some workflows (notably actions/checkout on private repos) can fail under act with:
  "Input required and not supplied: token"
Provide one via --github-token or env var GITHUB_TOKEN/GH_TOKEN.
EOF
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

command -v docker >/dev/null 2>&1 || {
  cat >&2 <<'EOF'
Error: docker is required.
Install one of the following and retry:
  - Docker Desktop: brew install --cask docker
  - Colima: brew install colima && colima start
  - OrbStack
EOF
  exit 1
}

command -v act >/dev/null 2>&1 || {
  echo "Error: act is required. Install with: brew install act" >&2
  exit 1
}

if ! docker info >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Error: Docker daemon is not reachable.

act requires a running Docker engine. Start one of the following, then retry:
  - Docker Desktop
  - Colima: colima start
  - OrbStack

If DOCKER_HOST is customized, ensure it points to a live Docker socket.
EOF
  exit 1
fi

cd "$ROOT_DIR"
"${CMD[@]}"
