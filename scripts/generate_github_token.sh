#!/usr/bin/env bash
set -euo pipefail

# Helper script to obtain a GitHub token from GitHub CLI (gh)
# for local tooling such as scripts/test_workflow.sh.

MODE="export"
OUTPUT_FILE=""
COPY_TO_CLIPBOARD=0

usage() {
  cat <<'EOF'
Usage: scripts/generate_github_token.sh [options]

Options:
      --plain                 Print raw token only
      --export                Print shell export command (default)
      --write-env-file <path> Write GITHUB_TOKEN=<token> to file
      --copy                  Copy token to clipboard on macOS (pbcopy)
  -h, --help                  Show this help

Examples:
  # Export into current shell
  export GITHUB_TOKEN="$(./scripts/generate_github_token.sh --plain)"

  # Print an export command you can eval
  eval "$(./scripts/generate_github_token.sh --export)"

  # Write token for local tooling
  ./scripts/generate_github_token.sh --write-env-file .act.env
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plain)
      MODE="plain"
      shift
      ;;
    --export)
      MODE="export"
      shift
      ;;
    --write-env-file)
      OUTPUT_FILE="${2:-}"
      if [[ -z "$OUTPUT_FILE" ]]; then
        echo "Error: --write-env-file requires a path" >&2
        exit 2
      fi
      shift 2
      ;;
    --copy)
      COPY_TO_CLIPBOARD=1
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

command -v gh >/dev/null 2>&1 || {
  echo "Error: GitHub CLI (gh) is required. Install with: brew install gh" >&2
  exit 1
}

if ! gh auth status >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Error: gh is not authenticated.

Run:
  gh auth login

Then rerun this script.
EOF
  exit 1
fi

TOKEN="$(gh auth token)"

if [[ -z "$TOKEN" ]]; then
  echo "Error: could not retrieve a token from gh auth token" >&2
  exit 1
fi

if [[ -n "$OUTPUT_FILE" ]]; then
  tmp_file="${OUTPUT_FILE}.tmp"
  {
    echo "GITHUB_TOKEN=$TOKEN"
  } > "$tmp_file"
  mv "$tmp_file" "$OUTPUT_FILE"
  chmod 600 "$OUTPUT_FILE" 2>/dev/null || true
fi

if [[ "$COPY_TO_CLIPBOARD" -eq 1 ]]; then
  if command -v pbcopy >/dev/null 2>&1; then
    printf '%s' "$TOKEN" | pbcopy
  else
    echo "Warning: pbcopy not found; skipping clipboard copy" >&2
  fi
fi

if [[ "$MODE" == "plain" ]]; then
  printf '%s\n' "$TOKEN"
else
  # shellcheck disable=SC2016
  printf "export GITHUB_TOKEN=%q\n" "$TOKEN"
fi
