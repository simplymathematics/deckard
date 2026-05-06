#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="build"
PKG="deckard"
TEST_DIR="${1:-test}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

mkdir -p "$OUT_DIR"

ERROR_LOG="$OUT_DIR/error.log"
TEST_OUT="$OUT_DIR/pytest.out"
TEST_ERR="$OUT_DIR/pytest.err"
COV_FILE="$OUT_DIR/coverage.txt"

: > "$ERROR_LOG"

rm -f "$TEST_OUT" "$TEST_ERR"

run_tests() {
  echo "[INFO] Running pytest..."

  "$PYTHON" -m pytest \
    -n auto \
    "$TEST_DIR" \
    -ra \
    --cov="$PKG" \
    --cov-append \
    --durations=0 \
    > "$TEST_OUT" \
    2> "$TEST_ERR"

  return $?
}

capture_failure() {
  {
    echo "=== PYTEST FAILED ==="
    echo
    echo "=== STDOUT ==="
    cat "$TEST_OUT"
    echo
    echo "=== STDERR ==="
    cat "$TEST_ERR"
  } >> "$ERROR_LOG"
}

# -------------------------
# Run tests once
# -------------------------
if ! run_tests; then
  capture_failure
  echo "[ERROR] Tests failed. See $ERROR_LOG"
  exit 1
fi

# -------------------------
# Coverage report (post-process only)
# -------------------------
echo "[INFO] Generating coverage report..."
"$PYTHON" -m coverage report -m > "$COV_FILE"

echo "[INFO] Coverage written to $COV_FILE"

# -------------------------
# Optional derived summaries
# -------------------------
echo "[INFO] Test timing summary (from pytest output):"
grep -E "slowest|seconds|slow call" "$TEST_OUT" || true

# -------------------------
# Cleanup temp files
# -------------------------
# rm -f "$TEST_OUT" "$TEST_ERR"

echo "[INFO] Done"