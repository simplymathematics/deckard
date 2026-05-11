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
RUN_FILE="$OUT_DIR/runtime.txt"
TIMING_FILE="$OUT_DIR/timing.txt"
COV_FILE="$OUT_DIR/coverage.txt"

: > "$ERROR_LOG"

rm -f "$TEST_OUT" "$TEST_ERR" "$RUN_FILE" "$TIMING_FILE" "$COV_FILE"

run_tests() {
  local started_epoch
  local ended_epoch
  local elapsed_seconds
  local status

  started_epoch="$(date +%s)"
  echo "[INFO] Running pytest..."

  set +e
  "$PYTHON" -m pytest \
    -n auto \
    "$TEST_DIR" \
    -ra \
    --cov="$PKG" \
    --cov-append \
    --durations=0 \
    > "$TEST_OUT" \
    2> "$TEST_ERR"
  status=$?
  set -e  

  ended_epoch="$(date +%s)"
  elapsed_seconds="$((ended_epoch - started_epoch))"

  {
    echo "test_dir=$TEST_DIR"
    echo "started_epoch=$started_epoch"
    echo "ended_epoch=$ended_epoch"
    echo "elapsed_seconds=$elapsed_seconds"
  } > "$RUN_FILE"

  return "$status"
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
# Timing summary (separate file)
# -------------------------
{
  echo "=== PYTEST TIMING SUMMARY ==="
  grep -E "^=+ slowest|^[[:space:]]*[0-9]+\.[0-9]+s|^[[:space:]]*[0-9]+\.[0-9]+ seconds" "$TEST_OUT" || true
  grep -E "^=+ slowest|^[[:space:]]*[0-9]+\.[0-9]+s|^[[:space:]]*[0-9]+\.[0-9]+ seconds" "$TEST_ERR" || true
} > "$TIMING_FILE"

echo "[INFO] Runtime written to $RUN_FILE"
echo "[INFO] Timing written to $TIMING_FILE"

# -------------------------
# Cleanup temp files
# -------------------------
# rm -f "$TEST_OUT" "$TEST_ERR"

echo "[INFO] Done"