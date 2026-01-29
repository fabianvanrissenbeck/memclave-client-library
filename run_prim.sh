#!/usr/bin/env bash
set -euo pipefail

# Simple PRiM/Memclave runner for build/ (ime-*-example)
# - run all or selected tests
# - log per test
# - classify PASS/FAIL based on "OK" vs "ERROR" in output

BUILD_DIR="$(cd "$(dirname "$0")" && pwd)/build"
LOGDIR="$BUILD_DIR/logs/$(date +%Y%m%d_%H%M%S)"

BFS_DATA_URL=https://zenodo.org/records/18307126/files/bfs-data.tar.zst

if [ ! -f ./data/LiveJournal1 ];
then
	if [ ! -f ./bfs-data.tar.zst ];
	then
		echo "BFS Example Graphs are Missing. Downloading from Zenodo."
		curl -o bfs-data.tar.zst $BFS_DATA_URL
	fi

	tar xf ./bfs-data.tar.zst --directory ./data/
	exit 1;
fi

# Canonical test name -> binary
declare -A BIN=(
  ["BFS"]="ime-bfs-example"
  ["BS"]="ime-bs-example"
  ["GEMV"]="ime-gemv-example"
  ["HSTL"]="ime-hstl-example"
  ["HSTS"]="ime-hsts-example"
  ["MLP"]="ime-mlp-example"
  ["RED"]="ime-red-example"
  ["SCANSSA"]="ime-scanssa-example"
  ["SCANRSS"]="ime-scanrss-example"
  ["SEL"]="ime-sel-example"
  ["TRNS"]="ime-trns-example"
  ["TS"]="ime-ts-example"
  ["UNI"]="ime-uni-example"
  ["VA"]="ime-va-example"
  ["SPMV"]="ime-spmv-example"
  ["NW"]="ime-nw-example"
)

# A few friendly aliases
alias_name() {
  local t="$1"
  t="$(echo "$t" | tr '[:lower:]' '[:upper:]' | tr -d ' ')"
  case "$t" in
    "SCAN_SSA"|"SCAN-SSA") echo "SCANSSA" ;;
    "SCAN_RSS"|"SCAN-RSS") echo "SCANRSS" ;;
    "HST-S"|"HSTS") echo "HSTS" ;;
    "HST-L"|"HSTL") echo "HSTL" ;;
    *) echo "$t" ;;
  esac
}

list_tests() {
  echo "Known tests:"
  for k in "${!BIN[@]}"; do
    printf "  %-8s -> %s\n" "$k" "${BIN[$k]}"
  done | sort
  echo
  echo "Discovered binaries in $BUILD_DIR:"
  (cd "$BUILD_DIR" && ls -1 ime-*-example 2>/dev/null || true) | sed 's/^/  /'
}

usage() {
  cat <<EOF
Usage:
  $0 --list
  $0 all
  $0 TRNS BS MLP
  $0 ime-trns-example ime-bs-example   # direct binary names also work

Pass/Fail rule:
  FAIL if output contains "ERROR"
  PASS if output contains "OK" (and no ERROR)
  Otherwise -> FAIL (unknown)

Logs:
  $LOGDIR/<TEST>.log
EOF
}

mkdir -p "$LOGDIR"

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ "${1:-}" == "--list" ]]; then
  list_tests
  exit 0
fi

# Build run list
RUN_TESTS=()

if [[ "${1:-}" == "all" || "${1:-}" == "ALL" ]]; then
  # fixed order (edit if you want)
  RUN_TESTS=(BS TS TRNS RED HSTS HSTL SEL UNI SCANSSA SCANRSS GEMV MLP VA BFS SPMV NW)
else
  for arg in "$@"; do
    # allow direct binary
    if [[ "$arg" == ime-*-example ]]; then
      RUN_TESTS+=("$arg")
      continue
    fi
    RUN_TESTS+=("$(alias_name "$arg")")
  done
fi

PASSED=()
FAILED=()

echo "Logs: $LOGDIR"
echo

run_one() {
  local test="$1"
  local bin="$2"
  local log="$3"

  if [[ ! -x "$BUILD_DIR/$bin" ]]; then
    echo "[FAIL] $test : missing $bin"
    FAILED+=("$test")
    return 0
  fi

  echo "==> Running $test ($bin)"
  (cd "$BUILD_DIR" && "./$bin") 2>&1 | tee "$log" >/dev/null

  if grep -q "ERROR" "$log"; then
    echo "[FAIL] $test (found ERROR)"
    FAILED+=("$test")
  elif grep -q "OK" "$log"; then
    echo "[PASS] $test (found OK)"
    PASSED+=("$test")
  else
    echo "[FAIL] $test (no OK/ERROR marker)"
    FAILED+=("$test")
  fi
  echo
}

for t in "${RUN_TESTS[@]}"; do
  if [[ "$t" == ime-*-example ]]; then
    test_name="$t"
    run_one "$test_name" "$t" "$LOGDIR/${test_name}.log"
    continue
  fi

  if [[ -z "${BIN[$t]+x}" ]]; then
    echo "[FAIL] $t : unknown test name (use --list)"
    FAILED+=("$t")
    echo
    continue
  fi

  run_one "$t" "${BIN[$t]}" "$LOGDIR/${t}.log"
done

echo "==================== Summary ===================="
echo "PASSED (${#PASSED[@]}):"
for t in "${PASSED[@]}"; do echo "  - $t"; done
echo
echo "FAILED (${#FAILED[@]}):"
for t in "${FAILED[@]}"; do echo "  - $t"; done
echo "================================================="

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  exit 1
fi
exit 0
