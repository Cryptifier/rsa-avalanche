#!/usr/bin/env bash
set -euo pipefail

RUNS=${RUNS:-1}
SEED_START=${SEED_START:-1}
CONFIG=${CONFIG:-"config/rsa_config_small_batch.json"}
ANALYSIS_LOG=${ANALYSIS_LOG:-"logs_current.log"}
SCRIPT_LOG=${SCRIPT_LOG:-"logs_current_script.log"}
RESUME=${RESUME:-0}
ANALYSIS_EXTRA_ARGS=${ANALYSIS_EXTRA_ARGS:-}
ANALYSIS_BATCHES=${ANALYSIS_BATCHES:-10}
ANALYSIS_BATCH_SIZE=${ANALYSIS_BATCH_SIZE:-50}
LOG_DIR=${LOG_DIR:-"logs"}
RUN_TESTS=${RUN_TESTS:-0}
RUN_PCA=${RUN_PCA:-0}
PCA_OUTPUT=${PCA_OUTPUT:-"pca_clusters.png"}

read -r -a EXTRA_ARGS <<< "${ANALYSIS_EXTRA_ARGS}"

TEST_ARGS=()
if [[ "${RUN_TESTS}" == "1" ]]; then
  TEST_ARGS=(--tests)
fi

RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
BLUE=$'\033[0;34m'
RESET=$'\033[0m'

if [[ "${RESUME}" != "1" ]]; then
  : > "${ANALYSIS_LOG}"
  : > "${SCRIPT_LOG}"
fi

exec > >(tee -a "${SCRIPT_LOG}") 2>&1

progress_bar() {
  local current=$1
  local total=$2
  local width=30
  local percent=$((current * 100 / total))
  local filled=$((percent * width / 100))
  local empty=$((width - filled))
  local bar
  local pad
  bar=$(printf "%${filled}s" "" | tr ' ' '#')
  pad=$(printf "%${empty}s" "")
  printf "\r${BLUE}[%-${width}s]${RESET} %3d%% (%d/%d)" "${bar}${pad}" "${percent}" "${current}" "${total}"
}

sum=0
sumsq=0
min=""
max=""
count=0
pass_count=0
fail_count=0
total_ns=0

echo "Running ${RUNS} iterations with config ${CONFIG}"
mkdir -p "${LOG_DIR}"
run_stamp=$(date +"%Y%m%d_%H%M%S")

for i in $(seq 1 "${RUNS}"); do
  seed=$((SEED_START + i - 1))
  run_output="$(mktemp)"
  start_ns=$(date +%s%N)
  session_path="${LOG_DIR}/session_${run_stamp}_seed_${seed}.json"

  echo ""
  echo "===== RUN ${i} (seed ${seed}) ====="
  set +e
  cargo run --bin analysis -- --same-r-batch --true --bits 56 --bits-decrypt 128 --seed "${seed}" -c "${CONFIG}" --tests --crypto-rng --session-json "${session_path}" \
    --mirror-invert-candidates --batches "${ANALYSIS_BATCHES}" --batch-size "${ANALYSIS_BATCH_SIZE}" "${TEST_ARGS[@]}" "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "${ANALYSIS_LOG}" | tee "${run_output}" > /dev/null
  status=$?
  set -e

  end_ns=$(date +%s%N)
  duration_ns=$((end_ns - start_ns))
  total_ns=$((total_ns + duration_ns))
  duration_s=$(awk -v ns="${duration_ns}" 'BEGIN { printf "%.3f", ns / 1000000000 }')

  match_line=$(grep -m1 "Bitwise speculative oracle match" "${run_output}" || true)
  match_pct=$(echo "${match_line}" | sed -n 's/.*(\([0-9.]*\)%).*/\1/p') 
  verdict=$(grep -m1 "Sufficiency verdict" "${run_output}" | sed -n 's/.*: //p' || true)

  if [[ -n "${match_pct}" ]]; then
    count=$((count + 1))
    sum=$(awk -v s="${sum}" -v v="${match_pct}" 'BEGIN { printf "%.6f", s + v }')
    sumsq=$(awk -v s="${sumsq}" -v v="${match_pct}" 'BEGIN { printf "%.6f", s + v * v }')
    if [[ -z "${min}" || $(awk -v a="${match_pct}" -v b="${min}" 'BEGIN { print (a < b) ? 1 : 0 }') -eq 1 ]]; then
      min="${match_pct}"
    fi
    if [[ -z "${max}" || $(awk -v a="${match_pct}" -v b="${max}" 'BEGIN { print (a > b) ? 1 : 0 }') -eq 1 ]]; then
      max="${match_pct}"
    fi
  fi

  if [[ "${verdict}" == *PASS* ]]; then
    pass_count=$((pass_count + 1))
  else
    fail_count=$((fail_count + 1))
  fi

  if [[ -n "${match_pct}" ]]; then
    if awk -v v="${match_pct}" 'BEGIN { exit (v >= 50.0) ? 0 : 1 }'; then
      match_color="${GREEN}"    
    else
      match_color="${RED}"
    fi
  else
    match_color="${YELLOW}"
  fi

  verdict_color="${GREEN}"
  if [[ "${verdict}" != *PASS* ]]; then
    verdict_color="${RED}"
  fi

  if [[ ${status} -eq 0 ]]; then
    echo "Run ${i} summary: match ${match_color}${match_pct:-N/A}%${RESET}, verdict ${verdict_color}${verdict:-UNKNOWN}${RESET}, duration ${duration_s}s"
  else
    echo "Run ${i} summary: ${RED}FAILED (exit ${status})${RESET}, match ${match_color}${match_pct:-N/A}%${RESET}, verdict ${verdict_color}${verdict:-UNKNOWN}${RESET}, duration ${duration_s}s"
  fi
  echo "Session JSON: ${session_path}"
  progress_bar "${i}" "${RUNS}"
  rm -f "${run_output}"
done

printf "\n"

if [[ "${count}" -gt 0 ]]; then
  mean=$(awk -v s="${sum}" -v n="${count}" 'BEGIN { printf "%.4f", s / n }')
  variance=$(awk -v s="${sum}" -v ss="${sumsq}" -v n="${count}" 'BEGIN { printf "%.6f", (ss / n) - (s / n) * (s / n) }')
  stddev=$(awk -v v="${variance}" 'BEGIN { if (v < 0) v = 0; printf "%.4f", sqrt(v) }')
else
  mean="N/A"
  stddev="N/A"
  min="N/A"
  max="N/A"
fi

avg_time_s=$(awk -v ns="${total_ns}" -v n="${RUNS}" 'BEGIN { printf "%.3f", ns / (n * 1000000000) }')

echo ""
echo "===== SUMMARY ====="
echo "Match % stats: mean ${mean}, std dev ${stddev}, min ${min}, max ${max}, n ${count}"
echo "Verdicts: PASS ${pass_count}, FAIL ${fail_count}"
echo "Average duration per run: ${avg_time_s}s"
if [[ -n "${session_path:-}" ]]; then
  echo "Viewer: python3 scripts/session_viewer.py ${session_path} (Beam vs R tab)"
fi

if [[ "${RUN_PCA}" == "1" && -n "${session_path:-}" ]]; then
  echo ""
  echo "Running PCA clustering via PyTorch script..."
  python3 scripts/r_candidate_cnn.py --session "${session_path}" --config "${CONFIG}" --output "${PCA_OUTPUT}"
  echo "PCA output written to ${PCA_OUTPUT}"
fi
