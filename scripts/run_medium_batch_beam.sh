#!/usr/bin/env bash
set -euo pipefail

RUNS=${RUNS:-1}
SEED_START=${SEED_START:-1}
CONFIG=${CONFIG:-"config/rsa_config_medium.json"}
ANALYSIS_LOG=${ANALYSIS_LOG:-"logs_current.log"}
SCRIPT_LOG=${SCRIPT_LOG:-"logs_current_script.log"}
RESUME=${RESUME:-0}
ANALYSIS_EXTRA_ARGS=${ANALYSIS_EXTRA_ARGS:-}
AVALANCHE_BATCHES=${AVALANCHE_BATCHES:-${ANALYSIS_BATCHES:-}}
AVALANCHE_BATCH_SIZE=${AVALANCHE_BATCH_SIZE:-${ANALYSIS_BATCH_SIZE:-}}
LOG_DIR=${LOG_DIR:-"logs"}
RUN_TESTS=${RUN_TESTS:-0}
RUN_PCA=${RUN_PCA:-0}
PCA_OUTPUT=${PCA_OUTPUT:-"pca_clusters.png"}

read -r -a EXTRA_ARGS <<< "${ANALYSIS_EXTRA_ARGS}"

arch_name=$(uname -m)
CARGO_RUN_ARGS=(--release)
ACCEL_FEATURE="none"

case "${arch_name}" in
  x86_64 | amd64)
    ACCEL_FEATURE="x86-hamming-accel"
    CARGO_RUN_ARGS+=(--features "${ACCEL_FEATURE}")
    ;;
  aarch64 | arm64)
    ACCEL_FEATURE="aarch64-hamming-accel"
    CARGO_RUN_ARGS+=(--features "${ACCEL_FEATURE}")
    ;;
esac

TEST_ARGS=()
if [[ "${RUN_TESTS}" == "1" ]]; then
  TEST_ARGS=(--tests)
fi

BATCH_ARGS=()
if [[ -n "${AVALANCHE_BATCHES}" ]]; then
  BATCH_ARGS+=(--batches "${AVALANCHE_BATCHES}")
fi
if [[ -n "${AVALANCHE_BATCH_SIZE}" ]]; then
  BATCH_ARGS+=(--batch-size "${AVALANCHE_BATCH_SIZE}")
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

cx_match_sum=0
cx_match_sumsq=0
cx_match_min=""
cx_match_max=""
cx_match_count=0
cx_candidates_sum=0
cx_candidates_count=0
avalanche_candidates_sum=0
avalanche_candidates_count=0
pass_count=0
fail_count=0
total_ns=0

echo "Running ${RUNS} iterations with config ${CONFIG}"
echo "Detected architecture ${arch_name}; cargo run args: ${CARGO_RUN_ARGS[*]} (acceleration feature: ${ACCEL_FEATURE})"
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
  cargo run "${CARGO_RUN_ARGS[@]}" --bin analysis -- --true --bits 256 --bits-decrypt 256 --seed "${seed}" -c "${CONFIG}" --crypto-rng --session-json "${session_path}" \
    "${BATCH_ARGS[@]}" "${TEST_ARGS[@]}" "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "${ANALYSIS_LOG}" | tee "${run_output}" > /dev/null
  status=$?
  set -e

  end_ns=$(date +%s%N)
  duration_ns=$((end_ns - start_ns))
  total_ns=$((total_ns + duration_ns))
  duration_s=$(awk -v ns="${duration_ns}" 'BEGIN { printf "%.3f", ns / 1000000000 }')

  cx_match_line=$(grep -F -m1 "Avalanche c^x run max:" "${run_output}" || true)
  cx_match_pct=$(echo "${cx_match_line}" | sed -n 's/.*match \([0-9.]*\)%.*/\1/p')
  beam_run_max_line=$(grep -F -m1 "Avalanche beam run max:" "${run_output}" || true)
  beam_run_max_match_pct=$(echo "${beam_run_max_line}" | sed -n 's/.*match \([0-9.]*\)%.*/\1/p')
  majority_vote_line=$(grep -F -m1 "Avalanche majority vote run max:" "${run_output}" || true)
  majority_vote_match_pct=$(echo "${majority_vote_line}" | sed -n 's/.*match \([0-9.]*\)%.*/\1/p')
  cx_total_line=$(grep -F -m1 "Avalanche c^x evaluated total:" "${run_output}" || true)
  cx_candidates_total=$(echo "${cx_total_line}" | sed -n 's/.*: \([0-9][0-9]*\)$/\1/p')
  avalanche_total_line=$(grep -m1 "Avalanche evaluated candidates total:" "${run_output}" || true)
  avalanche_candidates_total=$(echo "${avalanche_total_line}" | sed -n 's/.*: \([0-9][0-9]*\)$/\1/p')
  verdict=$(grep -m1 "Sufficiency verdict" "${run_output}" | sed -n 's/.*: //p' || true)

  if [[ -n "${cx_match_pct}" ]]; then
    cx_match_count=$((cx_match_count + 1))
    cx_match_sum=$(awk -v s="${cx_match_sum}" -v v="${cx_match_pct}" 'BEGIN { printf "%.6f", s + v }')
    cx_match_sumsq=$(awk -v s="${cx_match_sumsq}" -v v="${cx_match_pct}" 'BEGIN { printf "%.6f", s + v * v }')
    if [[ -z "${cx_match_min}" || $(awk -v a="${cx_match_pct}" -v b="${cx_match_min}" 'BEGIN { print (a < b) ? 1 : 0 }') -eq 1 ]]; then
      cx_match_min="${cx_match_pct}"
    fi
    if [[ -z "${cx_match_max}" || $(awk -v a="${cx_match_pct}" -v b="${cx_match_max}" 'BEGIN { print (a > b) ? 1 : 0 }') -eq 1 ]]; then
      cx_match_max="${cx_match_pct}"
    fi
  fi

  if [[ -n "${cx_candidates_total}" ]]; then
    cx_candidates_count=$((cx_candidates_count + 1))
    cx_candidates_sum=$((cx_candidates_sum + cx_candidates_total))
  fi

  if [[ -n "${avalanche_candidates_total}" ]]; then
    avalanche_candidates_count=$((avalanche_candidates_count + 1))
    avalanche_candidates_sum=$((avalanche_candidates_sum + avalanche_candidates_total))
  fi

  if [[ "${verdict}" == *PASS* ]]; then
    pass_count=$((pass_count + 1))
  else
    fail_count=$((fail_count + 1))
  fi

  if [[ -n "${cx_match_pct}" ]]; then
    if awk -v v="${cx_match_pct}" 'BEGIN { exit (v >= 50.0) ? 0 : 1 }'; then
      match_color="${GREEN}"
    else
      match_color="${RED}"
    fi
  else
    match_color="${YELLOW}"
  fi

  if [[ -n "${beam_run_max_match_pct}" ]]; then
    if awk -v v="${beam_run_max_match_pct}" 'BEGIN { exit (v >= 50.0) ? 0 : 1 }'; then
      beam_match_color="${GREEN}"
    else
      beam_match_color="${RED}"
    fi
  else
    beam_match_color="${YELLOW}"
  fi

  if [[ -n "${majority_vote_match_pct}" ]]; then
    if awk -v v="${majority_vote_match_pct}" 'BEGIN { exit (v >= 50.0) ? 0 : 1 }'; then
      majority_match_color="${GREEN}"
    else
      majority_match_color="${RED}"
    fi
  else
    majority_match_color="${YELLOW}"
  fi

  verdict_color="${GREEN}"
  if [[ "${verdict}" != *PASS* ]]; then
    verdict_color="${RED}"
  fi

  if [[ ${status} -eq 0 ]]; then
    echo "Run ${i} summary: c^x max match ${match_color}${cx_match_pct:-N/A}%${RESET}, beam run max ${beam_match_color}${beam_run_max_match_pct:-N/A}%${RESET}, majority vote match ${majority_match_color}${majority_vote_match_pct:-N/A}%${RESET}, c^x candidates ${cx_candidates_total:-N/A}, avalanche candidates ${avalanche_candidates_total:-N/A}, verdict ${verdict_color}${verdict:-UNKNOWN}${RESET}, duration ${duration_s}s"
  else
    echo "Run ${i} summary: ${RED}FAILED (exit ${status})${RESET}, c^x max match ${match_color}${cx_match_pct:-N/A}%${RESET}, beam run max ${beam_match_color}${beam_run_max_match_pct:-N/A}%${RESET}, majority vote match ${majority_match_color}${majority_vote_match_pct:-N/A}%${RESET}, c^x candidates ${cx_candidates_total:-N/A}, avalanche candidates ${avalanche_candidates_total:-N/A}, verdict ${verdict_color}${verdict:-UNKNOWN}${RESET}, duration ${duration_s}s"
  fi
  echo "Session JSON: ${session_path}"
  if [[ -n "${beam_run_max_line}" ]]; then
    echo "${beam_run_max_line}"
  fi
  if [[ -n "${majority_vote_line}" ]]; then
    echo "${majority_vote_line}"
  fi
  progress_bar "${i}" "${RUNS}"
  rm -f "${run_output}"
done

printf "\n"

if [[ "${cx_match_count}" -gt 0 ]]; then
  cx_match_mean=$(awk -v s="${cx_match_sum}" -v n="${cx_match_count}" 'BEGIN { printf "%.4f", s / n }')
  cx_match_variance=$(awk -v s="${cx_match_sum}" -v ss="${cx_match_sumsq}" -v n="${cx_match_count}" 'BEGIN { printf "%.6f", (ss / n) - (s / n) * (s / n) }')
  cx_match_stddev=$(awk -v v="${cx_match_variance}" 'BEGIN { if (v < 0) v = 0; printf "%.4f", sqrt(v) }')
else
  cx_match_mean="N/A"
  cx_match_stddev="N/A"
  cx_match_min="N/A"
  cx_match_max="N/A"
fi

if [[ "${cx_candidates_count}" -gt 0 ]]; then
  cx_candidates_avg=$(awk -v s="${cx_candidates_sum}" -v n="${cx_candidates_count}" 'BEGIN { printf "%.4f", s / n }')
else
  cx_candidates_avg="N/A"
fi

if [[ "${avalanche_candidates_count}" -gt 0 ]]; then
  avalanche_candidates_avg=$(awk -v s="${avalanche_candidates_sum}" -v n="${avalanche_candidates_count}" 'BEGIN { printf "%.4f", s / n }')
else
  avalanche_candidates_avg="N/A"
fi

avg_time_s=$(awk -v ns="${total_ns}" -v n="${RUNS}" 'BEGIN { printf "%.3f", ns / (n * 1000000000) }')

echo ""
echo "===== SUMMARY ====="
echo "c^x max match % stats: mean ${cx_match_mean}, std dev ${cx_match_stddev}, min ${cx_match_min}, max ${cx_match_max}, n ${cx_match_count}"
echo "c^x evaluated candidates: total ${cx_candidates_sum}, average ${cx_candidates_avg}, n ${cx_candidates_count}"
echo "Avalanche evaluated candidates: total ${avalanche_candidates_sum}, average ${avalanche_candidates_avg}, n ${avalanche_candidates_count}"
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
