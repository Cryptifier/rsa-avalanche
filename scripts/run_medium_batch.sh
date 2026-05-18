#!/usr/bin/env bash
set -euo pipefail

RUNS=${RUNS:-100}
SEED_START=${SEED_START:-1}
CONFIG=${CONFIG:-"config/rsa_config_medium.json"}
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
AVALANCHE_SOLVER_SUCCESS_MARKER="AVALANCHE SOLVER FOUND MESSAGE"

solver_enabled_for_config() {
  local config_path=$1
  if [[ ! -f "${config_path}" ]]; then
    echo 0
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 - "${config_path}" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
text = re.sub(r"//.*", "", text)
print("1" if re.search(r'"avalanche_solver_enable"\s*:\s*true\b', text) else "0")
PY
  else
    if grep -Eq '"avalanche_solver_enable"[[:space:]]*:[[:space:]]*true' "${config_path}"; then
      echo 1
    else
      echo 0
    fi
  fi
}

SOLVER_ENABLED=$(solver_enabled_for_config "${CONFIG}")

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
cx_match_sum=0
cx_match_sumsq=0
cx_match_min=""
cx_match_max=""
cx_match_count=0
cx_candidates_sum=0
cx_candidates_count=0
avalanche_candidates_sum=0
avalanche_candidates_count=0

echo "Running ${RUNS} iterations with config ${CONFIG}"
if [[ "${SOLVER_ENABLED}" == "1" ]]; then
  echo "Avalanche solver marker check: enabled"
else
  echo "Avalanche solver marker check: disabled"
fi
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
  cargo run --bin analysis -- --bits 256 --bits-decrypt 256 --seed "${seed}" -c "${CONFIG}" --crypto-rng \
    --session-json "${session_path}" --batches "${ANALYSIS_BATCHES}" --batch-size "${ANALYSIS_BATCH_SIZE}" \
    "${TEST_ARGS[@]}" "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "${ANALYSIS_LOG}" | tee "${run_output}" > /dev/null
  status=$?
  set -e

  end_ns=$(date +%s%N)
  duration_ns=$((end_ns - start_ns))
  total_ns=$((total_ns + duration_ns))
  duration_s=$(awk -v ns="${duration_ns}" 'BEGIN { printf "%.3f", ns / 1000000000 }')

  match_line=$(grep -m1 "Bitwise speculative oracle match" "${run_output}" || true)
  match_pct=$(echo "${match_line}" | sed -n 's/.*(\([0-9.]*\)%).*/\1/p')
  cx_match_line=$(grep -F -m1 "Avalanche c^x run max:" "${run_output}" || true)
  cx_match_pct=$(echo "${cx_match_line}" | sed -n 's/.*match \([0-9.]*\)%.*/\1/p')
  beam_run_max_line=$(grep -F -m1 "Avalanche beam run max:" "${run_output}" || true)
  beam_run_max_match_pct=$(echo "${beam_run_max_line}" | sed -n 's/.*match \([0-9.]*\)%.*/\1/p')
  majority_vote_line=$(grep -F -m1 "Avalanche majority vote run max:" "${run_output}" || true)
  majority_vote_match_pct=$(echo "${majority_vote_line}" | sed -n 's/.*match \([0-9.]*\)%.*/\1/p')
  cx_total_line=$(grep -F -m1 "Avalanche c^x evaluated total:" "${run_output}" || true)
  cx_candidates_total=$(echo "${cx_total_line}" | sed -n 's/.*: \([0-9][0-9]*\)$/\1/p')
  avalanche_total_line=$(grep -F -m1 "Avalanche evaluated candidates total:" "${run_output}" || true)
  avalanche_candidates_total=$(echo "${avalanche_total_line}" | sed -n 's/.*: \([0-9][0-9]*\)$/\1/p')
  verdict=$(grep -m1 "Sufficiency verdict" "${run_output}" | sed -n 's/.*: //p' || true)
  solver_status="DISABLED"
  solver_color="${YELLOW}"
  if [[ "${SOLVER_ENABLED}" == "1" ]]; then
    if grep -F -q "${AVALANCHE_SOLVER_SUCCESS_MARKER}" "${run_output}"; then
      solver_status="SUCCESS"
      solver_color="${GREEN}"
      verdict="PASS"
    else
      solver_status="FAIL"
      solver_color="${RED}"
      verdict="FAIL"
      echo "${RED}Avalanche solver failure: ${AVALANCHE_SOLVER_SUCCESS_MARKER} not found in run output.${RESET}"
    fi
  fi

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

  if [[ -n "${match_pct}" ]]; then
    if awk -v v="${match_pct}" 'BEGIN { exit (v >= 50.0) ? 0 : 1 }'; then
      match_color="${GREEN}"    
    else
      match_color="${RED}"
    fi
  else
    match_color="${YELLOW}"
  fi

  if [[ -n "${cx_match_pct}" ]]; then
    if awk -v v="${cx_match_pct}" 'BEGIN { exit (v >= 50.0) ? 0 : 1 }'; then
      cx_match_color="${GREEN}"
    else
      cx_match_color="${RED}"
    fi
  else
    cx_match_color="${YELLOW}"
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
    echo "Run ${i} summary: match ${match_color}${match_pct:-N/A}%${RESET}, c^x max match ${cx_match_color}${cx_match_pct:-N/A}%${RESET}, beam run max ${beam_match_color}${beam_run_max_match_pct:-N/A}%${RESET}, majority vote match ${majority_match_color}${majority_vote_match_pct:-N/A}%${RESET}, solver ${solver_color}${solver_status}${RESET}, c^x candidates ${cx_candidates_total:-N/A}, avalanche candidates ${avalanche_candidates_total:-N/A}, verdict ${verdict_color}${verdict:-UNKNOWN}${RESET}, duration ${duration_s}s"
  else
    echo "Run ${i} summary: ${RED}FAILED (exit ${status})${RESET}, match ${match_color}${match_pct:-N/A}%${RESET}, c^x max match ${cx_match_color}${cx_match_pct:-N/A}%${RESET}, beam run max ${beam_match_color}${beam_run_max_match_pct:-N/A}%${RESET}, majority vote match ${majority_match_color}${majority_vote_match_pct:-N/A}%${RESET}, solver ${solver_color}${solver_status}${RESET}, c^x candidates ${cx_candidates_total:-N/A}, avalanche candidates ${avalanche_candidates_total:-N/A}, verdict ${verdict_color}${verdict:-UNKNOWN}${RESET}, duration ${duration_s}s"
  fi
  echo "Session JSON: ${session_path}"
  if [[ -n "${beam_run_max_line}" ]]; then
    echo "${beam_run_max_line}"
  fi
  if [[ -n "${majority_vote_line}" ]]; then
    echo "${majority_vote_line}"
  fi
  beam_comparison_block=$(awk '
    /Avalanche beam colored hex/ {print; capture=1; next}
    capture {print; if (/^Hex match key:/) exit}
  ' "${run_output}")
  majority_comparison_block=$(awk '
    /Avalanche majority vote colored hex/ {print; capture=1; next}
    capture {print; if (/^Hex match key:/) exit}
  ' "${run_output}")
  if [[ -n "${beam_comparison_block}" || -n "${majority_comparison_block}" ]]; then
    if [[ -n "${beam_comparison_block}" ]]; then
      echo "${beam_comparison_block}"
    else
      echo "Avalanche beam colored comparison: N/A"
    fi
    echo "-----"
    if [[ -n "${majority_comparison_block}" ]]; then
      echo "${majority_comparison_block}"
    else
      echo "Avalanche majority vote colored comparison: N/A"
    fi
  fi
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
echo "Match % stats: mean ${mean}, std dev ${stddev}, min ${min}, max ${max}, n ${count}"
echo "c^x max match % stats: mean ${cx_match_mean}, std dev ${cx_match_stddev}, min ${cx_match_min}, max ${cx_match_max}, n ${cx_match_count}"
echo "c^x evaluated candidates: total ${cx_candidates_sum}, average ${cx_candidates_avg}, n ${cx_candidates_count}"
echo "Avalanche evaluated candidates: total ${avalanche_candidates_sum}, average ${avalanche_candidates_avg}, n ${avalanche_candidates_count}"
echo "Avalanche solver marker check: $( [[ "${SOLVER_ENABLED}" == "1" ]] && echo enabled || echo disabled )"
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
