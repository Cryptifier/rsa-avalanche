#!/usr/bin/env bash
set -euo pipefail

RUNS_PER_CONFIG=5
SEED_START=${SEED_START:-1}
BASE_CONFIG=${BASE_CONFIG:-"config/rsa_config_small_batch.json"}
CONFIG="./grid_config.json"
ANALYSIS_LOG=${ANALYSIS_LOG:-"logs_current.log"}
SCRIPT_LOG=${SCRIPT_LOG:-"logs_current_script.log"}
RESUME=${RESUME:-0}
ANALYSIS_EXTRA_ARGS=${ANALYSIS_EXTRA_ARGS:-}
LOG_DIR=${LOG_DIR:-"logs"}
RUN_TESTS=${RUN_TESTS:-0}
RUN_PCA=${RUN_PCA:-0}
PCA_OUTPUT=${PCA_OUTPUT:-"pca_clusters.png"}

analysis_batch_messages_values=(50000 100000)
analysis_batch_candidates_values=(1000 2000)
analysis_batch_batches_values=(1 2)
avalanche_combination_samples_values=(6000 12000)
avalanche_combination_size_values=(4 5)
avalanche_combination_pool_size_values=(1000)
avalanche_combination_majority_vote_values=(true)
avalanche_combination_sample_smoothing_values=(false true)
avalanche_combination_majority_vote_print_values=(true)

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

cleanup() {
  rm -f "${CONFIG}" "${CONFIG}.tmp"
}

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing base config ${BASE_CONFIG}" >&2
  exit 1
fi

if [[ "${RESUME}" != "1" ]]; then
  : > "${ANALYSIS_LOG}"
  : > "${SCRIPT_LOG}"
fi

trap cleanup EXIT
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

copy_base_config() {
  cp "${BASE_CONFIG}" "${CONFIG}"
  sed -i 's,//.*$,,' "${CONFIG}"
  jq '.' "${CONFIG}" > "${CONFIG}.tmp"
  mv "${CONFIG}.tmp" "${CONFIG}"
}

set_config_field() {
  local field_path=$1
  local value=$2

  jq --argjson value "${value}" "${field_path} = \$value" "${CONFIG}" > "${CONFIG}.tmp"
  mv "${CONFIG}.tmp" "${CONFIG}"
}

print_grid_parameters() {
  local analysis_batch_messages=$1
  local analysis_batch_candidates=$2
  local analysis_batch_batches=$3
  local avalanche_combination_samples=$4
  local avalanche_combination_size=$5
  local avalanche_combination_pool_size=$6
  local avalanche_combination_majority_vote=$7
  local avalanche_combination_sample_smoothing=$8
  local avalanche_combination_majority_vote_print=$9

  echo "Grid parameters:"
  echo "  analysis_batch_messages=${analysis_batch_messages}"
  echo "  analysis_batch_candidates=${analysis_batch_candidates}"
  echo "  analysis_batch_batches=${analysis_batch_batches}"
  echo "  avalanche_combination_samples=${avalanche_combination_samples}"
  echo "  avalanche_combination_size=${avalanche_combination_size}"
  echo "  avalanche_combination_pool_size=${avalanche_combination_pool_size}"
  echo "  avalanche_combination_majority_vote=${avalanche_combination_majority_vote}"
  echo "  avalanche_combination_sample_smoothing=${avalanche_combination_sample_smoothing}"
  echo "  avalanche_combination_majority_vote_print=${avalanche_combination_majority_vote_print}"
}

run_current_config() {
  local config_index=$1
  local total_configs=$2
  local analysis_batch_messages=$3
  local analysis_batch_candidates=$4
  local analysis_batch_batches=$5
  local avalanche_combination_samples=$6
  local avalanche_combination_size=$7
  local avalanche_combination_pool_size=$8
  local avalanche_combination_majority_vote=$9
  local avalanche_combination_sample_smoothing=${10}
  local avalanche_combination_majority_vote_print=${11}

  local cx_match_sum=0
  local cx_match_sumsq=0
  local cx_match_min=""
  local cx_match_max=""
  local cx_match_count=0
  local cx_candidates_sum=0
  local cx_candidates_count=0
  local avalanche_candidates_sum=0
  local avalanche_candidates_count=0
  local pass_count=0
  local fail_count=0
  local total_ns=0
  local session_path=""
  local run_stamp
  run_stamp=$(date +"%Y%m%d_%H%M%S_%N")

  echo ""
  echo "===== CONFIG ${config_index}/${total_configs} ====="
  echo "Running ${RUNS_PER_CONFIG} iterations with config ${CONFIG}"
  print_grid_parameters \
    "${analysis_batch_messages}" \
    "${analysis_batch_candidates}" \
    "${analysis_batch_batches}" \
    "${avalanche_combination_samples}" \
    "${avalanche_combination_size}" \
    "${avalanche_combination_pool_size}" \
    "${avalanche_combination_majority_vote}" \
    "${avalanche_combination_sample_smoothing}" \
    "${avalanche_combination_majority_vote_print}"

  mkdir -p "${LOG_DIR}"

  for i in $(seq 1 "${RUNS_PER_CONFIG}"); do
    local seed=$((SEED_START + i - 1))
    local run_output
    local start_ns
    local end_ns
    local duration_ns
    local duration_s
    local status
    local cx_match_line
    local cx_match_pct
    local majority_vote_line
    local majority_vote_match_pct
    local cx_total_line
    local cx_candidates_total
    local avalanche_total_line
    local avalanche_candidates_total
    local verdict
    local match_color
    local majority_match_color
    local verdict_color
    local majority_block
    local beam_block

    run_output="$(mktemp)"
    start_ns=$(date +%s%N)
    session_path="${LOG_DIR}/session_${run_stamp}_cfg_${config_index}_seed_${seed}.json"

    echo ""
    echo "===== CONFIG ${config_index}/${total_configs} RUN ${i}/${RUNS_PER_CONFIG} (seed ${seed}) ====="
    print_grid_parameters \
      "${analysis_batch_messages}" \
      "${analysis_batch_candidates}" \
      "${analysis_batch_batches}" \
      "${avalanche_combination_samples}" \
      "${avalanche_combination_size}" \
      "${avalanche_combination_pool_size}" \
      "${avalanche_combination_majority_vote}" \
      "${avalanche_combination_sample_smoothing}" \
      "${avalanche_combination_majority_vote_print}"

    set +e
    cargo run --bin analysis -- --true --bits 56 --bits-decrypt 128 --seed "${seed}" -c "${CONFIG}" --tests --crypto-rng --session-json "${session_path}" \
      "${TEST_ARGS[@]}" "${EXTRA_ARGS[@]}" \
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
    progress_bar "${i}" "${RUNS_PER_CONFIG}"
    rm -f "${run_output}"
  done

  printf "\n"

  if [[ "${cx_match_count}" -gt 0 ]]; then
    local cx_match_mean
    local cx_match_variance
    local cx_match_stddev
    cx_match_mean=$(awk -v s="${cx_match_sum}" -v n="${cx_match_count}" 'BEGIN { printf "%.4f", s / n }')
    cx_match_variance=$(awk -v s="${cx_match_sum}" -v ss="${cx_match_sumsq}" -v n="${cx_match_count}" 'BEGIN { printf "%.6f", (ss / n) - (s / n) * (s / n) }')
    cx_match_stddev=$(awk -v v="${cx_match_variance}" 'BEGIN { if (v < 0) v = 0; printf "%.4f", sqrt(v) }')
  else
    local cx_match_mean="N/A"
    local cx_match_stddev="N/A"
    cx_match_min="N/A"
    cx_match_max="N/A"
  fi

  if [[ "${cx_candidates_count}" -gt 0 ]]; then
    local cx_candidates_avg
    cx_candidates_avg=$(awk -v s="${cx_candidates_sum}" -v n="${cx_candidates_count}" 'BEGIN { printf "%.4f", s / n }')
  else
    local cx_candidates_avg="N/A"
  fi

  if [[ "${avalanche_candidates_count}" -gt 0 ]]; then
    local avalanche_candidates_avg
    avalanche_candidates_avg=$(awk -v s="${avalanche_candidates_sum}" -v n="${avalanche_candidates_count}" 'BEGIN { printf "%.4f", s / n }')
  else
    local avalanche_candidates_avg="N/A"
  fi

  local avg_time_s
  avg_time_s=$(awk -v ns="${total_ns}" -v n="${RUNS_PER_CONFIG}" 'BEGIN { printf "%.3f", ns / (n * 1000000000) }')

  echo ""
  echo "===== CONFIG ${config_index}/${total_configs} SUMMARY ====="
  echo "c^x max match % stats: mean ${cx_match_mean}, std dev ${cx_match_stddev}, min ${cx_match_min}, max ${cx_match_max}, n ${cx_match_count}"
  echo "c^x evaluated candidates: total ${cx_candidates_sum}, average ${cx_candidates_avg}, n ${cx_candidates_count}"
  echo "Avalanche evaluated candidates: total ${avalanche_candidates_sum}, average ${avalanche_candidates_avg}, n ${avalanche_candidates_count}"
  echo "Verdicts: PASS ${pass_count}, FAIL ${fail_count}"
  echo "Average duration per run: ${avg_time_s}s"
  if [[ -n "${session_path}" ]]; then
    echo "Viewer: python3 scripts/session_viewer.py ${session_path} (Beam vs R tab)"
  fi

  if [[ "${RUN_PCA}" == "1" && -n "${session_path}" ]]; then
    local pca_output="${PCA_OUTPUT%.png}_cfg_${config_index}.png"
    echo ""
    echo "Running PCA clustering via PyTorch script..."
    python3 scripts/r_candidate_cnn.py --session "${session_path}" --config "${CONFIG}" --output "${pca_output}"
    echo "PCA output written to ${pca_output}"
  fi
}

total_configs=$(( \
  ${#analysis_batch_messages_values[@]} * \
  ${#analysis_batch_candidates_values[@]} * \
  ${#analysis_batch_batches_values[@]} * \
  ${#avalanche_combination_samples_values[@]} * \
  ${#avalanche_combination_size_values[@]} * \
  ${#avalanche_combination_pool_size_values[@]} * \
  ${#avalanche_combination_majority_vote_values[@]} * \
  ${#avalanche_combination_sample_smoothing_values[@]} * \
  ${#avalanche_combination_majority_vote_print_values[@]} \
))

echo "Running ${total_configs} grid configurations from ${BASE_CONFIG}"
echo "Each configuration runs ${RUNS_PER_CONFIG} iterations."

config_index=0
for analysis_batch_messages in "${analysis_batch_messages_values[@]}"; do
  for analysis_batch_candidates in "${analysis_batch_candidates_values[@]}"; do
    for analysis_batch_batches in "${analysis_batch_batches_values[@]}"; do
      for avalanche_combination_samples in "${avalanche_combination_samples_values[@]}"; do
        for avalanche_combination_size in "${avalanche_combination_size_values[@]}"; do
          for avalanche_combination_pool_size in "${avalanche_combination_pool_size_values[@]}"; do
            for avalanche_combination_majority_vote in "${avalanche_combination_majority_vote_values[@]}"; do
              for avalanche_combination_sample_smoothing in "${avalanche_combination_sample_smoothing_values[@]}"; do
                for avalanche_combination_majority_vote_print in "${avalanche_combination_majority_vote_print_values[@]}"; do
                  config_index=$((config_index + 1))

                  copy_base_config
                  set_config_field ".engine.analysis_batch_messages" "${analysis_batch_messages}"
                  set_config_field ".engine.analysis_batch_candidates" "${analysis_batch_candidates}"
                  set_config_field ".engine.analysis_batch_batches" "${analysis_batch_batches}"
                  set_config_field ".engine.avalanche_combination_samples" "${avalanche_combination_samples}"
                  set_config_field ".engine.avalanche_combination_size" "${avalanche_combination_size}"
                  set_config_field ".engine.avalanche_combination_pool_size" "${avalanche_combination_pool_size}"
                  set_config_field ".engine.avalanche_combination_majority_vote" "${avalanche_combination_majority_vote}"
                  set_config_field ".engine.avalanche_combination_sample_smoothing" "${avalanche_combination_sample_smoothing}"
                  set_config_field ".engine.avalanche_combination_majority_vote_print" "${avalanche_combination_majority_vote_print}"

                  run_current_config \
                    "${config_index}" \
                    "${total_configs}" \
                    "${analysis_batch_messages}" \
                    "${analysis_batch_candidates}" \
                    "${analysis_batch_batches}" \
                    "${avalanche_combination_samples}" \
                    "${avalanche_combination_size}" \
                    "${avalanche_combination_pool_size}" \
                    "${avalanche_combination_majority_vote}" \
                    "${avalanche_combination_sample_smoothing}" \
                    "${avalanche_combination_majority_vote_print}"
                done
              done
            done
          done
        done
      done
    done
  done
done
