#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if command -v python3 >/dev/null 2>&1; then
  DEFAULT_PYTHON_BIN="python3"
else
  DEFAULT_PYTHON_BIN="python"
fi

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${PROJECT_ROOT}/experiments}"
CKPT_ROOT_2K="${CKPT_ROOT_2K:-${PROJECT_ROOT}/experiments/best_ckpts/2K}"
CKPT_ROOT_4K="${CKPT_ROOT_4K:-${PROJECT_ROOT}/experiments/best_ckpts/4K}"
PIPELINE_OUTPUT_ROOT="${PIPELINE_OUTPUT_ROOT:-${PROJECT_ROOT}/experiments/best_ckpts_2k4k_token_matrix}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${PIPELINE_OUTPUT_ROOT}}"

SCENARIOS="${SCENARIOS:-2k_short,2k_long,4k_short,4k_long}"

REPEAT_COUNT="${REPEAT_COUNT:-5}"
GPU_ID="${GPU_ID:-0}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.9}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-300}"
MAX_PARALLEL_REQUESTS="${MAX_PARALLEL_REQUESTS:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
SWAP_SPACE="${SWAP_SPACE:-16}"
DTYPE="${DTYPE:-bfloat16}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-800}"

RUN_EVAL="${RUN_EVAL:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
DUAL_GPU_MODE="${DUAL_GPU_MODE:-0}"
GPU_ID_2K="${GPU_ID_2K:-0}"
GPU_ID_4K="${GPU_ID_4K:-1}"
INTERNAL_CHILD_RUN="${INTERNAL_CHILD_RUN:-0}"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

append_csv_items() {
  local raw="$1"
  local target_name="$2"
  local part=""
  local IFS=','
  local parts=()

  read -r -a parts <<< "${raw}"
  for part in "${parts[@]}"; do
    part="$(trim "${part}")"
    [[ -z "${part}" ]] && continue
    case "${target_name}" in
      scenario)
        SCENARIO_ITEMS+=("${part}")
        ;;
      *)
        echo "Unknown CSV target: ${target_name}" >&2
        exit 1
        ;;
    esac
  done
}

join_csv_items() {
  local IFS=','
  printf '%s' "$*"
}

validate_binary_flag() {
  local name="$1"
  local value="$2"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${name} must be 0 or 1, got: ${value}" >&2
    exit 1
  fi
}

validate_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -le 0 ]]; then
    echo "${name} must be a positive integer, got: ${value}" >&2
    exit 1
  fi
}

validate_non_negative_int_optional() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    return
  fi
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer, got: ${value}" >&2
    exit 1
  fi
}

resolve_scenario_config() {
  local scenario="$1"
  case "${scenario}" in
    2k_short)
      SCENARIO_CKPT_ROOT="${CKPT_ROOT_2K}"
      SCENARIO_MAX_TOKENS="2048"
      SCENARIO_MYALGO_EXTRA="552"
      SCENARIO_EVAL_DATASETS="aime24,gsm8k"
      ;;
    2k_long)
      SCENARIO_CKPT_ROOT="${CKPT_ROOT_2K}"
      SCENARIO_MAX_TOKENS="30000"
      SCENARIO_MYALGO_EXTRA="2000"
      SCENARIO_EVAL_DATASETS="aime24,gsm8k,math500"
      ;;
    4k_short)
      SCENARIO_CKPT_ROOT="${CKPT_ROOT_4K}"
      SCENARIO_MAX_TOKENS="4048"
      SCENARIO_MYALGO_EXTRA="552"
      SCENARIO_EVAL_DATASETS="aime24,gsm8k"
      ;;
    4k_long)
      SCENARIO_CKPT_ROOT="${CKPT_ROOT_4K}"
      SCENARIO_MAX_TOKENS="30000"
      SCENARIO_MYALGO_EXTRA="2000"
      SCENARIO_EVAL_DATASETS="aime24,gsm8k,math500"
      ;;
    *)
      echo "Unknown scenario: ${scenario}. Expected one of: 2k_short,2k_long,4k_short,4k_long" >&2
      exit 1
      ;;
  esac
}

print_usage() {
  cat <<EOF
Usage:
  bash eval/run_best_ckpts_aime24_gsm8k_rejudge.sh

Description:
  Execute 4-scenario matrix for checkpoint sampling evaluation only (no rejudge).
  math500 is included only in long-token scenarios (2k_long, 4k_long).

Environment overrides:
  PYTHON_BIN
  EXPERIMENTS_ROOT
  CKPT_ROOT_2K
  CKPT_ROOT_4K
  PIPELINE_OUTPUT_ROOT
  EVAL_OUTPUT_ROOT
  SCENARIOS                          default: 2k_short,2k_long,4k_short,4k_long
  REPEAT_COUNT                       default: 5
  GPU_ID                             default: 0
  MAX_SAMPLES                        optional
  TEMPERATURE                        default: 0.6
  TOP_P                              default: 0.9
  REQUEST_TIMEOUT_S                  default: 300
  MAX_PARALLEL_REQUESTS              default: 128
  GPU_MEMORY_UTILIZATION             default: 0.92
  MAX_NUM_SEQS                       default: 256
  SWAP_SPACE                         default: 16
  DTYPE                              default: bfloat16
  WAIT_TIMEOUT_S                     default: 800
  RUN_EVAL                           default: 1
  FORCE                              default: 0
  DRY_RUN                            default: 0
  DUAL_GPU_MODE                      default: 0
  GPU_ID_2K                          default: 0
  GPU_ID_4K                          default: 1

Examples:
  DRY_RUN=1 bash eval/run_best_ckpts_aime24_gsm8k_rejudge.sh

  SCENARIOS=2k_long MAX_SAMPLES=5 RUN_EVAL=1 bash eval/run_best_ckpts_aime24_gsm8k_rejudge.sh

  DUAL_GPU_MODE=1 GPU_ID_2K=0 GPU_ID_4K=1 bash eval/run_best_ckpts_aime24_gsm8k_rejudge.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

validate_positive_int "REPEAT_COUNT" "${REPEAT_COUNT}"
validate_non_negative_int_optional "MAX_SAMPLES" "${MAX_SAMPLES}"

if ! [[ "${GPU_ID}" =~ ^-?[0-9]+$ ]]; then
  echo "GPU_ID must be an integer, got: ${GPU_ID}" >&2
  exit 1
fi

validate_binary_flag "RUN_EVAL" "${RUN_EVAL}"
validate_binary_flag "FORCE" "${FORCE}"
validate_binary_flag "DRY_RUN" "${DRY_RUN}"
validate_binary_flag "DUAL_GPU_MODE" "${DUAL_GPU_MODE}"
validate_binary_flag "INTERNAL_CHILD_RUN" "${INTERNAL_CHILD_RUN}"

declare -a SCENARIO_ITEMS=()
append_csv_items "${SCENARIOS}" "scenario"

if [[ ${#SCENARIO_ITEMS[@]} -eq 0 ]]; then
  echo "SCENARIOS resolved to an empty list." >&2
  exit 1
fi

if [[ "${DUAL_GPU_MODE}" == "1" && "${INTERNAL_CHILD_RUN}" != "1" ]]; then
  if ! [[ "${GPU_ID_2K}" =~ ^-?[0-9]+$ ]]; then
    echo "GPU_ID_2K must be an integer, got: ${GPU_ID_2K}" >&2
    exit 1
  fi
  if ! [[ "${GPU_ID_4K}" =~ ^-?[0-9]+$ ]]; then
    echo "GPU_ID_4K must be an integer, got: ${GPU_ID_4K}" >&2
    exit 1
  fi

  declare -a SCENARIOS_2K=()
  declare -a SCENARIOS_4K=()
  for scenario in "${SCENARIO_ITEMS[@]}"; do
    if [[ "${scenario}" == 2k_* ]]; then
      SCENARIOS_2K+=("${scenario}")
    elif [[ "${scenario}" == 4k_* ]]; then
      SCENARIOS_4K+=("${scenario}")
    else
      echo "Unknown scenario prefix: ${scenario}" >&2
      exit 1
    fi
  done

  declare -a CHILD_PIDS=()
  declare -a CHILD_LABELS=()
  script_path="${SCRIPT_DIR}/$(basename "$0")"

  if [[ ${#SCENARIOS_2K[@]} -gt 0 ]]; then
    scenarios_2k_csv="$(join_csv_items "${SCENARIOS_2K[@]}")"
    printf '[pipeline] launch 2K worker gpu=%s scenarios=%s\n' "${GPU_ID_2K}" "${scenarios_2k_csv}"
    SCENARIOS="${scenarios_2k_csv}" \
    GPU_ID="${GPU_ID_2K}" \
    DUAL_GPU_MODE="0" \
    INTERNAL_CHILD_RUN="1" \
    bash "${script_path}" &
    CHILD_PIDS+=("$!")
    CHILD_LABELS+=("2K")
  fi

  if [[ ${#SCENARIOS_4K[@]} -gt 0 ]]; then
    scenarios_4k_csv="$(join_csv_items "${SCENARIOS_4K[@]}")"
    printf '[pipeline] launch 4K worker gpu=%s scenarios=%s\n' "${GPU_ID_4K}" "${scenarios_4k_csv}"
    SCENARIOS="${scenarios_4k_csv}" \
    GPU_ID="${GPU_ID_4K}" \
    DUAL_GPU_MODE="0" \
    INTERNAL_CHILD_RUN="1" \
    bash "${script_path}" &
    CHILD_PIDS+=("$!")
    CHILD_LABELS+=("4K")
  fi

  if [[ ${#CHILD_PIDS[@]} -eq 0 ]]; then
    echo "No child workers launched in DUAL_GPU_MODE=1." >&2
    exit 1
  fi

  child_failed=0
  for idx in "${!CHILD_PIDS[@]}"; do
    pid="${CHILD_PIDS[$idx]}"
    label="${CHILD_LABELS[$idx]}"
    if wait "${pid}"; then
      printf '[pipeline] worker %s finished pid=%s\n' "${label}" "${pid}"
    else
      status=$?
      child_failed=1
      printf '[pipeline] worker %s failed pid=%s exit=%s\n' "${label}" "${pid}" "${status}" >&2
    fi
  done

  if [[ "${child_failed}" != "0" ]]; then
    exit 1
  fi
  printf '[pipeline] done (eval only, dual gpu mode)\n'
  exit 0
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  if [[ ! -d "${CKPT_ROOT_2K}" ]]; then
    echo "CKPT_ROOT_2K does not exist: ${CKPT_ROOT_2K}" >&2
    exit 1
  fi
  if [[ ! -d "${CKPT_ROOT_4K}" ]]; then
    echo "CKPT_ROOT_4K does not exist: ${CKPT_ROOT_4K}" >&2
    exit 1
  fi
fi

mkdir -p "${PIPELINE_OUTPUT_ROOT}" "${EVAL_OUTPUT_ROOT}"

printf '[pipeline] project_root:         %s\n' "${PROJECT_ROOT}"
printf '[pipeline] experiments_root:     %s\n' "${EXPERIMENTS_ROOT}"
printf '[pipeline] ckpt_root_2k:         %s\n' "${CKPT_ROOT_2K}"
printf '[pipeline] ckpt_root_4k:         %s\n' "${CKPT_ROOT_4K}"
printf '[pipeline] pipeline_output_root: %s\n' "${PIPELINE_OUTPUT_ROOT}"
printf '[pipeline] eval_output_root:     %s\n' "${EVAL_OUTPUT_ROOT}"
printf '[pipeline] scenarios:            %s\n' "${SCENARIOS}"
printf '[pipeline] repeat_count:         %s\n' "${REPEAT_COUNT}"
printf '[pipeline] run_eval:             %s\n' "${RUN_EVAL}"
printf '[pipeline] force:                %s\n' "${FORCE}"
printf '[pipeline] dry_run:              %s\n' "${DRY_RUN}"
printf '[pipeline] dual_gpu_mode:        %s\n' "${DUAL_GPU_MODE}"

scenario_idx=0
for scenario in "${SCENARIO_ITEMS[@]}"; do
  scenario_idx=$((scenario_idx + 1))
  resolve_scenario_config "${scenario}"

  eval_output_root="${EVAL_OUTPUT_ROOT}/${scenario}/eval"
  mkdir -p "${eval_output_root}"

  printf '\n[pipeline][%d/%d] scenario=%s\n' "${scenario_idx}" "${#SCENARIO_ITEMS[@]}" "${scenario}"
  printf '[pipeline]   ckpt_root:      %s\n' "${SCENARIO_CKPT_ROOT}"
  printf '[pipeline]   eval_datasets:  %s\n' "${SCENARIO_EVAL_DATASETS}"
  printf '[pipeline]   max_tokens:     %s\n' "${SCENARIO_MAX_TOKENS}"
  printf '[pipeline]   myalgo_extra:   %s\n' "${SCENARIO_MYALGO_EXTRA}"

  eval_cmd=(
    "${PYTHON_BIN}"
    "eval/eval_all_checkpoints.py"
    "${SCENARIO_CKPT_ROOT}"
    "--experiments-root" "${EXPERIMENTS_ROOT}"
    "--output-root" "${eval_output_root}"
    "--datasets" "${SCENARIO_EVAL_DATASETS}"
    "--repeat-count" "${REPEAT_COUNT}"
    "--gpu-id" "${GPU_ID}"
    "--max-tokens" "${SCENARIO_MAX_TOKENS}"
    "--myalgo-extra-max-tokens" "${SCENARIO_MYALGO_EXTRA}"
    "--temperature" "${TEMPERATURE}"
    "--top-p" "${TOP_P}"
    "--request-timeout-s" "${REQUEST_TIMEOUT_S}"
    "--max-parallel-requests" "${MAX_PARALLEL_REQUESTS}"
    "--gpu-memory-utilization" "${GPU_MEMORY_UTILIZATION}"
    "--max-num-seqs" "${MAX_NUM_SEQS}"
    "--swap-space" "${SWAP_SPACE}"
    "--dtype" "${DTYPE}"
    "--wait-timeout-s" "${WAIT_TIMEOUT_S}"
  )

  if [[ -n "${MAX_SAMPLES}" ]]; then
    eval_cmd+=("--max-samples" "${MAX_SAMPLES}")
  fi
  if [[ "${FORCE}" == "1" ]]; then
    eval_cmd+=("--force")
  fi

  printf '[pipeline]   eval command:'
  printf ' %q' "${eval_cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    "${eval_cmd[@]}"
  elif [[ ! -d "${eval_output_root}" ]]; then
    echo "Eval output missing while RUN_EVAL=0: ${eval_output_root}" >&2
    exit 1
  fi
done

printf '\n[pipeline] done (eval only)\n'
