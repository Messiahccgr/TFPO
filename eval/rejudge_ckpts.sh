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

DEFAULT_INPUT_ROOT="${PROJECT_ROOT}/experiments/manual_eval/all_checkpoints_randomized_5x"
DEFAULT_OUTPUT_ROOT="${PROJECT_ROOT}/experiments/manual_eval/all_checkpoints_randomized_5x_larger_model_rejudge"
DEFAULT_WORKERS="6"
DEFAULT_BATCH_SIZE="4"
DEFAULT_SAVE_EVERY="20"
DEFAULT_DATASET="math_500_test"
DEFAULT_SOURCE_RUN="best"

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
INPUT_ROOT="${INPUT_ROOT:-${DEFAULT_INPUT_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
WORKERS="${WORKERS:-${DEFAULT_WORKERS}}"
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
MAX_CASES_PER_DATASET="${MAX_CASES_PER_DATASET:-}"
SAVE_EVERY="${SAVE_EVERY:-${DEFAULT_SAVE_EVERY}}"
SOURCE_RUN="${SOURCE_RUN:-${DEFAULT_SOURCE_RUN}}"
LLM_BASE_URL_VALUE="${LLM_BASE_URL:-}"
LLM_MODEL_VALUE="${LLM_MODEL:-}"
LLM_API_KEY_VALUE="${LLM_API_KEY:-}"

declare -a DATASET_ITEMS=()
declare -a CHECKPOINT_ITEMS=()
declare -a INCLUDE_RELATIVE_CHECKPOINT_PATH_ITEMS=()
declare -a EXCLUDE_RELATIVE_CHECKPOINT_PATH_ITEMS=()

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

append_items() {
  local raw="$1"
  local target_name="$2"
  local part=""
  local IFS=','
  read -r -a parts <<< "${raw}"
  for part in "${parts[@]}"; do
    part="$(trim "${part}")"
    if [[ -n "${part}" ]]; then
      if [[ "${target_name}" == "dataset" ]]; then
        DATASET_ITEMS+=("${part}")
      elif [[ "${target_name}" == "checkpoint" ]]; then
        CHECKPOINT_ITEMS+=("${part}")
      elif [[ "${target_name}" == "include_relative_path" ]]; then
        INCLUDE_RELATIVE_CHECKPOINT_PATH_ITEMS+=("${part}")
      elif [[ "${target_name}" == "exclude_relative_path" ]]; then
        EXCLUDE_RELATIVE_CHECKPOINT_PATH_ITEMS+=("${part}")
      else
        echo "Unknown append target: ${target_name}" >&2
        exit 1
      fi
    fi
  done
}

print_usage() {
  cat <<EOF
Usage:
  bash eval/rejudge_ckpts.sh [options]

Description:
  Wrapper around eval/rejudge_ckpt_accuracy_with_larger_model.py.

Options:
  --input-root PATH              Input manual-eval root.
                                 Default: ${DEFAULT_INPUT_ROOT}
  --output-root PATH             Output rejudge root.
                                 Default: ${DEFAULT_OUTPUT_ROOT}
  --dataset NAME_OR_CSV          Repeatable dataset selector. Can also be comma-separated.
                                 Default: ${DEFAULT_DATASET}
  --source-run best|worst        Which run's per-example file to rejudge.
                                 Default: ${DEFAULT_SOURCE_RUN}
  --checkpoint NAME_OR_CSV       Repeatable checkpoint selector. Can also be comma-separated.
  --relative-checkpoint-path-contains TEXT_OR_CSV
                                 Keep only checkpoints whose relative path contains token(s).
  --exclude-relative-checkpoint-path-contains TEXT_OR_CSV
                                 Exclude checkpoints whose relative path contains token(s).
  --workers N                    Concurrent requests to the stronger model.
                                 Default: ${DEFAULT_WORKERS}
  --batch-size N                 Number of wrong cases per request.
                                 Default: ${DEFAULT_BATCH_SIZE}
  --max-cases-per-dataset N      Optional cap for debugging.
  --save-every N                 Flush cache files every N completed batch requests.
                                 Default: ${DEFAULT_SAVE_EVERY}
  --llm-base-url URL             Override LLM_BASE_URL for this run.
  --llm-model NAME               Override LLM_MODEL for this run.
  --llm-api-key KEY              Override LLM_API_KEY for this run.
  --dry-run                      Print resolved command and exit.
  -h, --help

Environment:
  PYTHON_BIN
  INPUT_ROOT
  OUTPUT_ROOT
  WORKERS
  BATCH_SIZE
  MAX_CASES_PER_DATASET
  SAVE_EVERY
  SOURCE_RUN
  LLM_BASE_URL
  LLM_MODEL
  LLM_API_KEY

Examples:
  bash eval/rejudge_ckpts.sh

  bash eval/rejudge_ckpts.sh \\
    --workers 4 \\
    --batch-size 2

  bash eval/rejudge_ckpts.sh \\
    --checkpoint iter_0399_actor \\
    --dataset math_500_test \\
    --source-run worst \\
    --llm-base-url http://127.0.0.1:8000/v1 \\
    --llm-model qwen3.5-397b-a17b
EOF
}

DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      INPUT_ROOT="$2"
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    -d|--dataset)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      append_items "$2" "dataset"
      shift 2
      ;;
    --source-run)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      SOURCE_RUN="$2"
      shift 2
      ;;
    --checkpoint)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      append_items "$2" "checkpoint"
      shift 2
      ;;
    --relative-checkpoint-path-contains)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      append_items "$2" "include_relative_path"
      shift 2
      ;;
    --exclude-relative-checkpoint-path-contains)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      append_items "$2" "exclude_relative_path"
      shift 2
      ;;
    --workers)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      WORKERS="$2"
      shift 2
      ;;
    --batch-size)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max-cases-per-dataset)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      MAX_CASES_PER_DATASET="$2"
      shift 2
      ;;
    --save-every)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      SAVE_EVERY="$2"
      shift 2
      ;;
    --llm-base-url)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      LLM_BASE_URL_VALUE="$2"
      shift 2
      ;;
    --llm-model)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      LLM_MODEL_VALUE="$2"
      shift 2
      ;;
    --llm-api-key)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      LLM_API_KEY_VALUE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

if [[ ${#DATASET_ITEMS[@]} -eq 0 ]]; then
  DATASET_ITEMS+=("${DEFAULT_DATASET}")
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "Input root does not exist: ${INPUT_ROOT}" >&2
  exit 1
fi

if ! [[ "${WORKERS}" =~ ^[0-9]+$ ]] || [[ "${WORKERS}" -le 0 ]]; then
  echo "WORKERS must be a positive integer, got: ${WORKERS}" >&2
  exit 1
fi

if ! [[ "${BATCH_SIZE}" =~ ^[0-9]+$ ]] || [[ "${BATCH_SIZE}" -le 0 ]]; then
  echo "BATCH_SIZE must be a positive integer, got: ${BATCH_SIZE}" >&2
  exit 1
fi

if [[ -n "${MAX_CASES_PER_DATASET}" ]]; then
  if ! [[ "${MAX_CASES_PER_DATASET}" =~ ^[0-9]+$ ]]; then
    echo "MAX_CASES_PER_DATASET must be a non-negative integer, got: ${MAX_CASES_PER_DATASET}" >&2
    exit 1
  fi
fi

if ! [[ "${SAVE_EVERY}" =~ ^[0-9]+$ ]] || [[ "${SAVE_EVERY}" -le 0 ]]; then
  echo "SAVE_EVERY must be a positive integer, got: ${SAVE_EVERY}" >&2
  exit 1
fi

if [[ "${SOURCE_RUN}" != "best" && "${SOURCE_RUN}" != "worst" ]]; then
  echo "SOURCE_RUN must be 'best' or 'worst', got: ${SOURCE_RUN}" >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}"
  "eval/rejudge_ckpt_accuracy_with_larger_model.py"
  "--input-root" "${INPUT_ROOT}"
  "--output-root" "${OUTPUT_ROOT}"
  "--workers" "${WORKERS}"
  "--batch-size" "${BATCH_SIZE}"
  "--save-every" "${SAVE_EVERY}"
  "--source-run" "${SOURCE_RUN}"
)

if [[ -n "${MAX_CASES_PER_DATASET}" ]]; then
  cmd+=("--max-cases-per-dataset" "${MAX_CASES_PER_DATASET}")
fi

for dataset in "${DATASET_ITEMS[@]}"; do
  cmd+=("--dataset" "${dataset}")
done

for checkpoint in "${CHECKPOINT_ITEMS[@]}"; do
  cmd+=("--checkpoint" "${checkpoint}")
done

for token in "${INCLUDE_RELATIVE_CHECKPOINT_PATH_ITEMS[@]}"; do
  cmd+=("--relative-checkpoint-path-contains" "${token}")
done

for token in "${EXCLUDE_RELATIVE_CHECKPOINT_PATH_ITEMS[@]}"; do
  cmd+=("--exclude-relative-checkpoint-path-contains" "${token}")
done

printf 'Run rejudge_ckpt_accuracy_with_larger_model\n'
printf '  python_bin:      %s\n' "${PYTHON_BIN}"
printf '  input_root:      %s\n' "${INPUT_ROOT}"
printf '  output_root:     %s\n' "${OUTPUT_ROOT}"
printf '  workers:         %s\n' "${WORKERS}"
printf '  batch_size:      %s\n' "${BATCH_SIZE}"
printf '  save_every:      %s\n' "${SAVE_EVERY}"
printf '  source_run:      %s\n' "${SOURCE_RUN}"
printf '  llm_base_url:    %s\n' "${LLM_BASE_URL_VALUE:-<inherit/default>}"
printf '  llm_model:       %s\n' "${LLM_MODEL_VALUE:-<inherit/default>}"
if [[ -n "${LLM_API_KEY_VALUE}" ]]; then
  printf '  llm_api_key:     <provided>\n'
else
  printf '  llm_api_key:     <inherit/default>\n'
fi
if [[ ${#DATASET_ITEMS[@]} -gt 0 ]]; then
  printf '  datasets:\n'
  for dataset in "${DATASET_ITEMS[@]}"; do
    printf '    %s\n' "${dataset}"
  done
fi
if [[ ${#CHECKPOINT_ITEMS[@]} -gt 0 ]]; then
  printf '  checkpoints:\n'
  for checkpoint in "${CHECKPOINT_ITEMS[@]}"; do
    printf '    %s\n' "${checkpoint}"
  done
fi
if [[ ${#INCLUDE_RELATIVE_CHECKPOINT_PATH_ITEMS[@]} -gt 0 ]]; then
  printf '  include_relative_paths:\n'
  for token in "${INCLUDE_RELATIVE_CHECKPOINT_PATH_ITEMS[@]}"; do
    printf '    %s\n' "${token}"
  done
fi
if [[ ${#EXCLUDE_RELATIVE_CHECKPOINT_PATH_ITEMS[@]} -gt 0 ]]; then
  printf '  exclude_relative_paths:\n'
  for token in "${EXCLUDE_RELATIVE_CHECKPOINT_PATH_ITEMS[@]}"; do
    printf '    %s\n' "${token}"
  done
fi
printf '  command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

if [[ -n "${LLM_BASE_URL_VALUE}" ]]; then
  export LLM_BASE_URL="${LLM_BASE_URL_VALUE}"
fi
if [[ -n "${LLM_MODEL_VALUE}" ]]; then
  export LLM_MODEL="${LLM_MODEL_VALUE}"
fi
if [[ -n "${LLM_API_KEY_VALUE}" ]]; then
  export LLM_API_KEY="${LLM_API_KEY_VALUE}"
fi

"${cmd[@]}"
