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

DEFAULT_EXPERIMENTS_ROOT="${PROJECT_ROOT}/experiments"
DEFAULT_OUTPUT_ROOT="${PROJECT_ROOT}/experiments/manual_eval/all_checkpoints_randomized_5x"
DEFAULT_DATASETS="math500"
DEFAULT_REPEAT_COUNT="5"
DEFAULT_GPU_ID="0"
DEFAULT_MAX_TOKENS="2048"
DEFAULT_MYALGO_EXTRA_MAX_TOKENS="0"
DEFAULT_MAX_TOKENS_INCREMENT_STEP="10"

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
EXPERIMENTS_ROOT="${DEFAULT_EXPERIMENTS_ROOT}"
OUTPUT_ROOT=""
REPEAT_COUNT="${DEFAULT_REPEAT_COUNT}"
GPU_ID="${DEFAULT_GPU_ID}"
MAX_SAMPLES=""
MAX_CHECKPOINTS=""
MAX_TOKENS="${DEFAULT_MAX_TOKENS}"
MYALGO_EXTRA_MAX_TOKENS="${DEFAULT_MYALGO_EXTRA_MAX_TOKENS}"
TEMPERATURE=""
TOP_P=""
REQUEST_TIMEOUT_S=""
MAX_PARALLEL_REQUESTS=""
GPU_MEMORY_UTILIZATION=""
MAX_NUM_SEQS=""
SWAP_SPACE=""
DTYPE=""
WAIT_TIMEOUT_S=""
RUN_ALL_UNDER_ROOT="false"
FORCE="false"
DRY_RUN="false"
FAIL_FAST="false"
INCLUDE_ANCHOR_EMA="false"
INCREMENT_MAX_TOKENS="false"
MAX_TOKENS_INCREMENT_STEP=""

declare -a TARGET_PATHS=()
declare -a SUBDIR_PATHS=()
declare -a DATASET_ITEMS=()
declare -a STOP_SEQS=()

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

append_csv_items() {
  local raw="$1"
  local part=""
  local IFS=','
  read -r -a parts <<< "${raw}"
  for part in "${parts[@]}"; do
    part="$(trim "${part}")"
    if [[ -n "${part}" ]]; then
      DATASET_ITEMS+=("${part}")
    fi
  done
}

join_by_comma() {
  local IFS=','
  printf '%s' "$*"
}

print_usage() {
  cat <<EOF
Usage:
  bash eval/eval_all_ckpts.sh [options] [target_path ...]

Description:
  Wrapper around eval/eval_all_checkpoints.py.
  You can point it at:
    1. a specific checkpoint directory, via --ckpt
    2. a checkpoint parent / experiment directory, via --ckpt-dir
    3. a relative subdirectory under experiments root, via --subdir
    4. positional target paths

Options:
  --ckpt PATH                  Specific checkpoint path, e.g. experiments/.../checkpoint-350
                               or experiments/.../checkpoints/iter_0199_actor
  --ckpt-dir PATH              Directory to scan for checkpoints, e.g. an experiment dir or checkpoints dir
  --subdir REL_PATH            Path relative to experiments root, e.g. GRPO/run_xxx or 1_dpsk_my_algo_2k
  --all                        Scan the whole experiments root
  --experiments-root PATH      Experiments root. Default: ${DEFAULT_EXPERIMENTS_ROOT}
  --output-root PATH           Evaluation output root. Default Python script output:
                               ${DEFAULT_OUTPUT_ROOT}
  --datasets CSV               Dataset aliases, comma-separated. Available: math500,gsm8k,aime24
  -d, --dataset NAME_OR_CSV    Repeatable dataset selector. Can also be comma-separated
  --repeat-count N             Default: ${DEFAULT_REPEAT_COUNT}
  --gpu-id N                   Default: ${DEFAULT_GPU_ID}
  --max-samples N              Optional per-dataset cap
  --max-checkpoints N          Optional max number of discovered checkpoints
  --max-tokens N               Default: ${DEFAULT_MAX_TOKENS}
  --myalgo-extra-max-tokens N  Add N tokens only for checkpoints whose path/name contains
                               "myalgo" or "my_algo". Default: ${DEFAULT_MYALGO_EXTRA_MAX_TOKENS}
  --increment-max-tokens-per-checkpoint [STEP]
                               Increase max_tokens across checkpoints. Without STEP,
                               uses default step=${DEFAULT_MAX_TOKENS_INCREMENT_STEP}.
  --temperature FLOAT
  --top-p FLOAT
  --stop TEXT                  Repeatable stop sequence
  --request-timeout-s N
  --max-parallel-requests N
  --gpu-memory-utilization FLOAT
  --max-num-seqs N
  --swap-space N
  --dtype TEXT
  --wait-timeout-s N
  --include-anchor-ema
  --force
  --dry-run
  --fail-fast
  -h, --help

Examples:
  bash eval/eval_all_ckpts.sh \\
    --ckpt ./experiments/GRPO/run_xxx/checkpoint-350 \\
    --dataset math500 \\
    --gpu-id 0

  bash eval/eval_all_ckpts.sh \\
    --ckpt-dir ./experiments/dpsk_my_algo_2k/2_dpsk_my_algo_2k \\
    --datasets math500,gsm8k \\
    --output-root ./experiments/manual_eval/dpsk_2k_math_gsm8k

  bash eval/eval_all_ckpts.sh \\
    --subdir 1_dpsk_my_algo_2k \\
    --dataset math500 \\
    --increment-max-tokens-per-checkpoint 20

  bash eval/eval_all_ckpts.sh \\
    --all \\
    --experiments-root ./experiments/GRPO \\
    --dataset math500
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt|--checkpoint)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      TARGET_PATHS+=("$2")
      shift 2
      ;;
    --ckpt-dir|--checkpoint-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      TARGET_PATHS+=("$2")
      shift 2
      ;;
    --subdir)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      SUBDIR_PATHS+=("$2")
      shift 2
      ;;
    --all)
      RUN_ALL_UNDER_ROOT="true"
      shift
      ;;
    --experiments-root)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      EXPERIMENTS_ROOT="$2"
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --datasets)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      append_csv_items "$2"
      shift 2
      ;;
    -d|--dataset)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      append_csv_items "$2"
      shift 2
      ;;
    --repeat-count)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      REPEAT_COUNT="$2"
      shift 2
      ;;
    --gpu-id)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      GPU_ID="$2"
      shift 2
      ;;
    --max-samples)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --max-checkpoints)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      MAX_CHECKPOINTS="$2"
      shift 2
      ;;
    --max-tokens)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      MAX_TOKENS="$2"
      shift 2
      ;;
    --myalgo-extra-max-tokens)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      MYALGO_EXTRA_MAX_TOKENS="$2"
      shift 2
      ;;
    --increment-max-tokens-per-checkpoint)
      INCREMENT_MAX_TOKENS="true"
      if [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]]; then
        MAX_TOKENS_INCREMENT_STEP="$2"
        shift 2
      else
        shift
      fi
      ;;
    --temperature)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      TEMPERATURE="$2"
      shift 2
      ;;
    --top-p)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      TOP_P="$2"
      shift 2
      ;;
    --stop)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      STOP_SEQS+=("$2")
      shift 2
      ;;
    --request-timeout-s)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      REQUEST_TIMEOUT_S="$2"
      shift 2
      ;;
    --max-parallel-requests)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      MAX_PARALLEL_REQUESTS="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --max-num-seqs)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      MAX_NUM_SEQS="$2"
      shift 2
      ;;
    --swap-space)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      SWAP_SPACE="$2"
      shift 2
      ;;
    --dtype)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      DTYPE="$2"
      shift 2
      ;;
    --wait-timeout-s)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      WAIT_TIMEOUT_S="$2"
      shift 2
      ;;
    --include-anchor-ema)
      INCLUDE_ANCHOR_EMA="true"
      shift
      ;;
    --force)
      FORCE="true"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --fail-fast)
      FAIL_FAST="true"
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        TARGET_PATHS+=("$1")
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      print_usage >&2
      exit 1
      ;;
    *)
      TARGET_PATHS+=("$1")
      shift
      ;;
  esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

for rel_path in "${SUBDIR_PATHS[@]}"; do
  if [[ "${rel_path}" = /* ]]; then
    TARGET_PATHS+=("${rel_path}")
  else
    TARGET_PATHS+=("${EXPERIMENTS_ROOT}/${rel_path}")
  fi
done

if [[ "${RUN_ALL_UNDER_ROOT}" != "true" && ${#TARGET_PATHS[@]} -eq 0 ]]; then
  echo "Please provide at least one target via --ckpt, --ckpt-dir, --subdir, or positional path." >&2
  echo "Use --all if you really want to scan the whole experiments root." >&2
  exit 1
fi

if [[ "${RUN_ALL_UNDER_ROOT}" == "true" && ${#TARGET_PATHS[@]} -gt 0 ]]; then
  echo "Do not combine --all with explicit target paths." >&2
  exit 1
fi

if [[ ! -d "${EXPERIMENTS_ROOT}" ]]; then
  echo "Experiments root does not exist: ${EXPERIMENTS_ROOT}" >&2
  exit 1
fi

if [[ ${#DATASET_ITEMS[@]} -eq 0 ]]; then
  append_csv_items "${DEFAULT_DATASETS}"
fi

if [[ "${RUN_ALL_UNDER_ROOT}" != "true" ]]; then
  for target in "${TARGET_PATHS[@]}"; do
    if [[ ! -e "${target}" ]]; then
      echo "Target path does not exist: ${target}" >&2
      exit 1
    fi
  done
fi

if [[ "${INCREMENT_MAX_TOKENS}" == "true" && -n "${MAX_TOKENS_INCREMENT_STEP}" ]]; then
  if ! [[ "${MAX_TOKENS_INCREMENT_STEP}" =~ ^[0-9]+$ ]] || [[ "${MAX_TOKENS_INCREMENT_STEP}" -le 0 ]]; then
    echo "Increment step for --increment-max-tokens-per-checkpoint must be a positive integer, got: ${MAX_TOKENS_INCREMENT_STEP}" >&2
    exit 1
  fi
fi

if ! [[ "${MYALGO_EXTRA_MAX_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "MYALGO_EXTRA_MAX_TOKENS must be a non-negative integer, got: ${MYALGO_EXTRA_MAX_TOKENS}" >&2
  exit 1
fi

DATASETS_CSV="$(join_by_comma "${DATASET_ITEMS[@]}")"

cmd=(
  "${PYTHON_BIN}"
  "eval/eval_all_checkpoints.py"
)

if [[ "${RUN_ALL_UNDER_ROOT}" != "true" ]]; then
  cmd+=("${TARGET_PATHS[@]}")
fi

cmd+=(
  "--experiments-root" "${EXPERIMENTS_ROOT}"
  "--datasets" "${DATASETS_CSV}"
  "--repeat-count" "${REPEAT_COUNT}"
  "--gpu-id" "${GPU_ID}"
  "--max-tokens" "${MAX_TOKENS}"
  "--myalgo-extra-max-tokens" "${MYALGO_EXTRA_MAX_TOKENS}"
)

if [[ -n "${OUTPUT_ROOT}" ]]; then
  cmd+=("--output-root" "${OUTPUT_ROOT}")
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  cmd+=("--max-samples" "${MAX_SAMPLES}")
fi
if [[ -n "${MAX_CHECKPOINTS}" ]]; then
  cmd+=("--max-checkpoints" "${MAX_CHECKPOINTS}")
fi
if [[ -n "${TEMPERATURE}" ]]; then
  cmd+=("--temperature" "${TEMPERATURE}")
fi
if [[ -n "${TOP_P}" ]]; then
  cmd+=("--top-p" "${TOP_P}")
fi
if [[ -n "${REQUEST_TIMEOUT_S}" ]]; then
  cmd+=("--request-timeout-s" "${REQUEST_TIMEOUT_S}")
fi
if [[ -n "${MAX_PARALLEL_REQUESTS}" ]]; then
  cmd+=("--max-parallel-requests" "${MAX_PARALLEL_REQUESTS}")
fi
if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  cmd+=("--gpu-memory-utilization" "${GPU_MEMORY_UTILIZATION}")
fi
if [[ -n "${MAX_NUM_SEQS}" ]]; then
  cmd+=("--max-num-seqs" "${MAX_NUM_SEQS}")
fi
if [[ -n "${SWAP_SPACE}" ]]; then
  cmd+=("--swap-space" "${SWAP_SPACE}")
fi
if [[ -n "${DTYPE}" ]]; then
  cmd+=("--dtype" "${DTYPE}")
fi
if [[ -n "${WAIT_TIMEOUT_S}" ]]; then
  cmd+=("--wait-timeout-s" "${WAIT_TIMEOUT_S}")
fi

for stop_seq in "${STOP_SEQS[@]}"; do
  cmd+=("--stop" "${stop_seq}")
done

if [[ "${INCREMENT_MAX_TOKENS}" == "true" ]]; then
  cmd+=("--increment-max-tokens-per-checkpoint")
  if [[ -n "${MAX_TOKENS_INCREMENT_STEP}" ]]; then
    cmd+=("${MAX_TOKENS_INCREMENT_STEP}")
  fi
fi
if [[ "${INCLUDE_ANCHOR_EMA}" == "true" ]]; then
  cmd+=("--include-anchor-ema")
fi
if [[ "${FORCE}" == "true" ]]; then
  cmd+=("--force")
fi
if [[ "${DRY_RUN}" == "true" ]]; then
  cmd+=("--dry-run")
fi
if [[ "${FAIL_FAST}" == "true" ]]; then
  cmd+=("--fail-fast")
fi

printf 'Run eval_all_checkpoints\n'
printf '  python_bin:       %s\n' "${PYTHON_BIN}"
printf '  experiments_root: %s\n' "${EXPERIMENTS_ROOT}"
if [[ -n "${OUTPUT_ROOT}" ]]; then
  printf '  output_root:      %s\n' "${OUTPUT_ROOT}"
else
  printf '  output_root:      %s\n' "${DEFAULT_OUTPUT_ROOT}"
fi
printf '  datasets:         %s\n' "${DATASETS_CSV}"
printf '  repeat_count:     %s\n' "${REPEAT_COUNT}"
printf '  gpu_id:           %s\n' "${GPU_ID}"
printf '  max_tokens:       %s\n' "${MAX_TOKENS}"
printf '  myalgo_extra:     %s\n' "${MYALGO_EXTRA_MAX_TOKENS}"
if [[ "${INCREMENT_MAX_TOKENS}" == "true" ]]; then
  printf '  increment_tokens: %s\n' "${MAX_TOKENS_INCREMENT_STEP:-${DEFAULT_MAX_TOKENS_INCREMENT_STEP}}"
fi
if [[ "${RUN_ALL_UNDER_ROOT}" == "true" ]]; then
  printf '  targets:          <scan whole experiments root>\n'
else
  printf '  targets:\n'
  for target in "${TARGET_PATHS[@]}"; do
    printf '    %s\n' "${target}"
  done
fi
printf '  command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
