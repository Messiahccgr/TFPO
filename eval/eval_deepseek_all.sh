#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN_DEFAULT="python3"
else
  PYTHON_BIN_DEFAULT="python"
fi

# ---------------------------------------------------------------------------
# Editable defaults
# Environment variables with the same names still take priority.
# ---------------------------------------------------------------------------
DEFAULT_PYTHON_BIN="${PYTHON_BIN_DEFAULT}"
DEFAULT_CONFIGS="configs/experiments/deepseek_r1_distill_qwen_1_5b_mainline_rl.jsonnet"
DEFAULT_MODEL_SOURCE=""
DEFAULT_MAX_TOKENS="2048"
DEFAULT_MATH500_MAX_SAMPLES=""
DEFAULT_GSM8K_MAX_SAMPLES=""
DEFAULT_AIME24_MAX_SAMPLES=""
DEFAULT_VLLM_GPU_IDX="7"
DEFAULT_NUM_INFERENCE_GPUS="1"
DEFAULT_GPU_IDS=""
DEFAULT_MAX_SAMPLES=""
DEFAULT_ENABLE_PASS_K="false"
DEFAULT_VLLM_DTYPE=""
DEFAULT_VLLM_ENFORCE_EAGER=""
DEFAULT_VLLM_ENABLE_PREFIX_CACHING=""
DEFAULT_VLLM_USE_V1_ENGINE=""
DEFAULT_VLLM_MAX_NUM_SEQS="512"
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION="0.7"
DEFAULT_EVAL_REQUEST_TIMEOUT_S="6000"
DEFAULT_EVAL_MAX_PARALLEL_REQUESTS="128"
DEFAULT_PASS_K_NUM_SAMPLES="4"
DEFAULT_PASS_K_TEMPERATURE="0.6"
DEFAULT_PASS_K_TOP_P="0.9"
DEFAULT_PASS_K_MAX_TOKENS=""

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
CONFIGS="${CONFIGS:-${DEFAULT_CONFIGS}}"
MODEL_SOURCE="${MODEL_SOURCE:-${DEFAULT_MODEL_SOURCE}}"
MAX_TOKENS="${MAX_TOKENS:-${DEFAULT_MAX_TOKENS}}"
MAX_SAMPLES="${MAX_SAMPLES:-${DEFAULT_MAX_SAMPLES}}"
ENABLE_PASS_K="${ENABLE_PASS_K:-${DEFAULT_ENABLE_PASS_K}}"
MATH500_MAX_SAMPLES="${MATH500_MAX_SAMPLES:-${DEFAULT_MATH500_MAX_SAMPLES:-${MAX_SAMPLES}}}"
GSM8K_MAX_SAMPLES="${GSM8K_MAX_SAMPLES:-${DEFAULT_GSM8K_MAX_SAMPLES:-${MAX_SAMPLES}}}"
AIME24_MAX_SAMPLES="${AIME24_MAX_SAMPLES:-${DEFAULT_AIME24_MAX_SAMPLES:-${MAX_SAMPLES}}}"
VLLM_GPU_IDX="${VLLM_GPU_IDX:-${DEFAULT_VLLM_GPU_IDX}}"
NUM_INFERENCE_GPUS="${NUM_INFERENCE_GPUS:-${DEFAULT_NUM_INFERENCE_GPUS}}"
GPU_IDS="${GPU_IDS:-${DEFAULT_GPU_IDS}}"
VLLM_DTYPE="${VLLM_DTYPE:-${DEFAULT_VLLM_DTYPE}}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-${DEFAULT_VLLM_ENFORCE_EAGER}}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-${DEFAULT_VLLM_ENABLE_PREFIX_CACHING}}"
VLLM_USE_V1_ENGINE="${VLLM_USE_V1_ENGINE:-${DEFAULT_VLLM_USE_V1_ENGINE}}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-${DEFAULT_VLLM_MAX_NUM_SEQS}}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-${DEFAULT_VLLM_GPU_MEMORY_UTILIZATION}}"
EVAL_REQUEST_TIMEOUT_S="${EVAL_REQUEST_TIMEOUT_S:-${DEFAULT_EVAL_REQUEST_TIMEOUT_S}}"
EVAL_MAX_PARALLEL_REQUESTS="${EVAL_MAX_PARALLEL_REQUESTS:-${DEFAULT_EVAL_MAX_PARALLEL_REQUESTS}}"
PASS_K_NUM_SAMPLES="${PASS_K_NUM_SAMPLES:-${DEFAULT_PASS_K_NUM_SAMPLES}}"
PASS_K_TEMPERATURE="${PASS_K_TEMPERATURE:-${DEFAULT_PASS_K_TEMPERATURE}}"
PASS_K_TOP_P="${PASS_K_TOP_P:-${DEFAULT_PASS_K_TOP_P}}"
PASS_K_MAX_TOKENS="${PASS_K_MAX_TOKENS:-${DEFAULT_PASS_K_MAX_TOKENS:-${MAX_TOKENS}}}"

find_latest_default_model_source() {
  local experiments_subdir="experiments/GRPO"
  local latest_run=""
  local latest_checkpoint=""

  if [[ ! -d "${experiments_subdir}" ]]; then
    return
  fi

  latest_run="$(
    find "${experiments_subdir}" -mindepth 1 -maxdepth 1 -type d \
      -name 'deepseek_r1_distill_qwen_1_5b_grpo_bigmath_processed_curriculum_phase1_2k_grpo_*' \
      | LC_ALL=C sort -r | head -n 1 || true
  )"
  if [[ -z "${latest_run}" ]]; then
    return
  fi

  latest_checkpoint="$(
    find "${latest_run}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
      | LC_ALL=C sort -V -r | head -n 1 || true
  )"
  if [[ -n "${latest_checkpoint}" ]]; then
    printf '%s\n' "${latest_checkpoint}"
  fi
}

# Optional positional overrides:
#   bash eval_deepseek_math500.sh /path/to/model_or_ckpt 1024 50
if [[ $# -ge 1 ]]; then
  MODEL_SOURCE="$1"
fi
if [[ $# -ge 2 ]]; then
  MAX_TOKENS="$2"
fi
if [[ $# -ge 3 ]]; then
  MAX_SAMPLES="$3"
  MATH500_MAX_SAMPLES="${MATH500_MAX_SAMPLES:-${MAX_SAMPLES}}"
  GSM8K_MAX_SAMPLES="${GSM8K_MAX_SAMPLES:-${MAX_SAMPLES}}"
  AIME24_MAX_SAMPLES="${AIME24_MAX_SAMPLES:-${MAX_SAMPLES}}"
fi

if [[ -z "${MODEL_SOURCE}" ]]; then
  MODEL_SOURCE="$(find_latest_default_model_source)"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -z "${MODEL_SOURCE}" ]]; then
  echo "MODEL_SOURCE must not be empty." >&2
  exit 1
fi

if ! [[ "${MAX_TOKENS}" =~ ^[0-9]+$ ]] || [[ "${MAX_TOKENS}" -le 0 ]]; then
  echo "MAX_TOKENS must be a positive integer, got: ${MAX_TOKENS}" >&2
  exit 1
fi

if [[ -n "${ENABLE_PASS_K}" ]]; then
  case "${ENABLE_PASS_K}" in
    true|false)
      ;;
    *)
      echo "ENABLE_PASS_K must be 'true' or 'false', got: ${ENABLE_PASS_K}" >&2
      exit 1
      ;;
  esac
fi

if ! [[ "${PASS_K_NUM_SAMPLES}" =~ ^[0-9]+$ ]] || [[ "${PASS_K_NUM_SAMPLES}" -le 0 ]]; then
  echo "PASS_K_NUM_SAMPLES must be a positive integer, got: ${PASS_K_NUM_SAMPLES}" >&2
  exit 1
fi

if ! [[ "${PASS_K_MAX_TOKENS}" =~ ^[0-9]+$ ]] || [[ "${PASS_K_MAX_TOKENS}" -le 0 ]]; then
  echo "PASS_K_MAX_TOKENS must be a positive integer, got: ${PASS_K_MAX_TOKENS}" >&2
  exit 1
fi

if ! [[ "${VLLM_MAX_NUM_SEQS}" =~ ^[0-9]+$ ]] || [[ "${VLLM_MAX_NUM_SEQS}" -le 0 ]]; then
  echo "VLLM_MAX_NUM_SEQS must be a positive integer, got: ${VLLM_MAX_NUM_SEQS}" >&2
  exit 1
fi

if [[ -n "${VLLM_ENFORCE_EAGER}" ]]; then
  case "${VLLM_ENFORCE_EAGER}" in
    true|false)
      ;;
    *)
      echo "VLLM_ENFORCE_EAGER must be 'true' or 'false', got: ${VLLM_ENFORCE_EAGER}" >&2
      exit 1
      ;;
  esac
fi

if [[ -n "${VLLM_ENABLE_PREFIX_CACHING}" ]]; then
  case "${VLLM_ENABLE_PREFIX_CACHING}" in
    true|false)
      ;;
    *)
      echo "VLLM_ENABLE_PREFIX_CACHING must be 'true' or 'false', got: ${VLLM_ENABLE_PREFIX_CACHING}" >&2
      exit 1
      ;;
  esac
fi

if [[ -n "${VLLM_USE_V1_ENGINE}" ]]; then
  case "${VLLM_USE_V1_ENGINE}" in
    true|false)
      ;;
    *)
      echo "VLLM_USE_V1_ENGINE must be 'true' or 'false', got: ${VLLM_USE_V1_ENGINE}" >&2
      exit 1
      ;;
  esac
fi

if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION}" ]]; then
  if ! [[ "${VLLM_GPU_MEMORY_UTILIZATION}" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "VLLM_GPU_MEMORY_UTILIZATION must be numeric, got: ${VLLM_GPU_MEMORY_UTILIZATION}" >&2
    exit 1
  fi
fi

if ! [[ "${EVAL_REQUEST_TIMEOUT_S}" =~ ^[0-9]+$ ]] || [[ "${EVAL_REQUEST_TIMEOUT_S}" -le 0 ]]; then
  echo "EVAL_REQUEST_TIMEOUT_S must be a positive integer, got: ${EVAL_REQUEST_TIMEOUT_S}" >&2
  exit 1
fi

if ! [[ "${EVAL_MAX_PARALLEL_REQUESTS}" =~ ^[0-9]+$ ]] || [[ "${EVAL_MAX_PARALLEL_REQUESTS}" -le 0 ]]; then
  echo "EVAL_MAX_PARALLEL_REQUESTS must be a positive integer, got: ${EVAL_MAX_PARALLEL_REQUESTS}" >&2
  exit 1
fi

MODEL_TAG="$(basename "${MODEL_SOURCE}")"
OUTPUT_ROOT_DEFAULT="experiments/manual_eval/deepseek_all_datasets/${MODEL_TAG}/max_tokens_${MAX_TOKENS}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_ROOT_DEFAULT}}"

TEMP_CONFIG=""
cleanup() {
  if [[ -n "${TEMP_CONFIG}" && -f "${TEMP_CONFIG}" ]]; then
    rm -f "${TEMP_CONFIG}"
  fi
}
trap cleanup EXIT

resolve_selected_gpu_ids() {
  if [[ "${NUM_INFERENCE_GPUS}" -gt 1 ]]; then
    if [[ -n "${GPU_IDS}" ]]; then
      printf '%s\n' "${GPU_IDS}"
      return
    fi

    local ids=()
    local idx
    for ((idx = 0; idx < NUM_INFERENCE_GPUS; idx++)); do
      ids+=("${idx}")
    done
    local joined
    joined="$(IFS=,; echo "${ids[*]}")"
    printf '%s\n' "${joined}"
    return
  fi

  printf '%s\n' "${VLLM_GPU_IDX}"
}

detect_v100_dtype_override() {
  if [[ -n "${VLLM_DTYPE}" ]]; then
    printf '%s\n' "${VLLM_DTYPE}"
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi

  local selected_gpu_ids
  selected_gpu_ids="$(resolve_selected_gpu_ids)"

  local selected_gpu_names
  selected_gpu_names="$(
    nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits | awk -F',' -v ids="${selected_gpu_ids}" '
      BEGIN {
        n = split(ids, a, ",")
        for (i = 1; i <= n; i++) {
          gsub(/^[ \t]+|[ \t]+$/, "", a[i])
          wanted[a[i]] = 1
        }
      }
      {
        idx = $1
        name = $2
        gsub(/^[ \t]+|[ \t]+$/, "", idx)
        sub(/^[ \t]+/, "", name)
        if (idx in wanted) {
          print name
        }
      }
    '
  )"

  if [[ -n "${selected_gpu_names}" ]] && grep -qi 'V100' <<< "${selected_gpu_names}"; then
    printf '%s\n' "float16"
  fi
}

detect_v100_enforce_eager_override() {
  if [[ -n "${VLLM_ENFORCE_EAGER}" ]]; then
    printf '%s\n' "${VLLM_ENFORCE_EAGER}"
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi

  local selected_gpu_ids
  selected_gpu_ids="$(resolve_selected_gpu_ids)"

  local selected_gpu_names
  selected_gpu_names="$(
    nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits | awk -F',' -v ids="${selected_gpu_ids}" '
      BEGIN {
        n = split(ids, a, ",")
        for (i = 1; i <= n; i++) {
          gsub(/^[ \t]+|[ \t]+$/, "", a[i])
          wanted[a[i]] = 1
        }
      }
      {
        idx = $1
        name = $2
        gsub(/^[ \t]+|[ \t]+$/, "", idx)
        sub(/^[ \t]+/, "", name)
        if (idx in wanted) {
          print name
        }
      }
    '
  )"

  if [[ -n "${selected_gpu_names}" ]] && grep -qi 'V100' <<< "${selected_gpu_names}"; then
    printf '%s\n' "true"
  fi
}

detect_v100_prefix_caching_override() {
  if [[ -n "${VLLM_ENABLE_PREFIX_CACHING}" ]]; then
    printf '%s\n' "${VLLM_ENABLE_PREFIX_CACHING}"
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi

  local selected_gpu_ids
  selected_gpu_ids="$(resolve_selected_gpu_ids)"

  local selected_gpu_names
  selected_gpu_names="$(
    nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits | awk -F',' -v ids="${selected_gpu_ids}" '
      BEGIN {
        n = split(ids, a, ",")
        for (i = 1; i <= n; i++) {
          gsub(/^[ \t]+|[ \t]+$/, "", a[i])
          wanted[a[i]] = 1
        }
      }
      {
        idx = $1
        name = $2
        gsub(/^[ \t]+|[ \t]+$/, "", idx)
        sub(/^[ \t]+/, "", name)
        if (idx in wanted) {
          print name
        }
      }
    '
  )"

  if [[ -n "${selected_gpu_names}" ]] && grep -qi 'V100' <<< "${selected_gpu_names}"; then
    printf '%s\n' "false"
  fi
}

detect_v100_use_v1_engine_override() {
  if [[ -n "${VLLM_USE_V1_ENGINE}" ]]; then
    printf '%s\n' "${VLLM_USE_V1_ENGINE}"
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi

  local selected_gpu_ids
  selected_gpu_ids="$(resolve_selected_gpu_ids)"

  local selected_gpu_names
  selected_gpu_names="$(
    nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits | awk -F',' -v ids="${selected_gpu_ids}" '
      BEGIN {
        n = split(ids, a, ",")
        for (i = 1; i <= n; i++) {
          gsub(/^[ \t]+|[ \t]+$/, "", a[i])
          wanted[a[i]] = 1
        }
      }
      {
        idx = $1
        name = $2
        gsub(/^[ \t]+|[ \t]+$/, "", idx)
        sub(/^[ \t]+/, "", name)
        if (idx in wanted) {
          print name
        }
      }
    '
  )"

  if [[ -n "${selected_gpu_names}" ]] && grep -qi 'V100' <<< "${selected_gpu_names}"; then
    printf '%s\n' "false"
  fi
}

prepare_config_overrides() {
  local effective_dtype
  effective_dtype="$(detect_v100_dtype_override)"
  local effective_enforce_eager
  effective_enforce_eager="$(detect_v100_enforce_eager_override)"
  local effective_prefix_caching
  effective_prefix_caching="$(detect_v100_prefix_caching_override)"
  local effective_use_v1_engine
  effective_use_v1_engine="$(detect_v100_use_v1_engine_override)"

  EFFECTIVE_CONFIGS="${CONFIGS}"
  EFFECTIVE_DTYPE=""
  EFFECTIVE_VLLM_ENFORCE_EAGER=""
  EFFECTIVE_VLLM_ENABLE_PREFIX_CACHING=""
  EFFECTIVE_VLLM_USE_V1_ENGINE=""
  EFFECTIVE_VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}"
  EFFECTIVE_VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}"
  EFFECTIVE_EVAL_REQUEST_TIMEOUT_S="${EVAL_REQUEST_TIMEOUT_S}"
  EFFECTIVE_EVAL_MAX_PARALLEL_REQUESTS="${EVAL_MAX_PARALLEL_REQUESTS}"

  if [[ -n "${effective_dtype}" ]]; then
    EFFECTIVE_DTYPE="${effective_dtype}"
  fi
  if [[ -n "${effective_enforce_eager}" ]]; then
    EFFECTIVE_VLLM_ENFORCE_EAGER="${effective_enforce_eager}"
  fi
  if [[ -n "${effective_prefix_caching}" ]]; then
    EFFECTIVE_VLLM_ENABLE_PREFIX_CACHING="${effective_prefix_caching}"
  fi
  if [[ -n "${effective_use_v1_engine}" ]]; then
    EFFECTIVE_VLLM_USE_V1_ENGINE="${effective_use_v1_engine}"
  fi

  TEMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/deepseek_eval_vllm_override.XXXXXX.jsonnet")"
  {
    printf '{\n'
    printf '  vllm+: {\n'
    if [[ -n "${EFFECTIVE_DTYPE}" ]]; then
      printf "    dtype: '%s',\n" "${EFFECTIVE_DTYPE}"
    fi
    if [[ -n "${EFFECTIVE_VLLM_ENFORCE_EAGER}" ]]; then
      printf '    enforce_eager: %s,\n' "${EFFECTIVE_VLLM_ENFORCE_EAGER}"
    fi
    if [[ -n "${EFFECTIVE_VLLM_ENABLE_PREFIX_CACHING}" ]]; then
      printf '    enable_prefix_caching: %s,\n' "${EFFECTIVE_VLLM_ENABLE_PREFIX_CACHING}"
    fi
    if [[ -n "${EFFECTIVE_VLLM_USE_V1_ENGINE}" ]]; then
      printf '    use_v1_engine: %s,\n' "${EFFECTIVE_VLLM_USE_V1_ENGINE}"
    fi
    if [[ -n "${EFFECTIVE_VLLM_GPU_MEMORY_UTILIZATION}" ]]; then
      printf '    gpu_memory_utilization: %s,\n' "${EFFECTIVE_VLLM_GPU_MEMORY_UTILIZATION}"
    fi
    printf '    max_num_seqs: %s,\n' "${EFFECTIVE_VLLM_MAX_NUM_SEQS}"
    printf '  },\n'
    printf '  evaluation+: {\n'
    printf '    request_timeout_s: %s,\n' "${EFFECTIVE_EVAL_REQUEST_TIMEOUT_S}"
    printf '    max_parallel_requests: %s,\n' "${EFFECTIVE_EVAL_MAX_PARALLEL_REQUESTS}"
    printf '  },\n'
    printf '}\n'
  } > "${TEMP_CONFIG}"
  EFFECTIVE_CONFIGS="${CONFIGS},${TEMP_CONFIG}"
}

run_eval() {
  local eval_name="$1"
  local output_subdir="$2"
  local dataset_name="$3"
  local dataset_config="$4"
  local dataset_split="$5"
  local question_field="$6"
  local answer_field="$7"
  local max_samples="$8"

  if [[ "${max_samples}" == "0" ]]; then
    printf '\n[%s] Skip because max_samples=0\n' "${eval_name}"
    return
  fi

  local output_dir="${OUTPUT_ROOT}/${output_subdir}"
  local cmd=(
    "${PYTHON_BIN}"
    "eval/eval_model.py"
    "--configs" "${EFFECTIVE_CONFIGS}"
    "--model-source" "${MODEL_SOURCE}"
    "--dataset-name" "${dataset_name}"
    "--dataset-split" "${dataset_split}"
    "--question-field" "${question_field}"
    "--answer-field" "${answer_field}"
    "--max-tokens" "${MAX_TOKENS}"
    "--pass-k-num-samples" "${PASS_K_NUM_SAMPLES}"
    "--pass-k-temperature" "${PASS_K_TEMPERATURE}"
    "--pass-k-top-p" "${PASS_K_TOP_P}"
    "--pass-k-max-tokens" "${PASS_K_MAX_TOKENS}"
    "--output-dir" "${output_dir}"
    "--num-inference-gpus" "${NUM_INFERENCE_GPUS}"
  )

  if [[ -n "${ENABLE_PASS_K}" ]]; then
    if [[ "${ENABLE_PASS_K}" == "true" ]]; then
      cmd+=("--enable-pass-k")
    else
      cmd+=("--disable-pass-k")
    fi
  fi

  if [[ -n "${dataset_config}" ]]; then
    cmd+=("--dataset-config" "${dataset_config}")
  fi

  if [[ -n "${max_samples}" ]]; then
    cmd+=("--max-samples" "${max_samples}")
  fi

  if [[ "${NUM_INFERENCE_GPUS}" -gt 1 ]]; then
    if [[ -n "${GPU_IDS}" ]]; then
      cmd+=("--gpu-ids" "${GPU_IDS}")
    fi
  else
    cmd+=("--vllm-gpu-idx" "${VLLM_GPU_IDX}")
  fi

  printf '\n[%s] Running eval:\n' "${eval_name}"
  printf '  model_source:   %s\n' "${MODEL_SOURCE}"
  printf '  dataset_name:   %s\n' "${dataset_name}"
  if [[ -n "${dataset_config}" ]]; then
    printf '  dataset_config: %s\n' "${dataset_config}"
  fi
  printf '  dataset_split:  %s\n' "${dataset_split}"
  if [[ -n "${ENABLE_PASS_K}" ]]; then
    printf '  enable_pass_k:  %s\n' "${ENABLE_PASS_K}"
  else
    printf '  enable_pass_k:  <inherit config>\n'
  fi
  printf '  max_tokens:     %s\n' "${MAX_TOKENS}"
  printf '  pass_k_samples: %s\n' "${PASS_K_NUM_SAMPLES}"
  printf '  pass_k_temp:    %s\n' "${PASS_K_TEMPERATURE}"
  printf '  pass_k_top_p:   %s\n' "${PASS_K_TOP_P}"
  printf '  pass_k_tokens:  %s\n' "${PASS_K_MAX_TOKENS}"
  if [[ -n "${max_samples}" ]]; then
    printf '  max_samples:    %s\n' "${max_samples}"
  else
    printf '  max_samples:    full split\n'
  fi
  printf '  output_dir:     %s\n' "${output_dir}"
  if [[ -n "${EFFECTIVE_DTYPE}" ]]; then
    printf '  vllm dtype:     %s\n' "${EFFECTIVE_DTYPE}"
  fi
  if [[ -n "${EFFECTIVE_VLLM_ENFORCE_EAGER}" ]]; then
    printf '  vllm eager:     %s\n' "${EFFECTIVE_VLLM_ENFORCE_EAGER}"
  fi
  if [[ -n "${EFFECTIVE_VLLM_ENABLE_PREFIX_CACHING}" ]]; then
    printf '  vllm prefix:    %s\n' "${EFFECTIVE_VLLM_ENABLE_PREFIX_CACHING}"
  fi
  if [[ -n "${EFFECTIVE_VLLM_USE_V1_ENGINE}" ]]; then
    printf '  vllm use v1:    %s\n' "${EFFECTIVE_VLLM_USE_V1_ENGINE}"
  fi
  if [[ -n "${EFFECTIVE_VLLM_GPU_MEMORY_UTILIZATION}" ]]; then
    printf '  vllm mem util:  %s\n' "${EFFECTIVE_VLLM_GPU_MEMORY_UTILIZATION}"
  fi
  printf '  vllm max seqs:  %s\n' "${EFFECTIVE_VLLM_MAX_NUM_SEQS}"
  printf '  req timeout s:  %s\n' "${EFFECTIVE_EVAL_REQUEST_TIMEOUT_S}"
  printf '  max parallel:   %s\n' "${EFFECTIVE_EVAL_MAX_PARALLEL_REQUESTS}"
  printf '  command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  "${cmd[@]}"
}

prepare_config_overrides

printf 'DeepSeek multi-dataset evaluation\n'
printf '  configs:          %s\n' "${EFFECTIVE_CONFIGS}"
printf '  model_source:     %s\n' "${MODEL_SOURCE}"
printf '  output_root:      %s\n' "${OUTPUT_ROOT}"
if [[ -n "${ENABLE_PASS_K}" ]]; then
  printf '  enable_pass_k:    %s\n' "${ENABLE_PASS_K}"
else
  printf '  enable_pass_k:    <inherit config>\n'
fi
printf '  max_tokens:       %s\n' "${MAX_TOKENS}"
printf '  pass_k_samples:   %s\n' "${PASS_K_NUM_SAMPLES}"
printf '  pass_k_temp:      %s\n' "${PASS_K_TEMPERATURE}"
printf '  pass_k_top_p:     %s\n' "${PASS_K_TOP_P}"
printf '  pass_k_tokens:    %s\n' "${PASS_K_MAX_TOKENS}"
printf '  num_infer_gpus:   %s\n' "${NUM_INFERENCE_GPUS}"
printf '  vllm max seqs:    %s\n' "${EFFECTIVE_VLLM_MAX_NUM_SEQS}"
if [[ -n "${EFFECTIVE_VLLM_ENFORCE_EAGER}" ]]; then
  printf '  vllm eager:       %s\n' "${EFFECTIVE_VLLM_ENFORCE_EAGER}"
else
  printf '  vllm eager:       <inherit config>\n'
fi
if [[ -n "${EFFECTIVE_VLLM_ENABLE_PREFIX_CACHING}" ]]; then
  printf '  vllm prefix:      %s\n' "${EFFECTIVE_VLLM_ENABLE_PREFIX_CACHING}"
else
  printf '  vllm prefix:      <inherit config>\n'
fi
if [[ -n "${EFFECTIVE_VLLM_USE_V1_ENGINE}" ]]; then
  printf '  vllm use v1:      %s\n' "${EFFECTIVE_VLLM_USE_V1_ENGINE}"
else
  printf '  vllm use v1:      <inherit runtime>\n'
fi
if [[ -n "${EFFECTIVE_VLLM_GPU_MEMORY_UTILIZATION}" ]]; then
  printf '  vllm mem util:    %s\n' "${EFFECTIVE_VLLM_GPU_MEMORY_UTILIZATION}"
else
  printf '  vllm mem util:    <inherit config>\n'
fi
printf '  req timeout s:    %s\n' "${EFFECTIVE_EVAL_REQUEST_TIMEOUT_S}"
printf '  max parallel req: %s\n' "${EFFECTIVE_EVAL_MAX_PARALLEL_REQUESTS}"
if [[ -n "${EFFECTIVE_DTYPE}" ]]; then
  printf '  dtype override:   %s\n' "${EFFECTIVE_DTYPE}"
else
  printf '  dtype override:   <none>\n'
fi

run_eval \
  "MATH-500" \
  "math500" \
  "./eval_data/MATH-500" \
  "" \
  "test" \
  "problem" \
  "answer" \
  "${MATH500_MAX_SAMPLES}"

run_eval \
  "GSM8K" \
  "gsm8k" \
  "./eval_data/GSM8K" \
  "socratic" \
  "test" \
  "question" \
  "answer" \
  "${GSM8K_MAX_SAMPLES}"

run_eval \
  "AIME24" \
  "aime24" \
  "./eval_data/AIME24" \
  "" \
  "train" \
  "problem" \
  "answer" \
  "${AIME24_MAX_SAMPLES}"

printf '\nFinished DeepSeek multi-dataset evaluation.\n'
printf '  MATH-500 output: %s\n' "${OUTPUT_ROOT}/math500"
printf '  GSM8K output:    %s\n' "${OUTPUT_ROOT}/gsm8k"
printf '  AIME24 output:   %s\n' "${OUTPUT_ROOT}/aime24"
