#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

SOURCE_RUN="${SOURCE_RUN:-worst}"
WORKERS="${WORKERS:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_EVERY="${SAVE_EVERY:-10}"
MAX_CASES_PER_DATASET="${MAX_CASES_PER_DATASET:-}"
DATASET_KEY="${DATASET_KEY:-}"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_LLM_BASE_URLS="http://10.128.202.100:3010/v1,https://api.siliconflow.cn/v1"
DEFAULT_LLM_MODELS="qwen3.5-397b-a17b,Qwen/Qwen3.5-397B-A17B"
DEFAULT_LLM_API_KEYS="sk-mHbHbIE1xWN7khlheTnL6E7eTRmHR1aQpMamEnasIk1S7jEx,sk-yqrulcqpqeccxiuxymhzabbdqukaoikzngqnlcchdxmsiflv"

LLM_BASE_URLS_RAW="${LLM_BASE_URLS:-${LLM_BASE_URL:-${DEFAULT_LLM_BASE_URLS}}}"
LLM_MODELS_RAW="${LLM_MODELS:-${LLM_MODEL:-${DEFAULT_LLM_MODELS}}}"
LLM_API_KEYS_RAW="${LLM_API_KEYS:-${LLM_API_KEY:-${DEFAULT_LLM_API_KEYS}}}"
WORKERS_PER_URLS_RAW="${WORKERS_PER_URLS:-}"
BATCH_SIZE_PER_URLS_RAW="${BATCH_SIZE_PER_URLS:-}"
PER_URL_JOB_LIMIT="${PER_URL_JOB_LIMIT:-1}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-1}"

declare -a JOB_LABELS=()
declare -a JOB_INPUT_ROOTS=()
declare -a JOB_OUTPUT_ROOTS=()

declare -a BASE_URL_ITEMS=()
declare -a MODEL_ITEMS=()
declare -a API_KEY_ITEMS=()
declare -a WORKER_ITEMS=()
declare -a BATCH_SIZE_ITEMS=()

declare -a SLOT_ENDPOINT_INDEXES=()
declare -a ACTIVE_PIDS=()
declare -a ACTIVE_LABELS=()
declare -a ACTIVE_ENDPOINTS=()

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
      base_url)
        BASE_URL_ITEMS+=("${part}")
        ;;
      model)
        MODEL_ITEMS+=("${part}")
        ;;
      api_key)
        API_KEY_ITEMS+=("${part}")
        ;;
      worker)
        WORKER_ITEMS+=("${part}")
        ;;
      batch_size)
        BATCH_SIZE_ITEMS+=("${part}")
        ;;
      *)
        printf 'Unknown target list: %s\n' "${target_name}" >&2
        exit 1
        ;;
    esac
  done
}

resolve_endpoint_lists() {
  if [[ -n "${LLM_BASE_URLS_RAW}" ]]; then
    append_csv_items "${LLM_BASE_URLS_RAW}" "base_url"
  elif [[ -n "${LLM_BASE_URL:-}" ]]; then
    BASE_URL_ITEMS+=("${LLM_BASE_URL}")
  fi

  if [[ -n "${LLM_MODELS_RAW}" ]]; then
    append_csv_items "${LLM_MODELS_RAW}" "model"
  elif [[ -n "${LLM_MODEL:-}" ]]; then
    MODEL_ITEMS+=("${LLM_MODEL}")
  fi

  if [[ -n "${LLM_API_KEYS_RAW}" ]]; then
    append_csv_items "${LLM_API_KEYS_RAW}" "api_key"
  elif [[ -n "${LLM_API_KEY:-}" ]]; then
    API_KEY_ITEMS+=("${LLM_API_KEY}")
  fi

  if [[ -n "${WORKERS_PER_URLS_RAW}" ]]; then
    append_csv_items "${WORKERS_PER_URLS_RAW}" "worker"
  fi

  if [[ -n "${BATCH_SIZE_PER_URLS_RAW}" ]]; then
    append_csv_items "${BATCH_SIZE_PER_URLS_RAW}" "batch_size"
  fi

  if [[ ${#BASE_URL_ITEMS[@]} -eq 0 ]]; then
    BASE_URL_ITEMS+=("")
  fi
}

pick_base_url() {
  local endpoint_idx="$1"
  if [[ ${#BASE_URL_ITEMS[@]} -eq 0 ]]; then
    printf ''
    return 0
  fi
  printf '%s' "${BASE_URL_ITEMS[$(( endpoint_idx % ${#BASE_URL_ITEMS[@]} ))]}"
}

pick_model() {
  local endpoint_idx="$1"
  if [[ ${#MODEL_ITEMS[@]} -eq 0 ]]; then
    printf ''
    return 0
  fi
  printf '%s' "${MODEL_ITEMS[$(( endpoint_idx % ${#MODEL_ITEMS[@]} ))]}"
}

pick_api_key() {
  local endpoint_idx="$1"
  if [[ ${#API_KEY_ITEMS[@]} -eq 0 ]]; then
    printf ''
    return 0
  fi
  printf '%s' "${API_KEY_ITEMS[$(( endpoint_idx % ${#API_KEY_ITEMS[@]} ))]}"
}

pick_workers() {
  local endpoint_idx="$1"
  if [[ ${#WORKER_ITEMS[@]} -eq 0 ]]; then
    printf '%s' "${WORKERS}"
    return 0
  fi
  printf '%s' "${WORKER_ITEMS[$(( endpoint_idx % ${#WORKER_ITEMS[@]} ))]}"
}

pick_batch_size() {
  local endpoint_idx="$1"
  if [[ ${#BATCH_SIZE_ITEMS[@]} -eq 0 ]]; then
    printf '%s' "${BATCH_SIZE}"
    return 0
  fi
  printf '%s' "${BATCH_SIZE_ITEMS[$(( endpoint_idx % ${#BATCH_SIZE_ITEMS[@]} ))]}"
}

add_job() {
  local label="$1"
  local input_root="$2"
  local output_root="$3"

  JOB_LABELS+=("${label}")
  JOB_INPUT_ROOTS+=("${input_root}")
  JOB_OUTPUT_ROOTS+=("${output_root}")
}

build_slots() {
  local endpoint_idx=0
  local replica_idx=0

  for (( endpoint_idx = 0; endpoint_idx < ${#BASE_URL_ITEMS[@]}; endpoint_idx++ )); do
    for (( replica_idx = 0; replica_idx < PER_URL_JOB_LIMIT; replica_idx++ )); do
      SLOT_ENDPOINT_INDEXES+=("${endpoint_idx}")
      ACTIVE_PIDS+=("")
      ACTIVE_LABELS+=("")
      ACTIVE_ENDPOINTS+=("")
    done
  done
}

run_job() {
  local label="$1"
  local input_root="$2"
  local output_root="$3"
  local endpoint_idx="$4"
  local slot_idx="$5"

  local base_url=""
  local model_name=""
  local api_key=""
  local workers_value=""
  local batch_size_value=""

  base_url="$(pick_base_url "${endpoint_idx}")"
  model_name="$(pick_model "${endpoint_idx}")"
  api_key="$(pick_api_key "${endpoint_idx}")"
  workers_value="$(pick_workers "${endpoint_idx}")"
  batch_size_value="$(pick_batch_size "${endpoint_idx}")"

  printf '[rejudge_metrics_from_train] %s\n' "${label}"
  printf '  input_root:   %s\n' "${input_root}"
  printf '  output_root:  %s\n' "${output_root}"
  printf '  source_run:   %s\n' "${SOURCE_RUN}"
  printf '  workers:      %s\n' "${workers_value}"
  printf '  batch_size:   %s\n' "${batch_size_value}"
  printf '  save_every:   %s\n' "${SAVE_EVERY}"
  printf '  slot:         %s\n' "${slot_idx}"
  printf '  endpoint:     %s\n' "${endpoint_idx}"
  printf '  llm_base_url: %s\n' "${base_url:-<inherit/default>}"
  printf '  llm_model:    %s\n' "${model_name:-<inherit/default>}"
  if [[ -n "${api_key}" ]]; then
    printf '  llm_api_key:  <provided>\n'
  else
    printf '  llm_api_key:  <inherit/default>\n'
  fi

  local cmd=(
    "${PYTHON_BIN}"
    "eval/rejudge_metrics_eval_steps.py"
    "--input-root" "${input_root}"
    "--output-root" "${output_root}"
    "--source-run" "${SOURCE_RUN}"
    "--workers" "${workers_value}"
    "--batch-size" "${batch_size_value}"
    "--save-every" "${SAVE_EVERY}"
  )

  if [[ -n "${MAX_CASES_PER_DATASET}" ]]; then
    cmd+=("--max-cases-per-dataset" "${MAX_CASES_PER_DATASET}")
  fi
  if [[ -n "${DATASET_KEY}" ]]; then
    cmd+=("--dataset-key" "${DATASET_KEY}")
  fi

  printf '  command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  local env_cmd=(env)
  if [[ -n "${base_url}" ]]; then
    env_cmd+=("LLM_BASE_URL=${base_url}")
  fi
  if [[ -n "${model_name}" ]]; then
    env_cmd+=("LLM_MODEL=${model_name}")
  fi
  if [[ -n "${api_key}" ]]; then
    env_cmd+=("LLM_API_KEY=${api_key}")
  fi

  "${env_cmd[@]}" "${cmd[@]}"
}

launch_job_in_slot() {
  local slot_idx="$1"
  local job_idx="$2"
  local endpoint_idx="${SLOT_ENDPOINT_INDEXES[$slot_idx]}"
  local label="${JOB_LABELS[$job_idx]}"
  local input_root="${JOB_INPUT_ROOTS[$job_idx]}"
  local output_root="${JOB_OUTPUT_ROOTS[$job_idx]}"

  run_job "${label}" "${input_root}" "${output_root}" "${endpoint_idx}" "${slot_idx}" &
  local pid=$!

  ACTIVE_PIDS[$slot_idx]="${pid}"
  ACTIVE_LABELS[$slot_idx]="${label}"
  ACTIVE_ENDPOINTS[$slot_idx]="${endpoint_idx}"

  printf '[rejudge_metrics_from_train] launched %s pid=%s slot=%s endpoint=%s\n' \
    "${label}" "${pid}" "${slot_idx}" "${endpoint_idx}"
}

has_active_jobs() {
  local pid=""
  for pid in "${ACTIVE_PIDS[@]}"; do
    if [[ -n "${pid}" ]]; then
      return 0
    fi
  done
  return 1
}

is_pid_active() {
  local pid="$1"
  local state=""

  state="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
  if [[ -z "${state}" ]]; then
    return 1
  fi
  if [[ "${state}" == Z* ]]; then
    return 1
  fi
  return 0
}

collect_finished_jobs() {
  local slot_idx=0
  local pid=""
  local label=""
  local endpoint_idx=""
  local status=0

  for slot_idx in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[$slot_idx]}"
    [[ -z "${pid}" ]] && continue

    if is_pid_active "${pid}"; then
      continue
    fi

    label="${ACTIVE_LABELS[$slot_idx]}"
    endpoint_idx="${ACTIVE_ENDPOINTS[$slot_idx]}"

    if wait "${pid}"; then
      printf '[rejudge_metrics_from_train] finished %s pid=%s slot=%s endpoint=%s\n' \
        "${label}" "${pid}" "${slot_idx}" "${endpoint_idx}"
    else
      status=$?
      printf '[rejudge_metrics_from_train] failed %s pid=%s slot=%s endpoint=%s exit=%s\n' \
        "${label}" "${pid}" "${slot_idx}" "${endpoint_idx}" "${status}" >&2
      failed_jobs=$((failed_jobs + 1))
    fi

    ACTIVE_PIDS[$slot_idx]=""
    ACTIVE_LABELS[$slot_idx]=""
    ACTIVE_ENDPOINTS[$slot_idx]=""
  done
}

print_endpoint_plan() {
  local endpoint_idx=0
  local base_url=""
  local model_name=""
  local api_key=""
  local workers_value=""
  local batch_size_value=""

  printf '[rejudge_metrics_from_train] endpoint_count:    %s\n' "${#BASE_URL_ITEMS[@]}"
  printf '[rejudge_metrics_from_train] per_url_job_limit: %s\n' "${PER_URL_JOB_LIMIT}"
  printf '[rejudge_metrics_from_train] total_slots:       %s\n' "${#SLOT_ENDPOINT_INDEXES[@]}"

  for (( endpoint_idx = 0; endpoint_idx < ${#BASE_URL_ITEMS[@]}; endpoint_idx++ )); do
    base_url="$(pick_base_url "${endpoint_idx}")"
    model_name="$(pick_model "${endpoint_idx}")"
    api_key="$(pick_api_key "${endpoint_idx}")"
    workers_value="$(pick_workers "${endpoint_idx}")"
    batch_size_value="$(pick_batch_size "${endpoint_idx}")"

    printf '[rejudge_metrics_from_train] endpoint[%s] base_url=%s model=%s workers=%s batch_size=%s api_key=%s\n' \
      "${endpoint_idx}" \
      "${base_url:-<inherit/default>}" \
      "${model_name:-<inherit/default>}" \
      "${workers_value}" \
      "${batch_size_value}" \
      "$([[ -n "${api_key}" ]] && printf '<provided>' || printf '<inherit/default>')"
  done
}

validate_positive_int_value() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -le 0 ]]; then
    printf '%s must be a positive integer, got: %s\n' "${name}" "${value}" >&2
    exit 1
  fi
}

validate_positive_int_value "WORKERS" "${WORKERS}"
validate_positive_int_value "BATCH_SIZE" "${BATCH_SIZE}"
validate_positive_int_value "SAVE_EVERY" "${SAVE_EVERY}"
validate_positive_int_value "PER_URL_JOB_LIMIT" "${PER_URL_JOB_LIMIT}"
validate_positive_int_value "POLL_INTERVAL_SECONDS" "${POLL_INTERVAL_SECONDS}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "${PYTHON_BIN}" >&2
  exit 1
fi

if ! command -v ps >/dev/null 2>&1; then
  printf 'ps command not found; cannot monitor background jobs safely.\n' >&2
  exit 1
fi

resolve_endpoint_lists

for item in "${WORKER_ITEMS[@]}"; do
  validate_positive_int_value "WORKERS_PER_URLS item" "${item}"
done

for item in "${BATCH_SIZE_ITEMS[@]}"; do
  validate_positive_int_value "BATCH_SIZE_PER_URLS item" "${item}"
done

build_slots



add_job \
  "sapo_4k_math500" \
  "/seu_share2/home/fenglei/220246350/TriPO/new_closed_form_proj/experiments/continue_sapo_bigmath_processed_curriculum_phase2_4k_grpo_20260410_000454/metrics" \
  "/seu_share2/home/fenglei/220246350/TriPO/new_closed_form_proj/experiments/rejudge/eval_sapo_4k_math500_0410_1005"

print_endpoint_plan
printf '[rejudge_metrics_from_train] job_count:         %s\n' "${#JOB_LABELS[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  job_idx=0
  for job_idx in "${!JOB_LABELS[@]}"; do
    slot_idx=$(( job_idx % ${#SLOT_ENDPOINT_INDEXES[@]} ))
    endpoint_idx="${SLOT_ENDPOINT_INDEXES[$slot_idx]}"
    run_job \
      "${JOB_LABELS[$job_idx]}" \
      "${JOB_INPUT_ROOTS[$job_idx]}" \
      "${JOB_OUTPUT_ROOTS[$job_idx]}" \
      "${endpoint_idx}" \
      "${slot_idx}"
  done
  exit 0
fi

failed_jobs=0
next_job_idx=0

while (( next_job_idx < ${#JOB_LABELS[@]} )); do
  slot_idx=0
  for slot_idx in "${!SLOT_ENDPOINT_INDEXES[@]}"; do
    [[ -n "${ACTIVE_PIDS[$slot_idx]}" ]] && continue
    (( next_job_idx >= ${#JOB_LABELS[@]} )) && break
    launch_job_in_slot "${slot_idx}" "${next_job_idx}"
    next_job_idx=$((next_job_idx + 1))
  done

  if (( next_job_idx < ${#JOB_LABELS[@]} )); then
    sleep "${POLL_INTERVAL_SECONDS}"
    collect_finished_jobs
  fi
done

while has_active_jobs; do
  sleep "${POLL_INTERVAL_SECONDS}"
  collect_finished_jobs
done

if [[ "${failed_jobs}" -gt 0 ]]; then
  exit 1
fi
