#!/usr/bin/env bash
# Two-phase RL training for DeepSeek-R1-Distill-Qwen-32B (6-train + 2-infer GPUs).
#
# Layout (8xH100): GPU 0,1,2,3,4,5 = HF Trainer (DeepSpeed ZeRO-3 +
# CPU param/optimizer offload); GPU 6,7 = vLLM TP=2.
#
# Env (set before running):
#   export LLM_PROVIDER=anthropic
#   export LLM_BASE_URL=https://api.minimaxi.com/anthropic
#   export LLM_MODEL=MiniMax-M2.7
#   export LLM_API_KEY=...        # required for rejudge
#
# Override-able env:
#   APP_SEED              (default 42)
#   CUDA_VISIBLE_DEVICES  (default 0,1,2,3,4,5,6,7 — all 8 GPUs must be visible)
#   TRAIN_GPUS            (default 0,1,2,3,4,5 — HF/DeepSpeed train ranks)
#                          vLLM GPU ids come from the jsonnet config.
#   PHASE1_CONFIG, PHASE2_CONFIG
#   PHASE1_OUTPUT_PREFIX, PHASE2_OUTPUT_PREFIX

set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export APP_SEED="${APP_SEED:-42}"

# LLM-as-judge ON by default (training rejudge + eval re-judge, enabled in the
# configs). Provider/URL/model are non-secret and defaulted here; export your own
# LLM_API_KEY to actually enable it. Without the key (or on an offline node) the
# judge auto-disables at runtime, so this is safe either way. Override via env.
export LLM_PROVIDER="${LLM_PROVIDER:-anthropic}"
export LLM_BASE_URL="${LLM_BASE_URL:-https://api.minimaxi.com/anthropic}"
export LLM_MODEL="${LLM_MODEL:-MiniMax-M2.7}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5}"
NUM_TRAIN_PROCS="$(awk -F',' '{print NF}' <<< "${TRAIN_GPUS}")"

PHASE1_CONFIG="${PHASE1_CONFIG:-configs/experiments/deepseek_r1_distill_qwen_32b_curriculum_phase1_2k.jsonnet}"
PHASE2_CONFIG="${PHASE2_CONFIG:-configs/experiments/deepseek_r1_distill_qwen_32b_curriculum_phase2_4k.jsonnet}"
PHASE1_OUTPUT_PREFIX="${PHASE1_OUTPUT_PREFIX:-experiments/deepseek_r1_distill_qwen_32b_curriculum_phase1_2k}"
PHASE2_OUTPUT_PREFIX="${PHASE2_OUTPUT_PREFIX:-experiments/deepseek_r1_distill_qwen_32b_curriculum_phase2_4k}"

if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
fi

unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK NODE_RANK MASTER_ADDR MASTER_PORT

if [ -z "${LLM_API_KEY:-}" ]; then
  echo "[warning] LLM_API_KEY is empty — rejudge will be disabled at runtime." >&2
fi

find_latest_run_dir() {
  ls -dt "${1}"_* 2>/dev/null | head -n 1 || true
}
find_latest_actor_ckpt() {
  ls -dt "${1}/checkpoints"/iter_*_actor 2>/dev/null | head -n 1 || true
}

echo "==== DeepSeek-R1-Distill-Qwen-32B two-phase RL ===="
echo "  visible_gpus:   ${CUDA_VISIBLE_DEVICES}"
echo "  train_gpus:     ${TRAIN_GPUS} (num_processes=${NUM_TRAIN_PROCS})"
echo "  phase1_config:  ${PHASE1_CONFIG}"
echo "  phase2_config:  ${PHASE2_CONFIG}"
echo "  llm_provider:   ${LLM_PROVIDER:-<unset>}"
echo "  llm_base_url:   ${LLM_BASE_URL:-<unset>}"
echo "  llm_model:      ${LLM_MODEL:-<unset>}"
echo

echo "[Phase 1] starting (max response 2K, 500 iter)"
# One shared run tag for ALL ranks (else each rank makes its own timestamped
# output dir and the disk-backed teacher-pairs hand-off breaks across ranks).
export APP_RUN_TAG="$(date +%Y%m%d_%H%M%S)"
accelerate launch --num_processes "${NUM_TRAIN_PROCS}" --num_machines 1 --mixed_precision bf16 --dynamo_backend no --multi_gpu \
  --gpu_ids "${TRAIN_GPUS}" \
  run.py --configs "${PHASE1_CONFIG}"

PHASE1_RUN_DIR="$(find_latest_run_dir "${PHASE1_OUTPUT_PREFIX}")"
if [ -z "${PHASE1_RUN_DIR}" ]; then
  echo "Failed to locate Phase 1 run dir for prefix: ${PHASE1_OUTPUT_PREFIX}" >&2
  exit 1
fi
PHASE1_FINAL_CKPT="$(find_latest_actor_ckpt "${PHASE1_RUN_DIR}")"
if [ -z "${PHASE1_FINAL_CKPT}" ]; then
  echo "Failed to locate Phase 1 final actor checkpoint under: ${PHASE1_RUN_DIR}" >&2
  exit 1
fi
echo "[Phase 1] done. final_ckpt=${PHASE1_FINAL_CKPT}"

echo "[Phase 2] starting (max response 4K, 500 iter, init from Phase 1)"
# Fresh shared run tag for phase 2 (distinct dir from phase 1).
export APP_RUN_TAG="$(date +%Y%m%d_%H%M%S)"
APP_ACTOR_NAME_OR_PATH="${PHASE1_FINAL_CKPT}" \
APP_TOKENIZER_NAME_OR_PATH="${PHASE1_FINAL_CKPT}" \
accelerate launch --num_processes "${NUM_TRAIN_PROCS}" --num_machines 1 --mixed_precision bf16 --dynamo_backend no --multi_gpu \
  --gpu_ids "${TRAIN_GPUS}" \
  run.py --configs "${PHASE2_CONFIG}"

PHASE2_RUN_DIR="$(find_latest_run_dir "${PHASE2_OUTPUT_PREFIX}")"
PHASE2_FINAL_CKPT="$(find_latest_actor_ckpt "${PHASE2_RUN_DIR}")"
echo
echo "==== finished ===="
echo "  phase1_run:    ${PHASE1_RUN_DIR}"
echo "  phase1_ckpt:   ${PHASE1_FINAL_CKPT}"
echo "  phase2_run:    ${PHASE2_RUN_DIR}"
echo "  phase2_ckpt:   ${PHASE2_FINAL_CKPT}"
