#!/usr/bin/env bash
# Run the three TFPO experiments (DeepSeek-R1-Distill-Qwen 1.5B / 14B / 32B),
# each a two-phase (2K -> 4K) curriculum RL run, one after another.
#
# They all share the same 4 GPUs, so they MUST run sequentially — this script
# just chains the per-model training scripts in order and stops on first failure.
#
# Env (set before running; required for the LLM-as-judge rejudge step):
#   export LLM_PROVIDER=anthropic
#   export LLM_BASE_URL=https://api.minimaxi.com/anthropic
#   export LLM_MODEL=MiniMax-M2.7
#   export LLM_API_KEY=...        # if empty, rejudge is disabled at runtime
#
# Usage:
#   bash run_all_experiments.sh                # run all three: 1.5b 14b 32b
#   bash run_all_experiments.sh 1.5b           # run only 1.5B
#   bash run_all_experiments.sh 14b 32b        # run 14B then 32B
#
# Other env passthrough (APP_SEED, CUDA_VISIBLE_DEVICES, TRAIN_GPUS, ...) is
# honored by the per-model scripts; export it here and it propagates.

set -euo pipefail

cd "$(dirname "$0")"

declare -A EXPERIMENT_SCRIPTS=(
  ["1.5b"]="train_deepseek_r1_distill_qwen_1.5b.sh"
  ["14b"]="train_deepseek_r1_distill_qwen_14b.sh"
  ["32b"]="train_deepseek_r1_distill_qwen_32b.sh"
)

# Default order if no args are given.
if [ "$#" -eq 0 ]; then
  EXPERIMENTS=("1.5b" "14b" "32b")
else
  EXPERIMENTS=("$@")
fi

# Validate selections up front so we fail fast on a typo.
for name in "${EXPERIMENTS[@]}"; do
  if [ -z "${EXPERIMENT_SCRIPTS[$name]:-}" ]; then
    echo "Unknown experiment '${name}'. Valid options: 1.5b 14b 32b" >&2
    exit 1
  fi
done

if [ -z "${LLM_API_KEY:-}" ]; then
  echo "[warning] LLM_API_KEY is empty — rejudge will be disabled at runtime." >&2
fi

echo "==== TFPO experiment runner ===="
echo "  experiments:  ${EXPERIMENTS[*]}"
echo "  llm_provider: ${LLM_PROVIDER:-<unset>}"
echo "  llm_base_url: ${LLM_BASE_URL:-<unset>}"
echo "  llm_model:    ${LLM_MODEL:-<unset>}"
echo

for name in "${EXPERIMENTS[@]}"; do
  script="${EXPERIMENT_SCRIPTS[$name]}"
  echo "########################################################################"
  echo "# [$(date '+%Y-%m-%d %H:%M:%S')] START experiment: ${name}  (${script})"
  echo "########################################################################"
  bash "./${script}"
  echo "# [$(date '+%Y-%m-%d %H:%M:%S')] DONE experiment: ${name}"
  echo
done

echo "==== all requested experiments finished: ${EXPERIMENTS[*]} ===="
