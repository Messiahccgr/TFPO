
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_ROOT="${PROJECT_ROOT}/experiments/eval_gspo_2k_math500_0408_1052/deepseek_r1_distill_qwen_1_5b_gspo_bigmath_processed_curriculum_phase1_2k_grpo_20260405_082934"
OUTPUT_ROOT="${PROJECT_ROOT}/experiments/rejudge/eval_gspo_2k_math500_0409_1435"

export LLM_BASE_URL="http://172.22.2.242:3010/v1"
export LLM_MODEL="qwen3.5-397b-a17b"
export LLM_API_KEY="sk-0nW7ADQHH3SAJ6kp5PBrjTPxqU31NFX9N0GvxLG32JJ1wJMG"


bash rejudge_ckpts.sh \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --dataset math_500_test \
  --workers 128 \
  --batch-size 32 \
  --source-run worst \
  --save-every 10
