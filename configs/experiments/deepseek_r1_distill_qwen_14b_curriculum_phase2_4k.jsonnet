// 14B Phase 2 (4K max response length) — continues from a Phase 1 checkpoint.
// The init checkpoint path is supplied at launch via APP_INIT_MODEL.
local base = import 'deepseek_r1_distill_qwen_14b_curriculum_phase1_2k.jsonnet';

base + {
  exp_name: 'deepseek_r1_distill_qwen_14b_curriculum_phase2_4k',
  output_dir: 'experiments/deepseek_r1_distill_qwen_14b_curriculum_phase2_4k',

  inference+: {
    max_tokens: 4096,
  },

  vllm+: {
    max_model_len: 6500,
  },

  train+: {
    max_sequence_length: 6500,
    // 4K responses double activation memory vs 2K; keep micro batch at 1.
    // 1 (per-device) x 4 (train ranks) x 4 (accum) = effective batch 16.
    per_device_train_batch_size: 1,
    gradient_accumulation_steps: 4,
  },

  evaluation+: {
    max_tokens: 4096,
  },

  curriculum+: {
    phase_name: 'phase2_4k',
    stages: [
      { name: 'phase2_easy',   iteration_start:   1, iteration_end: 150,
        sampling_mode: 'uniform_with_replacement',
        group_weights: { 'level_1:all': 0.25, 'level_2:all': 0.20,
                         'level_3:all': 0.30, 'level_4:all': 0.15, 'level_5:all': 0.10 } },
      { name: 'phase2_medium', iteration_start: 151, iteration_end: 300,
        sampling_mode: 'uniform_with_replacement',
        group_weights: { 'level_1:all': 0.10, 'level_2:all': 0.10,
                         'level_3:all': 0.30, 'level_4:all': 0.20, 'level_5:all': 0.30 } },
      { name: 'phase2_hard',   iteration_start: 301, iteration_end: 500,
        sampling_mode: 'uniform_with_replacement',
        group_weights: { 'level_1:all': 0.03, 'level_2:all': 0.02,
                         'level_3:all': 0.15, 'level_4:all': 0.30, 'level_5:all': 0.50 } },
    ],
  },
}
