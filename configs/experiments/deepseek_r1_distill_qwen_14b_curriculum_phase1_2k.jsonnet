// 14B Phase 1 (2K max response length) — easy-to-hard curriculum, 500 iter.
local base = import 'deepseek_r1_distill_qwen_14b_mainline_rl.jsonnet';
local processed_root = './train_data/Big-Math-RL-Verified-Processed';

local phase_cfg = {
  exp_name: 'deepseek_r1_distill_qwen_14b_curriculum_phase1_2k',
  output_dir: 'experiments/deepseek_r1_distill_qwen_14b_curriculum_phase1_2k',

  curriculum: {
    phase_name: 'phase1_2k',
    sources: {
      level_1: { dataset_name: processed_root + '/level_1', dataset_split: 'train',
                 grouping: { type: 'all', group_name: 'all' } },
      level_2: { dataset_name: processed_root + '/level_2', dataset_split: 'train',
                 grouping: { type: 'all', group_name: 'all' } },
      level_3: { dataset_name: processed_root + '/level_3', dataset_split: 'train',
                 grouping: { type: 'all', group_name: 'all' } },
      level_4: { dataset_name: processed_root + '/level_4', dataset_split: 'train',
                 grouping: { type: 'all', group_name: 'all' } },
      level_5: { dataset_name: processed_root + '/level_5', dataset_split: 'train',
                 grouping: { type: 'all', group_name: 'all' } },
    },
    stages: [
      { name: 'phase1_easy',   iteration_start:   1, iteration_end: 150,
        sampling_mode: 'uniform_with_replacement',
        group_weights: { 'level_1:all': 0.40, 'level_2:all': 0.35,
                         'level_3:all': 0.15, 'level_4:all': 0.05, 'level_5:all': 0.05 } },
      { name: 'phase1_medium', iteration_start: 151, iteration_end: 300,
        sampling_mode: 'uniform_with_replacement',
        group_weights: { 'level_1:all': 0.20, 'level_2:all': 0.20,
                         'level_3:all': 0.35, 'level_4:all': 0.15, 'level_5:all': 0.10 } },
      { name: 'phase1_hard',   iteration_start: 301, iteration_end: 500,
        sampling_mode: 'uniform_with_replacement',
        group_weights: { 'level_1:all': 0.05, 'level_2:all': 0.05,
                         'level_3:all': 0.20, 'level_4:all': 0.30, 'level_5:all': 0.40 } },
    ],
  },
};

base + {
  exp_name: phase_cfg.exp_name,
  output_dir: phase_cfg.output_dir,

  data+: {
    dataset_name: processed_root,
    excluded_question_indices: [],
    question_field: 'prompt',
    answer_field: 'solution',
    max_dataset_size: null,
    num_questions_per_iteration: 16,
  },

  inference+: {
    rollouts_per_question: 64,
    max_tokens: 2048,
    request_timeout_s: 300,
    max_parallel_requests: 64,
  },

  vllm+: {
    max_model_len: 4096,
  },

  train+: {
    max_sequence_length: 4096,
  },

  runtime+: {
    num_iterations: 500,
    // Unified schedule: save + rollout-sync + eval every 10 iterations.
    policy_save_interval: 10,
    save_total_limit: 10,
  },

  evaluation+: {
    interval: 10,
    max_tokens: 2048,
    request_timeout_s: 300,
    max_parallel_requests: 64,
  },

  curriculum: {
    enabled: true,
    phase_name: phase_cfg.curriculum.phase_name,
    sources: phase_cfg.curriculum.sources,
    stages: phase_cfg.curriculum.stages,
  },
}
