local base = import 'deepseek_r1_distill_qwen_1_5b_bigmath_processed_curriculum_phase1_2k.jsonnet';

local phase_cfg = {
  exp_name: 'deepseek_r1_distill_qwen_1_5b_bigmath_processed_curriculum_phase2_4k',
  output_dir: 'experiments/deepseek_r1_distill_qwen_1_5b_bigmath_processed_curriculum_phase2_4k',

  data: {
    num_questions_per_iteration: 16,
    question_field: 'prompt',
    answer_field: 'solution',
  },

  inference: {
    rollouts_per_question: 64,
    max_tokens: 4096,
    request_timeout_s: 300,
    max_parallel_requests: 64,
  },

  vllm: {
    max_model_len: 6500,
    max_num_seqs: 256,
    gpu_memory_utilization: 0.7,
  },

  train: {
    max_sequence_length: 6500,
    per_device_train_batch_size: 16,
    gradient_accumulation_steps: 4,
  },

  runtime: {
    num_iterations: 500,
    policy_save_interval: 10,
    save_total_limit: 10,
  },

  evaluation: {
    interval: 10,
    max_tokens: 4096,
    request_timeout_s: 300,
    max_parallel_requests: 64,
    pass_k_num_samples: 8,
    pass_k_temperature: 0.6,
    pass_k_top_p: 0.9,
  },

  curriculum: {
    phase_name: 'phase2_4k',
    stages: [
      {
        name: 'phase2_easy',
        iteration_start: 1,
        iteration_end: 150,
        sampling_mode: 'uniform_with_replacement',
        group_weights: {
          ['level_1:all']: 0.25,
          ['level_2:all']: 0.20,
          ['level_3:all']: 0.30,
          ['level_4:all']: 0.15,
          ['level_5:all']: 0.10,
        },
      },
      {
        name: 'phase2_medium',
        iteration_start: 151,
        iteration_end: 300,
        sampling_mode: 'uniform_with_replacement',
        group_weights: {
          ['level_1:all']: 0.10,
          ['level_2:all']: 0.10,
          ['level_3:all']: 0.30,
          ['level_4:all']: 0.20,
          ['level_5:all']: 0.30,
        },
      },
      {
        name: 'phase2_hard',
        iteration_start: 301,
        iteration_end: 500,
        sampling_mode: 'uniform_with_replacement',
        group_weights: {
          ['level_1:all']: 0.03,
          ['level_2:all']: 0.02,
          ['level_3:all']: 0.15,
          ['level_4:all']: 0.30,
          ['level_5:all']: 0.50,
        },
      },
    ],
  },
};

base + {
  exp_name: phase_cfg.exp_name,
  output_dir: phase_cfg.output_dir,

  data+: {
    num_questions_per_iteration: phase_cfg.data.num_questions_per_iteration,
    question_field: phase_cfg.data.question_field,
    answer_field: phase_cfg.data.answer_field,
  },

  inference+: {
    rollouts_per_question: phase_cfg.inference.rollouts_per_question,
    max_tokens: phase_cfg.inference.max_tokens,
    request_timeout_s: phase_cfg.inference.request_timeout_s,
    max_parallel_requests: phase_cfg.inference.max_parallel_requests,
  },

  vllm+: {
    max_model_len: phase_cfg.vllm.max_model_len,
    max_num_seqs: phase_cfg.vllm.max_num_seqs,
    gpu_memory_utilization: phase_cfg.vllm.gpu_memory_utilization,
  },

  train+: {
    max_sequence_length: phase_cfg.train.max_sequence_length,
    per_device_train_batch_size: phase_cfg.train.per_device_train_batch_size,
    gradient_accumulation_steps: phase_cfg.train.gradient_accumulation_steps,
  },

  runtime+: {
    num_iterations: phase_cfg.runtime.num_iterations,
    policy_save_interval: phase_cfg.runtime.policy_save_interval,
    save_total_limit: phase_cfg.runtime.save_total_limit,
  },

  evaluation+: {
    interval: phase_cfg.evaluation.interval,
    max_tokens: phase_cfg.evaluation.max_tokens,
    request_timeout_s: phase_cfg.evaluation.request_timeout_s,
    max_parallel_requests: phase_cfg.evaluation.max_parallel_requests,
    pass_k_num_samples: phase_cfg.evaluation.pass_k_num_samples,
    pass_k_temperature: phase_cfg.evaluation.pass_k_temperature,
    pass_k_top_p: phase_cfg.evaluation.pass_k_top_p,
  },

  curriculum+: {
    phase_name: phase_cfg.curriculum.phase_name,
    stages: phase_cfg.curriculum.stages,
  },
}
