local deepseek_math_prompt = |||
  <｜begin▁of▁sentence｜><｜User｜>Solve the following math problem efficiently and clearly. Think step by step before answering. Put the final answer at the very end of your response using exactly this format and : Therefore, the final answer is: $\boxed{{your\ answer}}$. I hope it is correct.

  {problem}<｜Assistant｜><think>
|||;

local deepseek_stop = ['<｜User｜>'];

{
  exp_name: 'deepseek_r1_distill_qwen_1_5b_mainline_rl',
  seed: std.parseInt(std.extVar('APP_SEED')),
  output_dir: 'experiments/deepseek_r1_distill_qwen_1_5b_mainline_rl',

  model: {
    actor_name_or_path: './init_model/DeepSeek-R1-Distill-Qwen-1.5B',
    tokenizer_name_or_path: './init_model/DeepSeek-R1-Distill-Qwen-1.5B',
    trust_remote_code: true,
    torch_dtype: 'bfloat16',
    attn_implementation: 'eager',
  },

  data: {
    dataset_name: './data/Big-Math-RL-Verified',
    dataset_split: 'train',
    question_field: 'problem',
    answer_field: 'answer',
    answer_format: 'deepseek_r1',
    question_id_field: null,
    excluded_question_indices: [229398, 192884, 210638],
    max_dataset_size: null,
    question_template: deepseek_math_prompt,
    sample_with_replacement: true,
    shuffle_on_each_iteration: true,
    num_questions_per_iteration: 16,
  },

  inference: {
    rollouts_per_question: 64,
    temperature: 0.6,
    top_p: 0.9,
    max_tokens: 2048,
    stop: deepseek_stop,
    request_timeout_s: 300,
    max_parallel_requests: 64,
  },

  vllm: {
    host: '127.0.0.1',
    port: null,
    gpu_idx: 0,
    swap_space: 12,
    dtype: 'bfloat16',
    trust_remote_code: true,
    gpu_memory_utilization: 0.7,
    max_num_seqs: 256,
    max_model_len: 4096,
    enable_prefix_caching: true,
    disable_sliding_window: false,
    disable_frontend_multiprocessing: false,
    wait_timeout_s: 800,
    log_file: 'vllm_server.log',
  },

  // LLM-as-judge rejudge of rule-graded reward=0 rollouts.
  // Reads provider/base_url/model/api_key from env (LLM_PROVIDER/LLM_BASE_URL/
  // LLM_MODEL/LLM_API_KEY) unless overridden here.
  rejudge: {
    // Enabled by default. Reads provider/base_url/model/api_key from env; if
    // LLM_BASE_URL or LLM_API_KEY is empty it auto-disables at runtime, so this is
    // safe on a node without internet (it just no-ops instead of erroring).
    enabled: true,
    max_concurrency: 32,
    // Judge reads the FULL response and may be a reasoning model, so it needs
    // room to think before emitting the final YES/NO (parser takes the last one).
    max_tokens: 2048,
    timeout_s: 120.0,
    max_retries: 1,
    flip_to_reward: 1.0,
  },

  algorithm: {
    beta: 0.1,
    reward_clip_min: -5.0,
    reward_clip_max: 5.0,
    // TFPO: reliable-prefix threshold m (paper Table 4: 24).
    min_success_count: 24,
    // Failure-frontier thresholds. Paper Table 4 uses k_min=n_min=8; relaxed to
    // k_min=4, n_min=2 so negative (dead-end) edges actually appear in the rollouts.
    frontier_min_success_count: 4,
    frontier_min_visit_count: 2,
    include_frontier_negative_samples: true,
    negative_weight_mode: 'uniform',
    deduplicate_rollouts: true,
    append_eos_to_response: true,
    teacher_prob_floor: 1e-4,
    negative_loss_weight: 0.5,
    negative_prob_clamp_eps: 1e-6,
    // LR-schedule floor: warmup from / decay to lr_min_ratio*peak, keeping lr in
    // [1e-6, 1e-5] with the peak 1e-5 set in `train` below.
    lr_min_ratio: 0.1,
    // GRPO complementary loss on tokens not covered by TFPO.
    grpo_loss_weight: 1.0,
    grpo_variant: 'grpo',
    grpo_skip_after_frontier: false,
    grpo_clip_low: 0.2,
    grpo_clip_high: 0.2,
    grpo_advantage_eps: 1e-6,
    grpo_sapo_alpha: 1.0,
  },

  train: {
    max_sequence_length: 4096,
    num_epochs_per_iteration: 1,
    per_device_train_batch_size: 32,
    gradient_accumulation_steps: 4,
    // Peak LR 1e-5 (paper Table 4 used 1e-6); with lr_min_ratio=0.1 the schedule
    // keeps lr in [1e-6, 1e-5].
    learning_rate: 1e-5,
    weight_decay: 0.0,
    warmup_ratio: 0.03,
    max_grad_norm: 1.0,
    bf16: true,
    logging_steps: 10,
    dataloader_num_workers: 4,
    gradient_checkpointing: true,
  },

  runtime: {
    num_iterations: 1000,
    // Single unified schedule: every `policy_save_interval` iterations we
    // save+keep a checkpoint, sync the rollout vLLM to it, and run evaluation.
    policy_save_interval: 10,
    save_rollouts_every: 20,
    save_total_limit: 10,
  },

  evaluation: {
    enabled: true,
    dataset_name: './eval_data/MATH-500',
    dataset_split: 'test',
    question_field: 'problem',
    answer_field: 'answer',
    answer_format: 'deepseek_r1',
    question_template: deepseek_math_prompt,
    max_samples: null,
    // Paper Table 4 evaluation sampling: (T, p) = (0.2, 0.9).
    temperature: 0.2,
    top_p: 0.9,
    max_tokens: 2048,
    stop: deepseek_stop,
    request_timeout_s: 300,
    max_parallel_requests: 64,
    mode: 'trainer',
    vllm_gpu_ids: null,
    keep_last_checkpoint_only: false,
    interval: null,
    enable_pass_k: false,
    // Enabled by default: re-judge rule-wrong pass@1 with the LLM (correct ones
    // untouched). Auto-disables at runtime if the rejudge LLM env is unset, so it
    // is safe on a node without internet.
    use_llm_judge: true,
  },

  deepspeed: {
    enabled: false,
    config_path: 'configs/shared/deepspeed/zero2.json',
  },
}
