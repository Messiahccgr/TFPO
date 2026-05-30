// 14B mainline config: inherits the 1.5B mainline (paper Table 4 hparams already
// baked in) and overrides only what the bigger model + 4-train/2-infer GPU
// layout requires.
//
// Layout (>=6 GPUs): train on GPU 0,1,2,3 (ZeRO-3, NO offload); vLLM TP=2 on
// GPU 4,5. With 4 ZeRO-3 ranks the full Adam state (~224GB) shards to ~56GB/GPU,
// so it fits in 80GB without CPU offload (avoids host-RAM OOM + cpu_adam build).
local base = import 'deepseek_r1_distill_qwen_1_5b_mainline_rl.jsonnet';

base + {
  exp_name: 'deepseek_r1_distill_qwen_14b_mainline_rl',
  output_dir: 'experiments/deepseek_r1_distill_qwen_14b_mainline_rl',

  model+: {
    actor_name_or_path: './init_model/DeepSeek-R1-Distill-Qwen-14B',
    tokenizer_name_or_path: './init_model/DeepSeek-R1-Distill-Qwen-14B',
  },

  vllm+: {
    // 2-GPU tensor parallel for the 14B inference server on GPU 4,5
    // (training uses GPU 0,1,2,3).
    num_inference_gpus: 2,
    inference_gpu_ids: [4, 5],
    gpu_memory_utilization: 0.8,
    max_model_len: 6500,
    max_num_seqs: 128,
  },

  train+: {
    per_device_train_batch_size: 1,
    // 1 (per-device) x 4 (train ranks) x 4 (accum) = effective batch 16.
    gradient_accumulation_steps: 4,
    gradient_checkpointing: true,
    // Peak LR 1e-5 (was 1e-6); the schedule warms up from and decays to
    // lr_min_ratio*peak = 1e-6, so lr stays in [1e-6, 1e-5].
    learning_rate: 1e-5,
  },

  algorithm+: {
    // Keep lr in [1e-6, 1e-5] instead of warming up from ~0.
    lr_min_ratio: 0.1,
    // Relax the failure-frontier thresholds (paper Table 4 uses k_min=n_min=8) so
    // negative edges actually appear: a dead-end next-token now only needs >=2
    // visits from a prefix with >=4 successes.
    frontier_min_success_count: 4,
    frontier_min_visit_count: 2,
  },

  deepspeed+: {
    enabled: true,
    config_path: 'configs/shared/deepspeed/zero3_14b_nooffload.json',
  },
}
