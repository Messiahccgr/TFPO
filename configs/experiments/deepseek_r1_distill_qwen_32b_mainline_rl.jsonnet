// 32B mainline config: inherits the 14B mainline (paper Table 4 hparams already
// baked in via the 1.5B base) and overrides only what the bigger model requires.
// Same 2-train / 2-infer GPU layout as 14B (train GPU 0,1; vLLM TP=2 on GPU 2,3),
// but with ZeRO-3 parameter offload to CPU so the 32B fits on 2xH100.
local base = import 'deepseek_r1_distill_qwen_14b_mainline_rl.jsonnet';

base + {
  exp_name: 'deepseek_r1_distill_qwen_32b_mainline_rl',
  output_dir: 'experiments/deepseek_r1_distill_qwen_32b_mainline_rl',

  model+: {
    actor_name_or_path: './init_model/DeepSeek-R1-Distill-Qwen-32B',
    tokenizer_name_or_path: './init_model/DeepSeek-R1-Distill-Qwen-32B',
  },

  vllm+: {
    // 2-GPU tensor parallel for the 32B inference server (GPU 2,3).
    num_inference_gpus: 2,
    inference_gpu_ids: [2, 3],
    // 32B bf16 weights ~32GB/GPU at TP=2; leave headroom for KV cache.
    gpu_memory_utilization: 0.8,
    max_model_len: 6500,
    max_num_seqs: 64,
  },

  train+: {
    per_device_train_batch_size: 1,
    // 1 (per-device) x 2 (train ranks) x 8 (accum) = effective batch 16.
    gradient_accumulation_steps: 8,
    gradient_checkpointing: true,
  },

  deepspeed+: {
    enabled: true,
    config_path: 'configs/shared/deepspeed/zero3_32b.json',
  },
}
