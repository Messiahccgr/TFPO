# TFPO Training Project


## 1. Download Models

Model download scripts are in `init_model/`. The training scripts do not
download models automatically, so run the matching script first:

```bash
# 1.5B
bash init_model/download_deepseek_r1_distill_qwen_1.5b.sh

# 14B
bash init_model/download_deepseek_r1_distill_qwen_14b.sh

# 32B
bash init_model/download_deepseek_r1_distill_qwen_32b.sh
```

Each script downloads into `init_model/DeepSeek-R1-Distill-Qwen-*`, which is the
path expected by the configs.

## 2. Download Training Data

Training dataset download scripts are in `train_data/`:

```bash
bash train_data/download_datasets.sh
```

The default command restores the datasets used by the current curriculum
training configs. To download every supported training source and rebuild the
local 2k derived datasets:

```bash
TOKENIZER_NAME_OR_PATH=./init_model/DeepSeek-R1-Distill-Qwen-1.5B \
  bash train_data/download_datasets.sh --dataset all --prepare-derived
```

## 3. Download Evaluation Data

Evaluation dataset download scripts are in `eval_data/`:

```bash
bash eval_data/download_datasets.sh
```

You can also download one evaluation dataset at a time:

```bash
bash eval_data/download_datasets.sh --dataset math500
bash eval_data/download_datasets.sh --dataset gsm8k
bash eval_data/download_datasets.sh --dataset aime24
```

## 4. Run Training

Choose the training script that matches the model size and available GPUs:

```bash
# 1.5B: 1 trainer GPU + 1 vLLM GPU
bash train_deepseek_r1_distill_qwen_1.5b.sh

# 14B: 4 trainer GPUs + 2 vLLM GPUs
bash train_deepseek_r1_distill_qwen_14b.sh

# 32B: 6 trainer GPUs + 2 vLLM GPUs
bash train_deepseek_r1_distill_qwen_32b.sh
```

LLM-as-judge rejudge is enabled in the configs. If `LLM_API_KEY` is not
set, the scripts print a warning and rejudge is disabled at runtime:

```bash
export LLM_PROVIDER=anthropic
export LLM_BASE_URL=xxx
export LLM_MODEL=xxx
export LLM_API_KEY=xxx
```
