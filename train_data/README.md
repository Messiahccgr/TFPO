# Training Data

Large dataset files are intentionally not committed.

Restore the datasets used by the training configs:

```bash
bash train_data/download_datasets.sh
```

Download every supported source dataset, including NuminaMath-1.5:

```bash
bash train_data/download_datasets.sh --dataset all
```

Rebuild the local 2k derived datasets after downloading sources:

```bash
TOKENIZER_NAME_OR_PATH=./init_model/DeepSeek-R1-Distill-Qwen-1.5B \
  bash train_data/download_datasets.sh --dataset all --prepare-derived
```

The derived rebuild step needs the project Python dependencies plus a tokenizer
available locally or from Hugging Face.
