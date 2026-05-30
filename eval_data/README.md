# Evaluation Data

Large evaluation dataset files are intentionally not committed.

Restore the local evaluation datasets:

```bash
bash eval_data/download_datasets.sh
```

You can also download one dataset at a time:

```bash
bash eval_data/download_datasets.sh --dataset math500
bash eval_data/download_datasets.sh --dataset gsm8k
bash eval_data/download_datasets.sh --dataset aime24
```
