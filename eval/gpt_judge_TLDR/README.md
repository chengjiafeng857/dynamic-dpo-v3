# Model-Judge Evaluation (GPT-4 Oracle)

This folder contains a lightweight pipeline to evaluate SFT vs. DPO variants on TLDR using GPT-4 as an external judge. The workflow is driven by `test/gpt_judge_TLDR/config_evaluation.yaml`.

## Quick Start

1) Prepare the evaluation set:
```bash
python test/gpt_judge_TLDR/data_process_tldr.py \
  --config test/gpt_judge_TLDR/config_evaluation.yaml
```

2) Generate summaries from all models:
```bash
python test/gpt_judge_TLDR/generate_summaries.py \
  --config test/gpt_judge_TLDR/config_evaluation.yaml \
  --eval_data test/gpt_judge_TLDR/data/tldr_eval_set.json
```

3) Run GPT-4 oracle comparisons (repeat for each comparison):
```bash
export OPENAI_API_KEY=...

python test/gpt_judge_TLDR/gpt4_oracle.py \
  --config test/gpt_judge_TLDR/config_evaluation.yaml \
  --summaries test/gpt_judge_TLDR/outputs/all_summaries.json \
  --comparison sft_vs_standard_dpo
```

4) Analyze results and generate plots:
```bash
python test/gpt_judge_TLDR/analyze_results.py \
  --config test/gpt_judge_TLDR/config_evaluation.yaml

python test/gpt_judge_TLDR/visualize_results.py \
  --config test/gpt_judge_TLDR/config_evaluation.yaml
```

Optional qualitative sample report:
```bash
python test/gpt_judge_TLDR/qualitative_analysis.py \
  --config test/gpt_judge_TLDR/config_evaluation.yaml
```

## Configuration Notes

Edit `test/gpt_judge_TLDR/config_evaluation.yaml` for:
- Dataset settings (`dataset.*`): name, split, max_samples, and field overrides.
- Model paths (`models.*`): set local or HF model IDs.
- Generation parameters (`generation.*`): max tokens, sampling, batch size, device.
- GPT-4 settings (`gpt4_oracle.*`): model name, retries, prompt template.
- Output directories (`output.*`).

If your model is seq2seq, set:
```yaml
models:
  standard_dpo:
    path: "path/to/standard_dpo"
    model_type: "seq2seq"
```

## Outputs

Generated artifacts are written under:
- `test/gpt_judge_TLDR/data/` for the evaluation dataset
- `test/gpt_judge_TLDR/outputs/` for model summaries
- `test/gpt_judge_TLDR/results/` for GPT-4 judgments and reports
- `test/gpt_judge_TLDR/visualizations/` for charts

These directories are ignored in `.gitignore`.

## Dependencies

- `datasets`, `transformers`, `torch`, `tqdm`, `pyyaml`, `scipy`, `matplotlib`
- `openai` (already included via `alpaca-eval` in this repo)

Make sure `OPENAI_API_KEY` is set before running `gpt4_oracle.py`.
