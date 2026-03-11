# Eval Pipeline

This repo keeps the evaluation pipeline under `eval/alpacaeval/`. The current
focus is AlpacaEval, with three installed entrypoints:

- `alpacaeval-infer`: run local generation over the AlpacaEval prompts and save
  model outputs.
- `alpacaeval-eval`: score saved outputs with `alpaca-eval`, or ask
  `alpaca-eval` to generate from a model config directly.
- `alpacaeval-batch`: run the same pipeline over a list of models from a batch
  config.

## Folder layout

- `eval/alpacaeval/alpacaeval_infer.py`: single-model inference pipeline.
- `eval/alpacaeval/alpacaeval_eval.py`: evaluation wrapper around
  `alpaca-eval`.
- `eval/alpacaeval/batch_runner.py`: config-driven batch runner for multiple
  models.
- `eval/alpacaeval/alpacaeval_common.py`: shared config, path, template, and
  JSON helpers.
- `eval/alpacaeval/config_alpacaeval.yaml`: single-model example config.
- `eval/alpacaeval/config_alpacaeval_batch.yaml`: batch config for the repo's
  UltraFeedback Qwen3 and Llama3 models.
- `eval/alpacaeval/templates/`: custom prompt templates used when
  `use_custom_chat_template: true`.
- `eval/alpacaeval/configs/`: checked-in AlpacaEval model-config YAMLs for the
  repo's Qwen3/Llama3 UltraFeedback and UltraChat checkpoints.

## Recommended usage

Single-model flow:

```bash
uv run alpacaeval-infer --config eval/alpacaeval/config_alpacaeval.yaml
uv run alpacaeval-eval --config eval/alpacaeval/config_alpacaeval.yaml
```

Batch flow:

```bash
uv run alpacaeval-batch --config eval/alpacaeval/config_alpacaeval_batch.yaml
```

Useful variants:

```bash
uv run alpacaeval-eval \
  --config eval/alpacaeval/config_alpacaeval.yaml \
  --model-outputs /absolute/path/to/model_outputs.json
```

```bash
uv run alpacaeval-eval \
  --config eval/alpacaeval/config_alpacaeval.yaml \
  --use-model-configs
```

```bash
uv run alpacaeval-batch --config eval/alpacaeval/config_alpacaeval_batch.yaml --inference-only
uv run alpacaeval-batch --config eval/alpacaeval/config_alpacaeval_batch.yaml --eval-only
uv run alpacaeval-batch --config eval/alpacaeval/config_alpacaeval_batch.yaml --use-model-configs
```

## How the pipeline works

The default single-model workflow is two-stage:

1. `alpacaeval-infer` loads `tatsu-lab/alpaca_eval`, renders prompts, runs
   generation with either `transformers` or `vllm`, and writes
   `model_outputs.json` plus `metadata.json`.
2. `alpacaeval-eval` invokes `alpaca-eval` on those saved outputs and writes a
   `results/` directory.

There is also a model-config path:

- Set `alpacaeval.evaluation_mode: model_configs` or pass
  `--use-model-configs`.
- In that mode, the repo writes `alpacaeval_model_config.yaml` and asks
  `alpaca-eval` to generate during evaluation.

## Prerequisites

- Python environment managed with `uv`.
- Project dependencies installed.
- If you want the eval-specific uv dependency group, install it with
  `uv sync --group eval`.
- Access to the target model in `policy_name` or
  `alpacaeval.model_name_or_path`.
- Access to the AlpacaEval dataset from Hugging Face.
- The repo now declares `vllm` in both the main dependency set and the `eval`
  dependency group for Linux environments.
- The default annotator config is OpenAI-backed, so `alpacaeval-eval`
  typically needs `OPENAI_API_KEY`.

## Llama 3 and chat templating

For Llama 3, apply chat templating exactly once.

Inference has two mutually exclusive prompt paths:

- `alpacaeval.use_custom_chat_template: true`: the repo formats prompts with a
  model-specific file template in `eval/alpacaeval/templates/`.
- `alpacaeval.use_custom_chat_template: false`: the repo calls the tokenizer's
  built-in `apply_chat_template(...)`.

If prompts are built with `apply_chat_template(..., tokenize=False)`, the
follow-up tokenizer call must use `add_special_tokens=False`. This avoids the
Llama 3 double-BOS problem that happens when a rendered chat prompt is treated
like plain text and special tokens are added again.

Use the custom template path only when the checkpoint expects the repo's prompt
format or when you are using model-config evaluation.

Llama 3 template notes:

- the checked-in Llama templates omit BOS
- batch defaults select a model-specific Llama template file
- if you want model-default chat templating instead, set
  `use_custom_chat_template: false`

## Batch config defaults

`eval/alpacaeval/config_alpacaeval_batch.yaml` is set up for the repo's eight
Qwen3/Llama3 UltraFeedback and UltraChat checkpoints.

- Backend defaults to `vllm`.
- Qwen3 models use the tokenizer default chat template and
  `stop_token_ids: [151645]`.
- Llama3 models use a model-specific custom template file and
  `stop_token_ids: [128001, 128009]`.
- `skip_existing: true` avoids rerunning inference or eval if outputs already
  exist.

## Key config fields

The pipeline reads the `alpacaeval` block in
`eval/alpacaeval/config_alpacaeval.yaml`.

- `model_name_or_path`: model to load. Falls back to top-level `policy_name` if
  omitted.
- `pretty_name`: label written into the AlpacaEval payload as `generator`.
- `backend`: must be `transformers` or `vllm`.
- `output_dir`: where outputs, metadata, and results are written.
- `dataset_name`, `dataset_config`, `dataset_split`: dataset source, defaulting
  to `tatsu-lab/alpaca_eval`, `alpaca_eval`, `eval`.
- `annotators_config`: AlpacaEval annotator setting passed through to
  `alpaca-eval`.
- `evaluation_mode`: `outputs` or `model_configs`.
- `use_custom_chat_template`: when `true`, prompts come from `prompt_template`;
  when `false`, the tokenizer's built-in chat template is used.
- `prompt_template`: required when `use_custom_chat_template: true`.
- `generation`: generation settings such as `batch_size`, `max_new_tokens`,
  `temperature`, `top_p`, and `stop_token_ids`.
- `transformers`: backend-specific settings like `device`, `device_map`, and
  `trust_remote_code`.
- `vllm`: backend-specific settings like `tensor_parallel_size` and
  `gpu_memory_utilization`.
- `simpo_compat`: enforces `alpaca-eval==0.6.2` during evaluation.

## Outputs

The pipeline writes into `alpacaeval.output_dir`.

Expected artifacts:

- `model_outputs.json`: one row per AlpacaEval example with the model response
  in `output`.
- `metadata.json`: run metadata including backend, prompt template source,
  dataset info, generation config, and package versions.
- `results/`: evaluation outputs produced by `alpaca-eval`.
- `alpacaeval_model_config.yaml`: only written when using model-config
  evaluation.

Relative paths in the config are resolved relative to the config file, not the
current shell directory.

## Tests

Relevant tests live in:

- `test/test_alpacaeval_pipeline.py`
- `test/test_alpacaeval_batch_runner.py`

The pipeline tests cover:

- single-model inference and evaluation wiring
- batch-runner config expansion
- Llama3/Qwen3 chat-template handling
- real 3-sample AlpacaEval smoke tests with real tokenizers and stubbed model
  generation
