# Eval Pipeline

This repo keeps the evaluation pipeline under `eval/alpacaeval/`. The current focus is AlpacaEval, with three installed entrypoints:

- `alpacaeval-infer`: run local generation over the AlpacaEval prompts and save model outputs.
- `alpacaeval-eval`: score saved outputs with `alpaca-eval`, or ask `alpaca-eval` to generate from a model config directly.
- `alpacaeval-batch`: run the same pipeline over a list of models from a batch config.

## Folder layout

- `eval/alpacaeval/alpacaeval_infer.py`: single-model inference pipeline.
- `eval/alpacaeval/alpacaeval_eval.py`: evaluation wrapper around `alpaca-eval`.
- `eval/alpacaeval/batch_runner.py`: config-driven batch runner for multiple models.
- `eval/alpacaeval/alpacaeval_common.py`: shared config, path, template, and JSON helpers.
- `eval/alpacaeval/config_alpacaeval.yaml`: single-model example config.
- `eval/alpacaeval/config_alpacaeval_batch.yaml`: batch config for the repo's Qwen3/Llama3 models.
- `eval/alpacaeval/templates/`: custom prompt templates used when `use_custom_chat_template: true`.
- `eval/alpacaeval/configs/`: reference AlpacaEval model-config YAMLs from SimPO.

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

1. `alpacaeval-infer` loads `tatsu-lab/alpaca_eval`, renders prompts, runs generation with either `transformers` or `vllm`, and writes `model_outputs.json` plus `metadata.json`.
2. `alpacaeval-eval` invokes `alpaca-eval` on those saved outputs and writes a `results/` directory.

There is also a model-config path:

- Set `alpacaeval.evaluation_mode: model_configs` or pass `--use-model-configs`.
- In that mode, the repo writes `alpacaeval_model_config.yaml` and asks `alpaca-eval` to generate during evaluation.

## Prerequisites

- Python environment managed with `uv`.
- Project dependencies installed.
- For the eval-only stack, install `uv sync --group alpaca-eval-0.6.2`.
- For AlpacaEval with `vllm`, install `uv sync --group eval-vllm`.
- Access to the target model in `policy_name` or `alpacaeval.model_name_or_path`.
- Access to the AlpacaEval dataset from Hugging Face.
- The default annotator config is OpenAI-backed, so `alpacaeval-eval` typically needs `OPENAI_API_KEY`.

## Llama 3 and chat templating

For Llama 3, the important rule is to apply chat templating exactly once.

Inference has two mutually exclusive prompt paths:

- `alpacaeval.use_custom_chat_template: true`: the repo formats prompts with a file template such as `templates/llama3.txt` or `templates/llama3-nobos.txt`.
- `alpacaeval.use_custom_chat_template: false`: the repo calls the tokenizer's built-in `apply_chat_template(...)`.

If you are evaluating a standard Llama 3 instruct model and want to avoid double templating, use the tokenizer path:

```yaml
alpacaeval:
  model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
  backend: transformers
  evaluation_mode: outputs
  use_custom_chat_template: false
```

Then run the normal two-step flow:

```bash
uv run alpacaeval-infer --config eval/alpacaeval/config_alpacaeval.yaml
uv run alpacaeval-eval --config eval/alpacaeval/config_alpacaeval.yaml
```

Use the custom template path only when the checkpoint expects the repo's prompt format or when you are using model-config evaluation.

Llama 3 template notes:

- `templates/llama3.txt` includes `<|begin_of_text|>`.
- `templates/llama3-nobos.txt` omits it.
- If your backend or tokenizer already injects BOS, prefer `llama3-nobos.txt`.

## Batch config defaults

`eval/alpacaeval/config_alpacaeval_batch.yaml` is set up for the repo's trained models.

- Backend defaults to `transformers`.
- Qwen3 models use the tokenizer default chat template and `stop_token_ids: [151645]`.
- Llama3 models use the custom `llama3-nobos.txt` template and `stop_token_ids: [128001, 128009]`.
- `skip_existing: true` avoids rerunning inference or eval if outputs already exist.

## Key config fields

The pipeline reads the `alpacaeval` block in [config_alpacaeval.yaml](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/eval/alpacaeval/config_alpacaeval.yaml).

- `model_name_or_path`: model to load. Falls back to top-level `policy_name` if omitted.
- `pretty_name`: label written into the AlpacaEval payload as `generator`.
- `backend`: must be `transformers` or `vllm`.
- `output_dir`: where outputs, metadata, and results are written.
- `dataset_name`, `dataset_config`, `dataset_split`: dataset source, defaulting to `tatsu-lab/alpaca_eval`, `alpaca_eval`, `eval`.
- `annotators_config`: AlpacaEval annotator setting passed through to `alpaca-eval`.
- `evaluation_mode`: `outputs` or `model_configs`.
- `use_custom_chat_template`: when `true`, prompts come from `prompt_template`; when `false`, the tokenizer's built-in chat template is used.
- `prompt_template`: required when `use_custom_chat_template: true`.
- `generation`: generation settings such as `batch_size`, `max_new_tokens`, `temperature`, `top_p`, and `stop_token_ids`.
- `transformers`: backend-specific settings like `device`, `device_map`, and `trust_remote_code`.
- `vllm`: backend-specific settings like `tensor_parallel_size` and `gpu_memory_utilization`.
- `simpo_compat`: enforces `alpaca-eval==0.6.2` during evaluation.

## Outputs

The pipeline writes into `alpacaeval.output_dir`. In the current single-model config, that resolves under `eval/alpacaeval/outputs/alpacaeval/llama-3-instruct-8b-simpo`.

Expected artifacts:

- `model_outputs.json`: one row per AlpacaEval example with the model response in `output`.
- `metadata.json`: run metadata including backend, prompt template source, dataset info, generation config, and package versions.
- `results/`: evaluation outputs produced by `alpaca-eval`.
- `alpacaeval_model_config.yaml`: only written when using model-config evaluation.

Relative paths in the config are resolved relative to the config file, not the current shell directory.

## Important constraints

- `alpacaeval.backend` only supports `transformers` and `vllm`.
- `alpacaeval-eval --use-model-configs` is only supported when `use_custom_chat_template: true`.
- If `use_custom_chat_template: false`, use the normal infer-then-eval flow instead of model-config evaluation.
- If `alpacaeval-eval` cannot find `model_outputs.json`, it fails and asks you to run inference first or pass `--model-outputs`.
- If `transformers.device` requests CUDA and CUDA is unavailable, inference fails fast.
- If the tokenizer does not expose a built-in chat template, set `use_custom_chat_template: true` and provide `prompt_template`.
- Do not pre-format Llama 3 prompts outside this pipeline and also set `use_custom_chat_template: true`, or you will effectively apply a chat template twice.

## Package and dataset caveats

- SimPO-compatible evaluation expects `alpaca-eval==0.6.2`.
- The default annotator config is `weighted_alpaca_eval_gpt4_turbo`, which means the scoring step uses an OpenAI-backed judge.
- Recent `datasets` versions reject the legacy `tatsu-lab/alpaca_eval` script loader in some contexts. The real-data tests in this repo fetch `alpaca_eval.json` from the dataset repo directly instead.

## Tests

Relevant tests live in:

- [test/test_alpacaeval_pipeline.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/test/test_alpacaeval_pipeline.py)
- [test/test_alpacaeval_batch_runner.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/test/test_alpacaeval_batch_runner.py)

The pipeline tests cover:

- single-model inference and evaluation wiring
- batch-runner config expansion
- Llama3/Qwen3 chat-template handling
- real 3-sample AlpacaEval smoke tests with real tokenizers and stubbed model generation

## Useful files to inspect

- [src/cli.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/src/cli.py)
- [eval/alpacaeval/alpacaeval_infer.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/eval/alpacaeval/alpacaeval_infer.py)
- [eval/alpacaeval/alpacaeval_eval.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/eval/alpacaeval/alpacaeval_eval.py)
- [eval/alpacaeval/batch_runner.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/eval/alpacaeval/batch_runner.py)
