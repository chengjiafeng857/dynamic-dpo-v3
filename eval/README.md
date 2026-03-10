# Eval Pipeline

This folder contains the repository's evaluation pipeline. Right now it is focused on AlpacaEval and provides two entrypoints:

- `alpacaeval-infer`: run local generation over the AlpacaEval prompts and save model outputs.
- `alpacaeval-eval`: score saved outputs with `alpaca-eval`, or ask `alpaca-eval` to generate from a model config directly.

## Folder layout

- `alpacaeval/config_alpacaeval.yaml`: default config used by both CLI commands.
- `alpacaeval/alpacaeval_infer.py`: inference pipeline.
- `alpacaeval/alpacaeval_eval.py`: evaluation wrapper around `alpaca-eval`.
- `alpacaeval/alpacaeval_common.py`: shared config, template, path, and JSON helpers.
- `alpacaeval/templates/`: prompt templates used when `use_custom_chat_template: true`.
- `alpacaeval/configs/`: example AlpacaEval model-config files.

## How the pipeline works

The default workflow is two-stage:

1. `alpacaeval-infer` loads `tatsu-lab/alpaca_eval`, renders prompts, runs generation with either `transformers` or `vllm`, and writes `model_outputs.json`.
2. `alpacaeval-eval` invokes `alpaca-eval` on those saved outputs and writes a `results/` directory.

There is also a one-step evaluation mode:

- Set `alpacaeval.evaluation_mode: model_configs` or pass `--use-model-configs`.
- In that mode, the repo writes `alpacaeval_model_config.yaml` and asks `alpaca-eval` to generate during evaluation.

## Prerequisites

- Python environment managed with `uv`.
- Project dependencies installed.
- If you want the eval-specific uv dependency group, install it with `uv sync --group eval`.
- Access to the target model in `policy_name` or `alpacaeval.model_name_or_path`.
- Access to the AlpacaEval dataset from Hugging Face.
- If you use `alpacaeval.backend: vllm`, install `vllm` separately. It is optional and not part of the base dependency list.

## Recommended usage

Run inference first:

```bash
uv run alpacaeval-infer --config eval/alpacaeval/config_alpacaeval.yaml
```

Then run evaluation on the saved outputs:

```bash
uv run alpacaeval-eval --config eval/alpacaeval/config_alpacaeval.yaml
```

If you already have a different outputs file, point evaluation at it directly:

```bash
uv run alpacaeval-eval \
  --config eval/alpacaeval/config_alpacaeval.yaml \
  --model-outputs /absolute/path/to/model_outputs.json
```

If you want AlpacaEval to generate directly from a model config instead of using saved outputs:

```bash
uv run alpacaeval-eval \
  --config eval/alpacaeval/config_alpacaeval.yaml \
  --use-model-configs
```

## Llama 3 and chat templating

For Llama 3, the important rule is to apply chat templating exactly once.

Inference in this repo has two mutually exclusive prompt paths:

- `alpacaeval.use_custom_chat_template: true`: the repo formats prompts with [llama3.txt](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/eval/alpacaeval/templates/llama3.txt).
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

Use the custom template path only when the checkpoint expects the repo's prompt format or when you are using model-config evaluation:

- `evaluation_mode: outputs` plus `use_custom_chat_template: false` is the safest no-double-template setup for standard Llama 3 instruct checkpoints.
- `evaluation_mode: model_configs` requires `use_custom_chat_template: true` in this repo, so use [llama3.txt](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/eval/alpacaeval/templates/llama3.txt) there.

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

The pipeline writes into `alpacaeval.output_dir`. With the default config that resolves to `outputs/alpacaeval/llama-3-instruct-8b-simpo` relative to the repo root.

Expected artifacts:

- `model_outputs.json`: one row per AlpacaEval example with the model response in `output`.
- `metadata.json`: run metadata including backend, prompt template source, dataset info, generation config, and package versions.
- `results/`: evaluation outputs produced by `alpaca-eval`.
- `alpacaeval_model_config.yaml`: only written when using model-config evaluation.

Relative paths in the config are resolved relative to the config file, not the current shell directory.

## Important constraints

- `alpacaeval.backend` only supports `transformers` and `vllm`.
- `--use-model-configs` requires `use_custom_chat_template: true`. The code rejects `use_custom_chat_template: false` in that mode.
- If `alpacaeval-eval` cannot find `model_outputs.json`, it fails and asks you to run inference first or pass `--model-outputs`.
- If `transformers.device` requests CUDA and CUDA is unavailable, inference fails fast.
- If the tokenizer does not expose a built-in chat template, set `use_custom_chat_template: true` and provide `prompt_template`.
- Do not pre-format Llama 3 prompts outside this pipeline and also set `use_custom_chat_template: true`, or you will effectively apply a chat template twice.

## Useful files to inspect

- [src/cli.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/src/cli.py)
- [eval/alpacaeval/alpacaeval_infer.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/eval/alpacaeval/alpacaeval_infer.py)
- [eval/alpacaeval/alpacaeval_eval.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/eval/alpacaeval/alpacaeval_eval.py)
- [test/test_alpacaeval_pipeline.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/test/test_alpacaeval_pipeline.py)
