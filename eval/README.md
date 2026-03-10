# Eval Pipeline

This repo keeps the AlpacaEval pipeline under `eval/alpacaeval/`.

## Layout

- `eval/alpacaeval/alpacaeval_infer.py`: model inference over the AlpacaEval dataset.
- `eval/alpacaeval/alpacaeval_eval.py`: wrapper around `alpaca_eval`.
- `eval/alpacaeval/batch_runner.py`: config-driven batch runner for multiple models.
- `eval/alpacaeval/config_alpacaeval.yaml`: single-model example config.
- `eval/alpacaeval/config_alpacaeval_batch.yaml`: batch config for the repo's Qwen3/Llama3 models.
- `eval/alpacaeval/templates/`: custom prompt templates copied for SimPO-compatible runs.
- `eval/alpacaeval/configs/`: reference AlpacaEval model-config YAMLs from SimPO.

## Entry Points

The installed commands are:

- `alpacaeval-infer`
- `alpacaeval-eval`
- `alpacaeval-batch`

Typical usage:

```bash
uv run alpacaeval-infer --config eval/alpacaeval/config_alpacaeval.yaml
uv run alpacaeval-eval --config eval/alpacaeval/config_alpacaeval.yaml
uv run alpacaeval-batch --config eval/alpacaeval/config_alpacaeval_batch.yaml
```

## Single-Model Flow

1. `alpacaeval-infer` loads the AlpacaEval eval split, renders prompts, runs generation, and writes:
   - `model_outputs.json`
   - `metadata.json`
2. `alpacaeval-eval` runs `alpaca_eval` on `model_outputs.json` and writes `results/`.

Default outputs live under `outputs/alpacaeval/<pretty_name>/` unless overridden.

## Chat Templates

`alpacaeval.use_custom_chat_template` controls prompt rendering.

- `true`: use a repo template from `eval/alpacaeval/templates/`.
- `false`: use the model tokenizer's own chat template with `apply_chat_template(...)`.

Important behavior:

- For model-default chat templates, the pipeline tokenizes the rendered prompt with `add_special_tokens=false`.
  This avoids the Llama3 double-BOS issue when the tokenizer template already emits `<|begin_of_text|>`.
- For custom templates, the pipeline keeps `add_special_tokens=true`.

Llama3 notes:

- If you use a custom Llama3 template, prefer `templates/llama3-nobos.txt` when the tokenizer or backend adds BOS automatically.
- `templates/llama3.txt` includes `<|begin_of_text|>`.
- `templates/llama3-nobos.txt` omits it.

## Batch Config Defaults

`eval/alpacaeval/config_alpacaeval_batch.yaml` is set up for the repo's trained models.

- Backend defaults to `transformers`.
  This is the safe default for Windows CUDA environments where `vllm` may not work.
- Qwen3 models use the tokenizer default chat template and `stop_token_ids: [151645]`.
- Llama3 models use the custom `llama3-nobos.txt` template and `stop_token_ids: [128001, 128009]`.

## `evaluate_from_model` Caveat

`alpacaeval-eval --use-model-configs` is only supported when `use_custom_chat_template=true`.

Reason: AlpacaEval model-config mode can represent file-based prompt templates, but it cannot faithfully encode arbitrary tokenizer-native chat templates.

If you use `use_custom_chat_template=false`, run:

```bash
uv run alpacaeval-infer --config eval/alpacaeval/config_alpacaeval.yaml
uv run alpacaeval-eval --config eval/alpacaeval/config_alpacaeval.yaml
```

## Package and Dataset Caveats

- SimPO-compatible evaluation expects `alpaca-eval==0.6.2`.
- Recent `datasets` versions reject the legacy `tatsu-lab/alpaca_eval` script loader.
  The real-data tests in this repo fetch `alpaca_eval.json` from the dataset repo directly instead.

## Tests

Relevant tests live in:

- `test/test_alpacaeval_pipeline.py`
- `test/test_alpacaeval_batch_runner.py`

The pipeline tests cover:

- single-model inference and evaluation wiring
- batch-runner config expansion
- Llama3/Qwen3 chat-template handling
- real 3-sample AlpacaEval smoke tests with real tokenizers and stubbed model generation
