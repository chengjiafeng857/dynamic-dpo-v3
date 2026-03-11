# Eval Pipeline

This repo keeps benchmark wrappers under `eval/`:

- `eval/alpacaeval/`: AlpacaEval local generation plus `alpaca-eval` scoring.
- `eval/arenahard/`: Arena-Hard local generation plus external judge wrapper.
- `eval/mtbench/`: MT-Bench local generation plus FastChat judge wrapper.

Installed entrypoints:

- `alpacaeval-infer`, `alpacaeval-eval`, `alpacaeval-batch`
- `arenahard-infer`, `arenahard-eval`, `arenahard-batch`
- `mtbench-infer`, `mtbench-eval`, `mtbench-batch`

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
- `eval/arenahard/`: Arena-Hard configs, templates, inference, evaluation, and
  batch orchestration.
- `eval/mtbench/`: MT-Bench configs, templates, inference, evaluation, and
  batch orchestration.
- `eval/benchmark_common.py`: shared path, config, JSONL, and command helpers
  for benchmark wrappers.
- `eval/model_generation.py`: shared tokenizer rendering and local generation
  helpers for `transformers` and `vllm`.

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

Arena-Hard:

```bash
uv run arenahard-infer --config eval/arenahard/config_arenahard.yaml
uv run arenahard-eval --config eval/arenahard/config_arenahard.yaml
uv run arenahard-batch --config eval/arenahard/config_arenahard_batch.yaml
```

MT-Bench:

```bash
uv run mtbench-infer --config eval/mtbench/config_mtbench.yaml
uv run mtbench-eval --config eval/mtbench/config_mtbench.yaml
uv run mtbench-batch --config eval/mtbench/config_mtbench_batch.yaml
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
- Arena-Hard and MT-Bench require benchmark question files. The checked-in base
  configs default to `questions.jsonl` inside each benchmark package, so update
  `arenahard.question_file` and `mtbench.question_file` to point at the real
  benchmark prompts in your environment.
- Arena-Hard and MT-Bench judging is wrapped, not bundled. Install the judge
  tools yourself and override `judge_command` if your local CLI differs from
  the checked-in examples.

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

Arena-Hard and MT-Bench use the same 8-model batch matrix and the same
Qwen3/Llama3 family defaults as AlpacaEval:

- Qwen3 uses tokenizer-default chat templating and `stop_token_ids: [151645]`.
- Llama3 uses checked-in model-specific templates and
  `stop_token_ids: [128001, 128009]`.

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

Arena-Hard and MT-Bench follow the same pattern with benchmark-specific config
blocks:

- `arenahard.*` and `mtbench.*` both include `model_name_or_path`,
  `pretty_name`, `backend`, `output_dir`, `use_custom_chat_template`,
  `prompt_template`, `generation`, `transformers`, and `vllm`.
- `arenahard.judge_command`, `arenahard.judge_config`,
  `arenahard.api_config`, `arenahard.question_file`, `arenahard.judge_model`,
  and `arenahard.baseline_model` control Arena-Hard judge invocation.
- `mtbench.judge_command`, `mtbench.show_result_command`,
  `mtbench.reference_answer_file`, `mtbench.question_file`, and
  `mtbench.judge_model` control MT-Bench judge invocation.

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

Arena-Hard and MT-Bench write benchmark-native answer payloads:

- `model_answer.jsonl`: generated answers in the format expected by the judge
  tool.
- `metadata.json`: local generation metadata.
- `results/`: judge outputs from the configured external command.

Relative paths in the config are resolved relative to the config file, not the
current shell directory.

## Tests

Relevant tests live in:

- `test/test_alpacaeval_pipeline.py`
- `test/test_alpacaeval_batch_runner.py`
- `test/test_arenahard_pipeline.py`
- `test/test_arenahard_batch_runner.py`
- `test/test_mtbench_pipeline.py`
- `test/test_mtbench_batch_runner.py`

The pipeline tests cover:

- single-model inference and evaluation wiring
- batch-runner config expansion
- Llama3/Qwen3 chat-template handling
- real 3-sample AlpacaEval smoke tests with real tokenizers and stubbed model
  generation
