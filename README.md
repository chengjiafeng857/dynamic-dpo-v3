# dynamic-dpo-v1

This repository is a small TRL + Transformers training workspace for:

- supervised fine-tuning (SFT)
- dynamic-beta DPO
- beta DPO

The most complete and directly runnable path in this checkout is the SFT pipeline.

## Requirements

- Python `>=3.11,<3.12`
- [`uv`](https://docs.astral.sh/uv/)

Dependencies and console scripts are defined in [pyproject.toml](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/pyproject.toml).

## Setup

Create a virtual environment and install the package in editable mode:

```bash
uv sync
```

You can also run commands with `uv run ...` without manually activating the venv once dependencies are installed.

## Available Commands

Installing the package exposes three console scripts:

- `train-sft`
- `train-dpo`
- `train-beta-dpo`

They map to:

- `src.cli:main_sft`
- `src.cli:main_dpo`
- `src.cli:main_beta_dpo`

## Run SFT

### 1. SFT on HH (`Anthropic/hh-rlhf`)

Use the default HH-focused config:

```bash
uv run train-sft --config config_sft.yaml
```

or, after activating the environment:

```bash
train-sft --config config_sft.yaml
```

This path:

- loads `Anthropic/hh-rlhf`
- converts each `chosen` conversation into `prompt` / `completion`
- uses `sft_training.completion_only_loss: true`
- applies the `llama3` chat template fallback from `dataset.chat_template_name`

### 2. SFT on UltraChat (`HuggingFaceH4/ultrachat_200k`)

Use the UltraChat config:

```bash
uv run train-sft --config config_sft_ultrachat.yaml
```

This path:

- loads `HuggingFaceH4/ultrachat_200k`
- uses `train_sft` for training and `test_sft` for eval
- keeps rows as `messages` because `sft_training.completion_only_loss: false`
- defaults to the `llama3` chat template fallback

## Choose the Model and Chat Template

The model is selected with `policy_name` in the YAML config.

The local chat-template fallback is selected with `dataset.chat_template_name`:

- `llama3`
- `qwen3`

Examples:

```yaml
policy_name: meta-llama/Llama-3.2-1B-Instruct
dataset:
  chat_template_name: llama3
```

```yaml
policy_name: Qwen/Qwen3-8B
dataset:
  chat_template_name: qwen3
```

If `dataset.chat_template_name` is omitted, the code will try to infer the template from `policy_name`:

- Llama 3 family -> `llama3`
- Qwen 3 family -> `qwen3`

## Important SFT Config Knobs

The main runtime behavior is controlled in the `sft_training` block.

The most important settings are:

- `completion_only_loss`
  - `true` -> train on `prompt` / `completion`
  - `false` -> train on full conversational `messages`
- `packing`
- `max_length`
- `gradient_accumulation`
- `save_strategy`
- `load_best_model_at_end`
- `save_dir`
- `push_to_hub`
- `hub_model_id`

## DPO and Beta DPO

The CLI entrypoints for DPO exist:

```bash
uv run train-dpo --config <your_dpo_config.yaml>
uv run train-beta-dpo --config <your_beta_dpo_config.yaml>
```

In this checkout, the packaged commands exist in [pyproject.toml](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/pyproject.toml), but the repository tree currently only includes the SFT trainer file under `src/trainers/`.

That means the SFT path is the only fully present training path in this repo snapshot.

If you restore the missing DPO trainer modules, note that this checkout also does not currently include example DPO config files in the repository root, while the CLI defaults expect:

- `config_dpo.yaml`
- `config_beta_dpo.yaml`

At that point, create those config files first or pass an explicit config path.

## Useful Validation Commands

Syntax-check the active Python modules:

```bash
uv run python -m py_compile \
  src/data/sft_dataset.py \
  src/data/ultrachat_dataset.py \
  src/data/templates.py \
  src/data/hh_dataset.py \
  src/trainers/sft_trainer.py \
  src/cli.py
```

Run the current test suite:

```bash
uv run python -m unittest -v test/test_chat_templates.py test/test_sft_pipeline_smoke.py
```

## What the SFT Runner Prints

When `train-sft` starts, it prints:

- the selected mode (`hh` or `ultrachat`)
- the dataset name
- `completion_only_loss`
- `packing`
- `max_length`
- `gradient_accumulation`
- one train sample preview
- one eval sample preview
- the best checkpoint path at the end, if available

This is useful for confirming that the dataset schema matches the intended mode before a long run.

## Common Workflow

1. Edit either [config_sft.yaml](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/config_sft.yaml) or [config_sft_ultrachat.yaml](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/config_sft_ultrachat.yaml).
2. Set `policy_name` and, if needed, `dataset.chat_template_name`.
3. Run `uv run train-sft --config <config-file>`.
4. Inspect the printed dataset preview.
5. Let training finish and optionally push to the Hub if `hub_model_id` is set.

## Current Example Configs

- [config_sft.yaml](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/config_sft.yaml): HH SFT, completion-only training
- [config_sft_ultrachat.yaml](/Users/seanmacbook/Research/dpo/dynamic-dpo-v3/config_sft_ultrachat.yaml): UltraChat SFT, full-sequence training
