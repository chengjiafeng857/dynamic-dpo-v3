# Agent Guide: dynamic-dpo-v2

## Scope
This repository is a training workspace for SFT/DPO workflows around TRL + Transformers.
The most stable/active paths in this checkout are:
- `src/trainers/sft_trainer.py`
- `src/data/sft_dataset.py`
- `src/data/hh_dataset.py`
- `src/cli.py`
- `config_sft.yaml`
- `config_sft_ultrachat.yaml`

## Environment
- Python: `>=3.11,<3.12`
- Package/deps: `pyproject.toml`, `uv.lock`
- Console scripts:
  - `train-sft`
  - `train-dpo`
  - `train-beta-dpo`

Suggested local setup:
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

## Runbook
SFT (HH config):
```bash
train-sft --config config_sft.yaml
```

SFT (UltraChat config):
```bash
train-sft --config config_sft_ultrachat.yaml
```

## Data/Loss Contracts (Important)
HH SFT path (`Anthropic/hh-rlhf` style):
- Preprocessing emits `prompt` + `completion`.
- `completion_only_loss: true` -> train only on completion tokens.
- `completion_only_loss: false` -> full-sequence loss on prompt+completion tokens.

UltraChat SFT path (`HuggingFaceH4/ultrachat_200k` allowlist):
- `completion_only_loss: false` -> dataset rows stay as `messages` (full-sequence CLM).
- `completion_only_loss: true` -> each assistant turn becomes one `prompt` + `completion` sample.

## Configuration Keys to Treat Carefully
In `sft_training`:
- `completion_only_loss` controls masking/data format behavior.
- `packing` and `max_length` strongly affect tokenization/runtime.
- `push_to_hub` + `hub_model_id` can cause remote side effects.

In `dataset`:
- `dataset_name`, `subset`, `eval_subset`, `seed`, `val_ratio` define data split and source.

## Safe Change Rules
- Prefer minimal diffs; keep current data contracts intact.
- Do not silently change column schema (`messages` vs `prompt`/`completion`) without updating trainer logic.
- If modifying loss behavior, update both:
  - preprocessing (`src/data/sft_dataset.py`)
  - config resolution (`src/trainers/sft_trainer.py`)
- Keep generated artifacts out of commits (`.venv/`, `*.egg-info/`, logs, model outputs).

## Validation Checklist
For small code edits:
```bash
python3 -m py_compile src/data/sft_dataset.py src/trainers/sft_trainer.py src/cli.py
```

For behavior changes:
- Verify SFT preprocessing output columns match the intended loss mode.
- Verify config defaults in both SFT config files.
- Run one short smoke train with reduced subset if credentials/hardware are available.

## Notes
- This checkout currently has no dedicated test suite; use targeted smoke checks.
- `extract_sft_pipeline_prompt.md` documents the existing SFT flow and invariants.
