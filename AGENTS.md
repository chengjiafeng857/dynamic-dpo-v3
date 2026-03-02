# AGENTS.md (Strict Operating Rules)

This file defines hard rules for autonomous/code-assist agents working in this repository.
If this file conflicts with informal notes, this file wins.

## 1) Mission and Scope
- Keep changes minimal, local, and reversible.
- Preserve training behavior unless the task explicitly requests behavior change.
- Focus on active paths:
  - `src/trainers/sft_trainer.py`
  - `src/data/sft_dataset.py`
  - `src/data/hh_dataset.py`
  - `src/cli.py`
  - `src/config/loader.py`
  - `config_sft.yaml`
  - `config_sft_ultrachat.yaml`
  - `test/test_sft_pipeline_smoke.py`

## 2) Hard Constraints
- MUST NOT perform broad refactors, renames, or file moves unless explicitly requested.
- MUST NOT change dataset column contracts silently.
- MUST NOT add new dependencies without explicit user request.
- MUST NOT change default config behavior unless explicitly requested.
- MUST NOT commit secrets, credentials, tokens, or local env files.
- MUST keep edits ASCII unless file already requires Unicode.

## 3) Data and Loss Invariants (Do Not Break)
- HH SFT path expects prompt-completion style rows.
  - Source: `build_sft_dataset`.
  - Contract: `{"prompt": [...], "completion": [...]}`.
- UltraChat SFT path behavior is controlled by `sft_training.completion_only_loss`.
  - `false` -> `{"messages": ...}` rows (full-sequence).
  - `true` -> exploded `{"prompt": ..., "completion": ...}` rows (assistant turns).
- `sft_training.completion_only_loss` is the only supported key for this toggle.
- SFT trainer prints one train and one eval sample right before `SFTTrainer` construction.
  - Keep this preview behavior unless explicitly requested to remove it.

## 4) Configuration Rules
- Any behavior change must be represented in config, not hardcoded.
- Keep config keys stable unless explicitly requested.
- If changing semantics, update both:
  - code path
  - example/default config files (`config_sft.yaml`, `config_sft_ultrachat.yaml`)
- Current SFT best-checkpoint defaults are config-driven and should remain explicit:
  - `save_strategy: best`
  - `load_best_model_at_end: true`
  - `metric_for_best_model: eval_loss`
  - `greater_is_better: false`
  - `save_total_limit: 2`
- Current FSDP behavior is config-driven under `sft_training.fsdp`:
  - Keep `gradient_accumulation` as the single accumulation key for both FSDP and non-FSDP.
  - Keep auto-correction: if `fsdp.enabled=true` and `load_best_model_at_end=true`, force `save_only_model=false`.
  - `gradient_checkpointing` is allowed with full-shard FSDP in this repo (Transformers may emit a warning).

## 5) Editing Workflow (Required)
- First inspect relevant files with targeted search/read.
- Implement smallest possible patch.
- Avoid touching unrelated lines/files.
- Prefer explicit variable names over implicit logic.
- Keep comments sparse and only for non-obvious logic.
- Use `uv` as the default execution path for Python commands in this repo.
- Prefer `uv run <cmd>` over calling `.venv/bin/python` directly.

## 6) Validation (Required Before Handover)
- Run syntax checks for touched Python modules:
  - `uv run python -m py_compile <touched_py_files>`
- For SFT pipeline changes, run smoke tests:
  - `uv run python -m unittest -v test/test_sft_pipeline_smoke.py`
- If behavior changes affect preprocessing:
  - verify output schema matches intended mode (`messages` vs `prompt/completion`).
- If a check cannot run (missing dependency/environment):
  - state exactly what failed and why.

## 7) Git and Artifact Hygiene
- Do not modify generated artifacts intentionally (`*.egg-info`, caches, model outputs, logs).
- Respect `.gitignore`.
- Do not remove user changes outside task scope.
- Never use destructive git commands unless explicitly asked.

## 8) Output Expectations
- Report exactly what changed and why.
- Provide concrete file references with line anchors when possible.
- Call out residual risks or unverified assumptions.
- Keep responses concise and technical.
