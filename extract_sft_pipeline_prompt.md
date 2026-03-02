# Existing SFT Pipeline Extraction (Code-Accurate)

## 1) Pipeline Overview
1. Entry command `train-sft` resolves to `src.cli:main_sft`.  
   Reference: `pyproject.toml:31`.
2. `main_sft` parses `--config` (default `config_sft.yaml`), loads YAML, then calls `run_sft_training(config)`.  
   Reference: `src/cli.py:159`, `src/cli.py:162`, `src/cli.py:163`, `src/config/loader.py:6`.
3. `run_sft_training` reads `policy_name`, `sft_training`, and `dataset`; loads tokenizer with `padding_side="right"`, pad-token fallback to EOS, and chat-template fallback.  
   Reference: `src/trainers/sft_trainer.py:23`, `src/trainers/sft_trainer.py:24`, `src/trainers/sft_trainer.py:25`, `src/trainers/sft_trainer.py:27`, `src/data/sft_dataset.py:27`, `src/data/sft_dataset.py:29`, `src/data/sft_dataset.py:31`, `src/trainers/sft_trainer.py:29`, `src/trainers/sft_trainer.py:30`.
4. Model is loaded from `policy_name` with `AutoModelForCausalLM.from_pretrained`.  
   Reference: `src/trainers/sft_trainer.py:32`.
5. Raw dataset is loaded from HF using `dataset_name` and `subset`, then transformed by `build_sft_dataset` into prompt-completion examples.  
   Reference: `src/trainers/sft_trainer.py:34`, `src/trainers/sft_trainer.py:35`, `src/data/sft_dataset.py:36`.
6. Dataset is split into train/eval with `train_test_split(test_size=val_ratio, seed=seed)`.  
   Reference: `src/trainers/sft_trainer.py:37`, `src/trainers/sft_trainer.py:38`, `src/trainers/sft_trainer.py:39`.
7. Precision flags are derived from top-level `precision`; SFTConfig is created from `sft_training` values.  
   Reference: `src/trainers/sft_trainer.py:43`, `src/trainers/sft_trainer.py:44`, `src/trainers/sft_trainer.py:45`, `src/trainers/sft_trainer.py:47`.
8. SFTConfig hard-sets `dataset_text_field="messages"` and `completion_only_loss=True`.  
   Reference: `src/trainers/sft_trainer.py:68`, `src/trainers/sft_trainer.py:69`.
9. Optional manual W&B init occurs only on main rank when `wandb_project` is set.  
   Reference: `src/trainers/sft_trainer.py:73`, `src/trainers/sft_trainer.py:75`, `src/trainers/sft_trainer.py:76`, `src/trainers/sft_trainer.py:79`.
10. `SFTTrainer` is instantiated, training runs, then `trainer.save_model()` is called.  
    Reference: `src/trainers/sft_trainer.py:85`, `src/trainers/sft_trainer.py:93`, `src/trainers/sft_trainer.py:94`.
11. Back in `main_sft`, main rank optionally prompts for an additional explicit `trainer.push_to_hub()` if `hub_model_id` exists.  
    Reference: `src/cli.py:174`, `src/cli.py:176`, `src/cli.py:177`, `src/cli.py:180`, `src/cli.py:188`.

## 2) Data Contract at Each Stage

### Raw HH Input Format
- `build_sft_dataset` expects each row to contain a `chosen` text conversation in Anthropic HH textual turn format.
- HH turn tags are parsed by regex `\n\n(Human|Assistant): ?`.
Reference: `src/data/sft_dataset.py:52`, `src/data/sft_dataset.py:54`, `src/data/templates.py:7`.

### Post-Parse Format
- `parse_hh_to_messages(text)` normalizes newlines, ensures an initial turn tag, splits by HH tags, trims content, skips empty blocks, and maps:
  - `Human -> {"role": "user", "content": ...}`
  - `Assistant -> {"role": "assistant", "content": ...}`
Reference: `src/data/templates.py:30`, `src/data/templates.py:41`, `src/data/templates.py:42`, `src/data/templates.py:45`, `src/data/templates.py:50`, `src/data/templates.py:51`, `src/data/templates.py:53`, `src/data/templates.py:54`.

### Final SFT Dataset Schema Used by Trainer
- `build_sft_dataset` outputs HuggingFace `Dataset` rows with:
  - `prompt`: `messages[:-1]` (list of message dicts)
  - `completion`: `[messages[-1]]` (single-item list containing final assistant message)
- Row filter invariants:
  - skip if no parse or too short (`len(messages) < 2`)
  - skip if last message role is not assistant
  - skip if prompt is empty or prompt last role is not user
Reference: `src/data/sft_dataset.py:58`, `src/data/sft_dataset.py:59`, `src/data/sft_dataset.py:61`, `src/data/sft_dataset.py:64`, `src/data/sft_dataset.py:65`, `src/data/sft_dataset.py:67`, `src/data/sft_dataset.py:72`, `src/data/sft_dataset.py:73`.

### Tokenization and Masking Behavior
- TRL SFT detects prompt-completion examples via `"prompt" in example`.
- For conversational prompt-completion rows, tokenization path:
  - tokenizes prompt with `apply_chat_template(..., add_generation_prompt=True)`
  - tokenizes prompt+completion
  - builds `completion_mask = [0...0(prompt), 1...1(completion)]`
- Collator applies label masking:
  - padding tokens -> `-100`
  - if `completion_only_loss=True` and `completion_mask` exists, tokens with mask 0 -> `-100`
Reference: `.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:977`, `.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:987`, `.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:997`, `.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:1037`, `.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:1038`, `.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:201`, `.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:213`, `.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:217`.

## 3) Config-to-Behavior Mapping

### Top-Level
- `policy_name`: model/tokenizer source for SFT training.
  Reference: `src/trainers/sft_trainer.py:23`, `src/trainers/sft_trainer.py:27`, `src/trainers/sft_trainer.py:32`, `config_sft.yaml:1`.
- `precision`: controls `fp16`/`bf16` flags passed into `SFTConfig`.
  Reference: `src/trainers/sft_trainer.py:43`, `src/trainers/sft_trainer.py:44`, `src/trainers/sft_trainer.py:45`, `src/trainers/sft_trainer.py:61`, `src/trainers/sft_trainer.py:62`, `config_sft.yaml:3`.

### `dataset` keys
- `dataset.dataset_name`: HF dataset ID used by `load_dataset`.
  Reference: `src/trainers/sft_trainer.py:34`, `config_sft.yaml:6`.
- `dataset.subset`: split expression passed to `load_dataset(..., split=...)`.
  Reference: `src/trainers/sft_trainer.py:34`, `config_sft.yaml:9`.
- `dataset.val_ratio`: test size for train/eval split.
  Reference: `src/trainers/sft_trainer.py:37`, `src/trainers/sft_trainer.py:39`, `config_sft.yaml:10`.
- `dataset.seed`: split seed.
  Reference: `src/trainers/sft_trainer.py:38`, `src/trainers/sft_trainer.py:39`, `config_sft.yaml:11`.
- `dataset.generated_data`: no runtime effect in SFT pipeline.
  Reference: `src/trainers/sft_trainer.py:14`, `src/trainers/sft_trainer.py:25`, `config_sft.yaml:7`.
- `dataset.chat_template`: no runtime effect in SFT pipeline (template handling is internal tokenizer fallback logic).
  Reference: `src/trainers/sft_trainer.py:29`, `src/data/sft_dataset.py:31`, `config_sft.yaml:8`.
- `dataset.max_len`: no runtime effect in SFT pipeline (sequence length control uses `sft_training.max_length`).
  Reference: `src/trainers/sft_trainer.py:59`, `config_sft.yaml:12`.

### `sft_training` keys
- `learning_rate` -> `SFTConfig.learning_rate`.
  Reference: `src/trainers/sft_trainer.py:49`, `config_sft.yaml:31`.
- `batch_size` -> `per_device_train_batch_size`.
  Reference: `src/trainers/sft_trainer.py:50`, `config_sft.yaml:32`.
- `eval_batch_size` -> `per_device_eval_batch_size`.
  Reference: `src/trainers/sft_trainer.py:51`, `config_sft.yaml:33`.
- `epochs` -> `num_train_epochs`.
  Reference: `src/trainers/sft_trainer.py:52`, `config_sft.yaml:34`.
- `log_steps` -> `logging_steps`.
  Reference: `src/trainers/sft_trainer.py:53`, `config_sft.yaml:35`.
- `wandb_project` -> enables both `report_to=["wandb"]` and manual `wandb.init`.
  Reference: `src/trainers/sft_trainer.py:63`, `src/trainers/sft_trainer.py:73`, `src/trainers/sft_trainer.py:79`, `config_sft.yaml:36`.
- `save_steps` -> checkpoint save interval; also fallback `eval_steps` if `eval_steps` absent.
  Reference: `src/trainers/sft_trainer.py:55`, `src/trainers/sft_trainer.py:57`, `config_sft.yaml:37`.
- `warmup_steps` -> `SFTConfig.warmup_steps`.
  Reference: `src/trainers/sft_trainer.py:58`, `config_sft.yaml:38`.
- `max_length` -> truncation length in dataset prep when packing disabled.
  Reference: `src/trainers/sft_trainer.py:59`, `.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:1103`, `.venv/lib/python3.11/site-packages/trl/data_utils.py:787`, `config_sft.yaml:39`.
- `save_dir` -> `output_dir` for checkpoints and explicit final `save_model()`.
  Reference: `src/trainers/sft_trainer.py:48`, `src/trainers/sft_trainer.py:94`, `.venv/lib/python3.11/site-packages/transformers/trainer.py:4185`, `config_sft.yaml:40`.
- `save_only_model` -> if true, checkpoints omit optimizer/scheduler/scaler/RNG state.
  Reference: `src/trainers/sft_trainer.py:60`, `.venv/lib/python3.11/site-packages/transformers/trainer.py:3343`, `config_sft.yaml:41`.
- `push_to_hub` -> enables Trainer auto hub syncing on save/checkpoint paths.
  Reference: `src/trainers/sft_trainer.py:67`, `.venv/lib/python3.11/site-packages/transformers/trainer.py:3364`, `.venv/lib/python3.11/site-packages/transformers/trainer.py:4230`, `config_sft.yaml:42`.
- `hub_model_id` -> hub repo id in training args and used by interactive push prompt in CLI.
  Reference: `src/trainers/sft_trainer.py:66`, `src/cli.py:177`, `config_sft.yaml:43`.

## 4) Loss Semantics Verification (`completion_only_loss=True`)

### Conclusion
- `completion_only_loss=True` is active under the current dataset format.

### Why (Code Path)
1. Local SFT setup explicitly sets `completion_only_loss=True`.
   Reference: `src/trainers/sft_trainer.py:69`.
2. Dataset rows are prompt-completion rows (`prompt`, `completion`) built by `build_sft_dataset`.
   Reference: `src/data/sft_dataset.py:72`, `src/data/sft_dataset.py:73`.
3. TRL tokenizer path for prompt-completion creates `completion_mask`.
   Reference: `.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:977`, `.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:1037`.
4. TRL collator masks all non-completion tokens in labels when completion-only loss is enabled.
   Reference: `.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:213`, `.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:217`.

### Note on `dataset_text_field="messages"`
- In this pipeline it is effectively inert for the training set because prompt-completion rows use the `"prompt"` branch, not the language-modeling branch that reads `dataset_text_field`.
Reference: `.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:977`, `.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:1060`.

## 5) Critical Invariants / Assumptions
- Raw rows must provide `chosen` in HH text format, otherwise skipped.
  Reference: `src/data/sft_dataset.py:52`, `src/data/sft_dataset.py:56`.
- Parsed conversation must contain at least 2 messages.
  Reference: `src/data/sft_dataset.py:59`.
- Final message must be assistant.
  Reference: `src/data/sft_dataset.py:61`.
- Prompt (conversation prefix) must end with user.
  Reference: `src/data/sft_dataset.py:67`.
- Tokenizer must have pad token and chat template (local code assigns fallbacks).
  Reference: `src/data/sft_dataset.py:29`, `src/data/sft_dataset.py:31`.
- For push behavior, authenticated hub credentials and repo permissions are required.
  Reference: `.venv/lib/python3.11/site-packages/transformers/trainer.py:5078`.

## 6) Risks and Gaps (Current Behavior)
- Config drift: `dataset.generated_data`, `dataset.chat_template`, and `dataset.max_len` are present in config but unused by SFT path.
  Reference: `config_sft.yaml:7`, `config_sft.yaml:8`, `config_sft.yaml:12`, `src/trainers/sft_trainer.py:25`.
- `dataset_text_field="messages"` can mislead readers, since the active path is prompt-completion tokenization rather than language-modeling-on-`messages`.
  Reference: `src/trainers/sft_trainer.py:68`, `.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:977`.
- Silent example dropping in `build_sft_dataset` (no counters/logging for skipped rows).
  Reference: `src/data/sft_dataset.py:56`, `src/data/sft_dataset.py:59`, `src/data/sft_dataset.py:61`, `src/data/sft_dataset.py:68`.
- Potential duplicate pushes: if `push_to_hub=True`, Trainer may already push during saves; CLI can still trigger an additional interactive push.
  Reference: `src/trainers/sft_trainer.py:67`, `.venv/lib/python3.11/site-packages/transformers/trainer.py:3364`, `.venv/lib/python3.11/site-packages/transformers/trainer.py:4230`, `src/cli.py:188`.

## 7) Explicit Prompt+Completion Data-Path Check
- Executed check on current code path:
  - Constructed a one-row synthetic HH dataset and ran `build_sft_dataset`.
  - Observed output columns: `['prompt', 'completion']`.
  - Verified `prompt[-1]['role'] == 'user'` and `completion[0]['role'] == 'assistant'`.
- This confirms that current preprocessing emits prompt-completion conversational records, matching TRL completion-mask training behavior.
Reference: `src/data/sft_dataset.py:36`, `src/data/sft_dataset.py:64`, `src/data/sft_dataset.py:65`, `src/data/sft_dataset.py:72`, `src/data/sft_dataset.py:73`.

## Environment Verification Used for Semantics
- Installed TRL: `0.26.2`
- Installed Transformers: `4.57.3`
