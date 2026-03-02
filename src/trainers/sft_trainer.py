"""SFT training utilities."""

from typing import Any, Dict

from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from ..data.sft_dataset import (
    build_sft_dataset,
    build_ultrachat_sft_dataset,
    load_tokenizer,
)
from ..data.templates import LLAMA3_CHAT_TEMPLATE


ULTRACHAT_DATASET_ALLOWLIST = {"HuggingFaceH4/ultrachat_200k"}


def run_sft_training(config: Dict[str, Any]) -> SFTTrainer:
    """Run SFT training based on configuration.
    
    Args:
        config: Configuration dictionary with model, dataset, and training settings.
        
    Returns:
        The trained SFTTrainer instance.
    """
    model_name = config["policy_name"]
    sft_cfg = config["sft_training"]
    dataset_cfg = config["dataset"]
    dataset_name = dataset_cfg["dataset_name"]
    is_ultrachat = dataset_name in ULTRACHAT_DATASET_ALLOWLIST

    if "completion_only_loss" in sft_cfg:
        completion_only_loss = bool(sft_cfg["completion_only_loss"])
    else:
        completion_only_loss = False if is_ultrachat else True

    tok = load_tokenizer(model_name, padding_side="right")

    if not tok.chat_template:
        tok.chat_template = LLAMA3_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if is_ultrachat:
        train_subset = dataset_cfg.get("subset", "train_sft")
        eval_subset = dataset_cfg.get("eval_subset", "test_sft")
        train_raw_ds = load_dataset(dataset_name, split=train_subset)
        eval_raw_ds = load_dataset(dataset_name, split=eval_subset)
        train_ds = build_ultrachat_sft_dataset(
            train_raw_ds,
            completion_only_loss=completion_only_loss,
        )
        eval_ds = build_ultrachat_sft_dataset(
            eval_raw_ds,
            completion_only_loss=completion_only_loss,
        )
    else:
        raw_ds = load_dataset(dataset_name, split=dataset_cfg["subset"])
        sft_ds = build_sft_dataset(raw_ds, tok)
        val_ratio = float(dataset_cfg["val_ratio"])
        seed = int(dataset_cfg["seed"])
        split = sft_ds.train_test_split(test_size=val_ratio, seed=seed)
        train_ds = split["train"]
        eval_ds = split["test"]

    prec = config["precision"].lower()
    fp16 = prec == "fp16"
    bf16 = prec == "bf16"

    packing = bool(sft_cfg.get("packing", is_ultrachat))
    scheduler = str(
        sft_cfg.get("lr_scheduler_type", "cosine" if is_ultrachat else "linear")
    )

    sft_args_kwargs = dict(
        output_dir=sft_cfg["save_dir"],
        learning_rate=float(sft_cfg["learning_rate"]),
        per_device_train_batch_size=int(sft_cfg["batch_size"]),
        per_device_eval_batch_size=int(sft_cfg["eval_batch_size"]),
        gradient_accumulation_steps=int(sft_cfg.get("gradient_accumulation", 1)),
        num_train_epochs=int(sft_cfg["epochs"]),
        logging_steps=int(sft_cfg["log_steps"]),
        eval_strategy="steps",
        eval_steps=int(sft_cfg.get("eval_steps", sft_cfg["save_steps"])),
        save_strategy="steps",
        save_steps=int(sft_cfg["save_steps"]),
        max_length=int(sft_cfg["max_length"]),
        packing=packing,
        lr_scheduler_type=scheduler,
        save_only_model=bool(sft_cfg.get("save_only_model", True)),
        fp16=fp16,
        bf16=bf16,
        report_to=["wandb"] if sft_cfg.get("wandb_project") else [],
        run_name=str(sft_cfg.get("run_name", "sft")),
        remove_unused_columns=False,
        hub_model_id=sft_cfg.get("hub_model_id"),
        push_to_hub=bool(sft_cfg.get("push_to_hub")),
        dataset_text_field="messages",
        completion_only_loss=completion_only_loss,
    )

    warmup_steps = sft_cfg.get("warmup_steps")
    if warmup_steps is not None and int(warmup_steps) > 0:
        sft_args_kwargs["warmup_steps"] = int(warmup_steps)
    elif "warmup_ratio" in sft_cfg:
        sft_args_kwargs["warmup_ratio"] = float(sft_cfg["warmup_ratio"])

    training_args = SFTConfig(**sft_args_kwargs)

    mode = "ultrachat" if is_ultrachat else "hh"
    print(
        f"[SFT] mode={mode} dataset={dataset_name} "
        f"completion_only_loss={completion_only_loss} packing={packing} "
        f"max_length={training_args.max_length} "
        f"grad_accum={training_args.gradient_accumulation_steps}"
    )

    # WandB initialization
    wandb_project = sft_cfg.get("wandb_project")
    if wandb_project:
        import torch.distributed as dist
        is_main = (
            (not dist.is_available())
            or (not dist.is_initialized())
            or (dist.get_rank() == 0)
        )
        if is_main:
            import wandb

            wandb.init(
                project=str(wandb_project),
                name=training_args.run_name,
                config=config,
            )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
    )

    trainer.train()
    trainer.save_model()

    return trainer
