"""SFT training utilities."""

import os
from typing import Any, Dict

from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from ..data.sft_dataset import (
    build_hh_sft_dataset,
    load_tokenizer,
)
from ..data.ultrachat_dataset import build_ultrachat_sft_dataset


ULTRACHAT_DATASET_ALLOWLIST = {"HuggingFaceH4/ultrachat_200k"}


def _summarize_sample(sample: Any, *, max_chars: int = 500) -> str:
    text = repr(sample)
    if len(text) > max_chars:
        return f"{text[:max_chars]}..."
    return text


def _print_dataset_preview(dataset: Any, *, label: str) -> None:
    size = len(dataset)
    if size == 0:
        print(f"[SFT] {label}_sample=None (empty dataset)")
        return

    sample = dataset[0]
    if isinstance(sample, dict):
        print(f"[SFT] {label}_sample_keys={list(sample.keys())}")
    print(f"[SFT] {label}_sample={_summarize_sample(sample)}")


def _parse_fsdp_options(sft_cfg: Dict[str, Any]) -> Dict[str, Any]:
    fsdp_cfg = sft_cfg.get("fsdp")
    if not isinstance(fsdp_cfg, dict) or not bool(fsdp_cfg.get("enabled", False)):
        return {"enabled": False, "args": {}, "state_dict_type": None}

    auto_wrap_policy = str(fsdp_cfg.get("auto_wrap_policy", "transformer_based_wrap")).lower()
    if auto_wrap_policy not in {"transformer_based_wrap", "size_based_wrap", "no_wrap"}:
        raise ValueError(
            "sft_training.fsdp.auto_wrap_policy must be one of "
            "'transformer_based_wrap', 'size_based_wrap', or 'no_wrap'."
        )

    mode_tokens = [token for token in str(fsdp_cfg.get("mode", "full_shard")).split() if token]
    if not mode_tokens:
        raise ValueError("sft_training.fsdp.mode must contain at least one FSDP option.")

    has_auto_wrap = "auto_wrap" in mode_tokens
    if auto_wrap_policy == "no_wrap":
        mode_tokens = [token for token in mode_tokens if token != "auto_wrap"]
    elif not has_auto_wrap:
        mode_tokens.append("auto_wrap")

    if not mode_tokens:
        raise ValueError("sft_training.fsdp.mode resolved to empty options.")

    fsdp_config: Dict[str, Any] = {
        "version": int(fsdp_cfg.get("version", 1)),
        "offload_params": bool(fsdp_cfg.get("offload_params", False)),
        "backward_prefetch": str(
            fsdp_cfg.get("backward_prefetch", "no_prefetch")
        ).upper(),
        "forward_prefetch": bool(fsdp_cfg.get("forward_prefetch", False)),
        "use_orig_params": bool(fsdp_cfg.get("use_orig_params", True)),
        "cpu_ram_efficient_loading": bool(
            fsdp_cfg.get("cpu_ram_efficient_loading", True)
        ),
        "sync_module_states": bool(fsdp_cfg.get("sync_module_states", True)),
        "activation_checkpointing": bool(
            fsdp_cfg.get("activation_checkpointing", False)
        ),
    }

    layer_cls = fsdp_cfg.get("transformer_layer_cls_to_wrap")
    min_num_params = fsdp_cfg.get("min_num_params")
    if layer_cls is not None and min_num_params is not None:
        raise ValueError(
            "sft_training.fsdp.transformer_layer_cls_to_wrap and "
            "sft_training.fsdp.min_num_params are mutually exclusive."
        )

    if auto_wrap_policy == "transformer_based_wrap" and layer_cls is not None:
        if isinstance(layer_cls, str):
            layer_names = [name.strip() for name in layer_cls.split(",") if name.strip()]
        elif isinstance(layer_cls, list):
            layer_names = [str(name).strip() for name in layer_cls if str(name).strip()]
        else:
            raise ValueError(
                "sft_training.fsdp.transformer_layer_cls_to_wrap must be a string or list."
            )
        if layer_names:
            fsdp_config["transformer_layer_cls_to_wrap"] = layer_names

    if auto_wrap_policy == "size_based_wrap":
        fsdp_config["min_num_params"] = int(
            min_num_params if min_num_params is not None else 100_000_000
        )

    state_dict_type = str(
        fsdp_cfg.get("state_dict_type", "FULL_STATE_DICT")
    ).upper()
    valid_state_dict_types = {
        "FULL_STATE_DICT",
        "LOCAL_STATE_DICT",
        "SHARDED_STATE_DICT",
    }
    if state_dict_type not in valid_state_dict_types:
        raise ValueError(
            "sft_training.fsdp.state_dict_type must be one of FULL_STATE_DICT, "
            "LOCAL_STATE_DICT, or SHARDED_STATE_DICT."
        )

    return {
        "enabled": True,
        "args": {
            "fsdp": " ".join(mode_tokens),
            "fsdp_config": fsdp_config,
        },
        "state_dict_type": state_dict_type,
    }


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
    dataset_config_name = dataset_cfg.get("config_name")
    chat_template_name = dataset_cfg.get("chat_template_name")
    is_ultrachat = dataset_name in ULTRACHAT_DATASET_ALLOWLIST

    if "completion_only_loss" in sft_cfg:
        completion_only_loss = bool(sft_cfg["completion_only_loss"])
    else:
        completion_only_loss = False if is_ultrachat else True

    tok = load_tokenizer(
        model_name,
        padding_side="right",
        chat_template_name=chat_template_name,
    )

    if is_ultrachat:
        train_subset = dataset_cfg.get("subset", "train_sft")
        eval_subset = dataset_cfg.get("eval_subset", "test_sft")
        train_raw_ds = load_dataset(
            dataset_name,
            name=dataset_config_name,
            split=train_subset,
        )
        eval_raw_ds = load_dataset(
            dataset_name,
            name=dataset_config_name,
            split=eval_subset,
        )
        train_ds = build_ultrachat_sft_dataset(
            train_raw_ds,
            completion_only_loss=completion_only_loss,
        )
        eval_ds = build_ultrachat_sft_dataset(
            eval_raw_ds,
            completion_only_loss=completion_only_loss,
        )
    else:
        raw_ds = load_dataset(
            dataset_name,
            name=dataset_config_name,
            split=dataset_cfg["subset"],
        )
        sft_ds = build_hh_sft_dataset(raw_ds, tok)
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
    grad_accum = int(sft_cfg.get("gradient_accumulation", 1))
    if grad_accum < 1:
        raise ValueError("sft_training.gradient_accumulation must be >= 1.")

    fsdp_options = _parse_fsdp_options(sft_cfg)
    if fsdp_options["enabled"] and fsdp_options["state_dict_type"] is not None:
        os.environ["FSDP_STATE_DICT_TYPE"] = str(fsdp_options["state_dict_type"])

    gradient_checkpointing = bool(sft_cfg.get("gradient_checkpointing", True))

    save_strategy = str(sft_cfg.get("save_strategy", "best")).lower()
    load_best_model_at_end = bool(sft_cfg.get("load_best_model_at_end", True))
    metric_for_best_model = str(sft_cfg.get("metric_for_best_model", "eval_loss"))
    greater_is_better = bool(sft_cfg.get("greater_is_better", False))
    save_only_model = bool(sft_cfg.get("save_only_model", True))
    if fsdp_options["enabled"] and load_best_model_at_end and save_only_model:
        print(
            "[SFT] Overriding save_only_model=False for FSDP with "
            "load_best_model_at_end to avoid checkpoint incompatibilities."
        )
        save_only_model = False

    sft_args_kwargs = dict(
        output_dir=sft_cfg["save_dir"],
        learning_rate=float(sft_cfg["learning_rate"]),
        per_device_train_batch_size=int(sft_cfg["batch_size"]),
        per_device_eval_batch_size=int(sft_cfg["eval_batch_size"]),
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=int(sft_cfg["epochs"]),
        logging_steps=int(sft_cfg["log_steps"]),
        eval_strategy="steps",
        eval_steps=int(sft_cfg.get("eval_steps", sft_cfg["save_steps"])),
        save_strategy=save_strategy,
        save_steps=int(sft_cfg["save_steps"]),
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        max_length=int(sft_cfg["max_length"]),
        packing=packing,
        lr_scheduler_type=scheduler,
        save_only_model=save_only_model,
        fp16=fp16,
        bf16=bf16,
        report_to=["wandb"] if sft_cfg.get("wandb_project") else [],
        run_name=str(sft_cfg.get("run_name", "sft")),
        remove_unused_columns=False,
        gradient_checkpointing=gradient_checkpointing,
        hub_model_id=sft_cfg.get("hub_model_id"),
        push_to_hub=bool(sft_cfg.get("push_to_hub")),
        dataset_text_field="messages",
        completion_only_loss=completion_only_loss,
    )
    if "save_total_limit" in sft_cfg and sft_cfg["save_total_limit"] is not None:
        sft_args_kwargs["save_total_limit"] = int(sft_cfg["save_total_limit"])
    else:
        sft_args_kwargs["save_total_limit"] = 2
    if fsdp_options["enabled"]:
        sft_args_kwargs.update(fsdp_options["args"])

    warmup_steps = sft_cfg.get("warmup_steps")
    if warmup_steps is not None and int(warmup_steps) > 0:
        sft_args_kwargs["warmup_steps"] = int(warmup_steps)
    elif "warmup_ratio" in sft_cfg:
        sft_args_kwargs["warmup_ratio"] = float(sft_cfg["warmup_ratio"])

    training_args = SFTConfig(**sft_args_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    mode = "ultrachat" if is_ultrachat else "hh"
    print(
        f"[SFT] mode={mode} dataset={dataset_name} "
        f"completion_only_loss={completion_only_loss} packing={packing} "
        f"max_length={training_args.max_length} "
        f"grad_accum={training_args.gradient_accumulation_steps}"
    )
    _print_dataset_preview(train_ds, label="train")
    _print_dataset_preview(eval_ds, label="eval")

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
    if trainer.state.best_model_checkpoint:
        print(f"[SFT] best_model_checkpoint={trainer.state.best_model_checkpoint}")
    else:
        print("[SFT] best_model_checkpoint=None")
    trainer.save_model()

    return trainer
