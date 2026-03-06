"""CLI entry points for DPO and SFT training."""

import argparse
import os
from typing import Any, Dict, Tuple

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig

from .config.loader import load_yaml
from .data.hh_dataset import (
    apply_chat_template_to_dataset,
    build_HH_dataset,
    load_generated_dataset_from_config,
)
from .trainers.sft_trainer import run_sft_training


def _is_main_process() -> bool:
    import torch.distributed as dist

    return (
        (not dist.is_available())
        or (not dist.is_initialized())
        or (dist.get_rank() == 0)
    )


def _build_dpo_parser(
    *, description: str, default_config: str, default_output_dir: str
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    return parser


def _load_policy_ref_and_tokenizer(
    config: Dict[str, Any],
) -> Tuple[Any, Any, Any, str]:
    policy_name = config["policy_name"]
    ref_name = config["ref_name"]

    tokenizer = AutoTokenizer.from_pretrained(policy_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(policy_name)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_name)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    return policy, ref_model, tokenizer, policy_name


def _load_hh_raw_dataset(config: Dict[str, Any]):
    dataset_cfg = config["dataset"]
    dataset_name = dataset_cfg["dataset_name"]
    subset = dataset_cfg["subset"]
    hh_data_dir = dataset_cfg.get("data_dir")
    if hh_data_dir is None and dataset_cfg.get("config_name") is not None:
        hh_data_dir = dataset_cfg["config_name"]
    return load_dataset(dataset_name, data_dir=hh_data_dir, split=subset)


def _build_hh_dpo_datasets(
    config: Dict[str, Any], tokenizer: Any, policy_name: str
) -> Tuple[Dataset, Dataset]:
    dataset_cfg = config["dataset"]

    if bool(dataset_cfg.get("generated_data", False)):
        hh_ds = load_generated_dataset_from_config(config)
    else:
        raw_ds = _load_hh_raw_dataset(config)
        hh_ds = build_HH_dataset(raw_ds)

    if bool(dataset_cfg.get("chat_template", False)):
        hh_ds = apply_chat_template_to_dataset(
            hh_ds,
            tokenizer,
            model_name=policy_name,
            chat_template_name=dataset_cfg.get("chat_template_name"),
        )

    split = hh_ds.train_test_split(
        test_size=float(dataset_cfg["val_ratio"]),
        seed=int(dataset_cfg["seed"]),
    )
    return split["train"], split["test"]


def _parse_dpo_fsdp_options(dpo_cfg: Dict[str, Any]) -> Dict[str, Any]:
    fsdp_cfg = dpo_cfg.get("fsdp")
    if not isinstance(fsdp_cfg, dict) or not bool(fsdp_cfg.get("enabled", False)):
        return {"enabled": False, "args": {}, "state_dict_type": None}

    auto_wrap_policy = str(
        fsdp_cfg.get("auto_wrap_policy", "transformer_based_wrap")
    ).lower()
    if auto_wrap_policy not in {"transformer_based_wrap", "size_based_wrap", "no_wrap"}:
        raise ValueError(
            "dpo_training.fsdp.auto_wrap_policy must be one of "
            "'transformer_based_wrap', 'size_based_wrap', or 'no_wrap'."
        )

    mode_tokens = [
        token for token in str(fsdp_cfg.get("mode", "full_shard")).split() if token
    ]
    if not mode_tokens:
        raise ValueError(
            "dpo_training.fsdp.mode must contain at least one FSDP option."
        )

    has_auto_wrap = "auto_wrap" in mode_tokens
    if auto_wrap_policy == "no_wrap":
        mode_tokens = [token for token in mode_tokens if token != "auto_wrap"]
    elif not has_auto_wrap:
        mode_tokens.append("auto_wrap")

    if not mode_tokens:
        raise ValueError("dpo_training.fsdp.mode resolved to empty options.")

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
            "dpo_training.fsdp.transformer_layer_cls_to_wrap and "
            "dpo_training.fsdp.min_num_params are mutually exclusive."
        )

    if auto_wrap_policy == "transformer_based_wrap" and layer_cls is not None:
        if isinstance(layer_cls, str):
            layer_names = [
                name.strip() for name in layer_cls.split(",") if name.strip()
            ]
        elif isinstance(layer_cls, list):
            layer_names = [str(name).strip() for name in layer_cls if str(name).strip()]
        else:
            raise ValueError(
                "dpo_training.fsdp.transformer_layer_cls_to_wrap must be a string or list."
            )
        if layer_names:
            fsdp_config["transformer_layer_cls_to_wrap"] = layer_names

    if auto_wrap_policy == "size_based_wrap":
        fsdp_config["min_num_params"] = int(
            min_num_params if min_num_params is not None else 100_000_000
        )

    state_dict_type = str(fsdp_cfg.get("state_dict_type", "FULL_STATE_DICT")).upper()
    valid_state_dict_types = {
        "FULL_STATE_DICT",
        "LOCAL_STATE_DICT",
        "SHARDED_STATE_DICT",
    }
    if state_dict_type not in valid_state_dict_types:
        raise ValueError(
            "dpo_training.fsdp.state_dict_type must be one of FULL_STATE_DICT, "
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


def _build_common_dpo_config_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    dpo_train_args = config["dpo_training"]
    dataset_cfg = config.get("dataset", {})
    precision = str(config.get("precision", "fp32")).lower()
    wandb_project = dpo_train_args.get("wandb_project") or dpo_train_args.get("report")
    fsdp_options = _parse_dpo_fsdp_options(dpo_train_args)
    if fsdp_options["enabled"] and fsdp_options["state_dict_type"] is not None:
        os.environ["FSDP_STATE_DICT_TYPE"] = str(fsdp_options["state_dict_type"])

    kwargs: Dict[str, Any] = {
        "learning_rate": float(dpo_train_args["learning_rate"]),
        "per_device_train_batch_size": int(dpo_train_args["batch_size"]),
        "per_device_eval_batch_size": int(dpo_train_args["eval_batch_size"]),
        "num_train_epochs": int(dpo_train_args["epochs"]),
        "logging_steps": int(dpo_train_args["log_steps"]),
        "eval_strategy": str(dpo_train_args.get("eval_strategy", "steps")),
        "eval_steps": int(dpo_train_args["eval_steps"]),
        "save_strategy": str(dpo_train_args.get("save_strategy", "steps")),
        "save_steps": int(dpo_train_args["save_steps"]),
        "fp16": precision == "fp16",
        "bf16": precision == "bf16",
        "gradient_accumulation_steps": int(
            dpo_train_args.get("gradient_accumulation", 1)
        ),
        "max_grad_norm": float(dpo_train_args.get("max_grad_norm", 1.0)),
        "warmup_steps": int(dpo_train_args.get("warmup_steps", 0)),
        "report_to": ["wandb"] if wandb_project else [],
        "run_name": dpo_train_args.get("run_name", "dpo"),
        "remove_unused_columns": False,
        "output_dir": dpo_train_args["save_dir"],
    }

    max_len = dataset_cfg.get("max_len")
    if max_len is not None:
        max_length = int(max_len)
        if max_length <= 0:
            raise ValueError("dataset.max_len must be > 0.")
        kwargs["max_length"] = max_length

    if "gradient_checkpointing" in dpo_train_args:
        kwargs["gradient_checkpointing"] = bool(
            dpo_train_args["gradient_checkpointing"]
        )

    optional_fields = (
        "save_total_limit",
        "load_best_model_at_end",
        "metric_for_best_model",
        "greater_is_better",
        "save_only_model",
    )
    for field_name in optional_fields:
        if field_name in dpo_train_args:
            kwargs[field_name] = dpo_train_args[field_name]

    if (
        fsdp_options["enabled"]
        and bool(kwargs.get("load_best_model_at_end", False))
        and bool(kwargs.get("save_only_model", False))
    ):
        print(
            "[DPO] Overriding save_only_model=False for FSDP with "
            "load_best_model_at_end to avoid checkpoint incompatibilities."
        )
        kwargs["save_only_model"] = False

    if fsdp_options["enabled"]:
        kwargs.update(fsdp_options["args"])

    return kwargs


def _maybe_init_wandb(config: Dict[str, Any]) -> None:
    dpo_train_args = config["dpo_training"]
    wandb_project = dpo_train_args.get("wandb_project") or dpo_train_args.get("report")
    if not wandb_project or not _is_main_process():
        return

    import wandb

    wandb.init(
        project=str(wandb_project),
        name=dpo_train_args.get("run_name", "dpo"),
        config=config,
    )


def _finalize_dpo_training(
    trainer: Any, cli_output_dir: str, dpo_train_args: Dict[str, Any]
) -> None:
    trainer.train()
    trainer.save_model(os.path.join(cli_output_dir, "final"))

    hub_model_id = dpo_train_args.get("hub_model_id")
    if hub_model_id and _is_main_process():
        print(f"\nPushing model to HuggingFace Hub: {hub_model_id}")
        trainer.model.push_to_hub(hub_model_id)
        print(f"Model uploaded successfully to: https://huggingface.co/{hub_model_id}")


def main_sft():
    """Main entry point for SFT training."""
    parser = argparse.ArgumentParser(description="Run SFT training")
    parser.add_argument("--config", type=str, default="config_sft.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    trainer = run_sft_training(config)

    if _is_main_process():
        sft_cfg = config.get("sft_training", {})
        hub_model_id = sft_cfg.get("hub_model_id")
        if hub_model_id:
            try:
                push = (
                    input(
                        f"\nDo you want to push the model to the Hub ({hub_model_id})? [y/N]: "
                    )
                    .strip()
                    .lower()
                )
                if push == "y":
                    print(f"Pushing model to {hub_model_id}...")
                    trainer.push_to_hub()
            except EOFError:
                pass
        else:
            print("No hub_model_id configured, skipping interactive push.")


def main_dpo():
    """Legacy DPO entry point."""
    raise NotImplementedError(
        "The generic train-dpo entry point is not wired in this checkout. "
        "Use train-beta-dpo or train-margin-dpo instead."
    )


def run_beta_dpo_training(config: Dict[str, Any], *, output_dir: str) -> None:
    """Run one Beta DPO training job from an in-memory config."""
    from .trainers.beta_dpo_trainer import BetaDPOConfig, BetaDPOTrainer

    policy, ref_model, tokenizer, policy_name = _load_policy_ref_and_tokenizer(config)
    train_ds, eval_ds = _build_hh_dpo_datasets(config, tokenizer, policy_name)

    beta_cfg = config.get("beta_dpo", {})
    training_args = BetaDPOConfig(
        **(config),
        beta=float(beta_cfg.get("beta", 0.1)),
        rho=float(beta_cfg.get("rho", 0.8)),
        alpha=float(beta_cfg.get("alpha", 1.0)),
        ema_momentum=float(beta_cfg.get("ema_momentum", 0.9)),
        beta_min=float(beta_cfg.get("beta_min", 1e-3)),
        sync_global_mask=bool(beta_cfg.get("sync_global_mask", True)),
    )

    _maybe_init_wandb(config)

    trainer = BetaDPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=None,
    )
    _finalize_dpo_training(trainer, output_dir, config["dpo_training"])


def run_margin_dpo_training(config: Dict[str, Any], *, output_dir: str) -> None:
    """Run one Margin DPO training job from an in-memory config."""
    from .trainers.margin_dpo_trainer import MarginDPOTrainer

    policy, ref_model, tokenizer, policy_name = _load_policy_ref_and_tokenizer(config)
    train_ds, eval_ds = _build_hh_dpo_datasets(config, tokenizer, policy_name)

    training_args = DPOConfig(**_build_common_dpo_config_kwargs(config))
    margin_cfg = config.get("margin_log", {})

    _maybe_init_wandb(config)

    trainer = MarginDPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        margin_log_path=str(margin_cfg.get("log_dir", "logs/margins")),
        processing_class=None,
    )
    _finalize_dpo_training(trainer, output_dir, config["dpo_training"])


def main_beta_dpo():
    """Main entry point for Beta DPO training."""
    parser = _build_dpo_parser(
        description="Run Beta DPO training",
        default_config="config_beta_dpo.yaml",
        default_output_dir="beta_dpo_out",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    run_beta_dpo_training(config, output_dir=args.output_dir)


def main_margin_dpo():
    """Main entry point for Margin DPO training."""
    parser = _build_dpo_parser(
        description="Run Margin DPO training",
        default_config="config_margin_dpo.yaml",
        default_output_dir="margin_dpo_out",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    run_margin_dpo_training(config, output_dir=args.output_dir)
