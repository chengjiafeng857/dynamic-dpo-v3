"""CLI entry points for DPO and SFT training."""

import argparse
import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig

from .config.loader import load_yaml
from .data.hh_dataset import (
    apply_chat_template_to_dataset,
    build_HH_dataset,
    load_generated_dataset_from_config,
)
from .trainers.beta_dpo import BetaDPOConfig, BetaDPOTrainer
from .trainers.dynamic_beta_dpo import DynamicBetaDPOConfig, DynamicBetaDPOTrainer
from .trainers.sft_trainer import run_sft_training
from .utils.debug import log_dpo_debug_samples


def main_dpo():
    """Main entry point for DPO training."""
    parser = argparse.ArgumentParser(description="Run Dynamic Beta DPO training")
    parser.add_argument("--config", type=str, default="config_dpo.yaml")
    parser.add_argument("--output_dir", type=str, default="trl_dynamic_beta_out")
    args = parser.parse_args()

    config = load_yaml(args.config)

    # Load policy, ref and tokenizer
    policy_name = config["policy_name"]
    ref_name = config["ref_name"]

    tok = AutoTokenizer.from_pretrained(policy_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    policy = AutoModelForCausalLM.from_pretrained(policy_name)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_name)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    # Load dataset
    dataset_name = config["dataset"]["dataset_name"]
    dataset_cfg = config["dataset"]
    raw_ds = load_dataset(dataset_name, split=dataset_cfg["subset"])

    if bool(dataset_cfg.get("generated_data", False)):
        hh_ds = load_generated_dataset_from_config(config)
    else:
        hh_ds = build_HH_dataset(raw_ds)

    if bool(dataset_cfg.get("chat_template", False)):
        hh_ds = apply_chat_template_to_dataset(hh_ds, tok)

    # Split train/val
    val_ratio = float(config["dataset"]["val_ratio"])
    seed = int(config["dataset"]["seed"])
    split = hh_ds.train_test_split(test_size=val_ratio, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    debug_train_ds = train_ds
    try:
        debug_train_ds = train_ds.with_format("python")
    except Exception:
        pass

    # Training args
    prec = config["precision"].lower()
    fp16 = prec == "fp16"
    bf16 = prec == "bf16"

    dpo_train_args = config["dpo_training"]
    training_args = DPOConfig(
        learning_rate=float(dpo_train_args["learning_rate"]),
        per_device_train_batch_size=int(dpo_train_args["batch_size"]),
        per_device_eval_batch_size=int(dpo_train_args["eval_batch_size"]),
        num_train_epochs=int(dpo_train_args["epochs"]),
        logging_steps=int(dpo_train_args["log_steps"]),
        eval_strategy="steps",
        eval_steps=int(dpo_train_args["eval_steps"]),
        save_strategy="steps",
        save_steps=int(dpo_train_args["save_steps"]),
        fp16=fp16,
        bf16=bf16,
        gradient_accumulation_steps=int(dpo_train_args["gradient_accumulation"]),
        max_grad_norm=float(dpo_train_args["max_grad_norm"]),
        warmup_steps=int(dpo_train_args["warmup_steps"]),
        report_to=["wandb"] if dpo_train_args.get("report") else [],
        run_name=dpo_train_args["run_name"],
        remove_unused_columns=False,
        output_dir=dpo_train_args["save_dir"],
    )

    # Dynamic-beta config
    risk = config["risk_test"]
    beta_up = config["beta_update"]
    margin_log = config["margin_log"]

    dyn_cfg = DynamicBetaDPOConfig(
        delta=float(risk["delta"]),
        momentum=float(risk["lambda"]),
        warmup_steps=int(risk["beta_warmup"]),
        beta_0=float(beta_up["beta_0"]),
        alpha=float(beta_up["alpha"]),
        gamma=float(beta_up["gamma"]),
        beta_min=float(beta_up["beta_min"]),
        beta_max=float(beta_up["beta_max"]),
        log_margins=True,
        log_dir=str(margin_log["log_dir"]),
        jsonl_sample_size=int(margin_log["jsonl_sample_size"]),
        save_per_rank=bool(margin_log["save_per_rank"]),
    )

    # Optional wandb init
    wandb_project = dpo_train_args.get("wandb_project") or dpo_train_args.get("report")
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
                name=dpo_train_args["run_name"],
                config=config,
            )

    trainer = DynamicBetaDPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dynamic_cfg=dyn_cfg,
        processing_class=None,
    )

    try:
        log_dpo_debug_samples(trainer, raw_dataset=debug_train_ds)
    except Exception:
        pass

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))


def main_sft():
    """Main entry point for SFT training."""
    parser = argparse.ArgumentParser(description="Run SFT training")
    parser.add_argument("--config", type=str, default="config_sft.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    trainer = run_sft_training(config)

    # Interactive hub push (main process only)
    import torch.distributed as dist

    is_main = (
        (not dist.is_available())
        or (not dist.is_initialized())
        or (dist.get_rank() == 0)
    )

    if is_main:
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


def main_beta_dpo():
    """Main entry point for Beta DPO training."""
    parser = argparse.ArgumentParser(description="Run Beta DPO training")
    parser.add_argument("--config", type=str, default="config_beta_dpo.yaml")
    parser.add_argument("--output_dir", type=str, default="beta_dpo_out")
    args = parser.parse_args()

    config = load_yaml(args.config)

    # Load policy, ref and tokenizer
    policy_name = config["policy_name"]
    ref_name = config["ref_name"]

    tok = AutoTokenizer.from_pretrained(policy_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    policy = AutoModelForCausalLM.from_pretrained(policy_name)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_name)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    # Load dataset
    dataset_cfg = config["dataset"]
    raw_ds = load_dataset(dataset_cfg["dataset_name"], split=dataset_cfg["subset"])

    if bool(dataset_cfg.get("generated_data", False)):
        hh_ds = load_generated_dataset_from_config(config)
    else:
        hh_ds = build_HH_dataset(raw_ds)

    if bool(dataset_cfg.get("chat_template", False)):
        hh_ds = apply_chat_template_to_dataset(hh_ds, tok)

    # Split train/val
    val_ratio = float(config["dataset"]["val_ratio"])
    seed = int(config["dataset"]["seed"])
    split = hh_ds.train_test_split(test_size=val_ratio, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    # Training args
    prec = config["precision"].lower()
    fp16 = prec == "fp16"
    bf16 = prec == "bf16"

    dpo_train_args = config["dpo_training"]
    training_args = DPOConfig(
        learning_rate=float(dpo_train_args["learning_rate"]),
        per_device_train_batch_size=int(dpo_train_args["batch_size"]),
        per_device_eval_batch_size=int(dpo_train_args["eval_batch_size"]),
        num_train_epochs=int(dpo_train_args["epochs"]),
        logging_steps=int(dpo_train_args["log_steps"]),
        eval_strategy="steps",
        eval_steps=int(dpo_train_args["eval_steps"]),
        save_strategy="steps",
        save_steps=int(dpo_train_args["save_steps"]),
        fp16=fp16,
        bf16=bf16,
        gradient_accumulation_steps=int(dpo_train_args.get("gradient_accumulation", 1)),
        max_grad_norm=float(dpo_train_args.get("max_grad_norm", 1.0)),
        warmup_steps=int(dpo_train_args.get("warmup_steps", 0)),
        report_to=["wandb"] if dpo_train_args.get("report") else [],
        run_name=dpo_train_args.get("run_name", "beta-dpo"),
        remove_unused_columns=False,
        output_dir=dpo_train_args["save_dir"],
    )

    # Beta DPO config
    beta_cfg_dict = config.get("beta_dpo", {})
    beta_cfg = BetaDPOConfig(
        beta_0=float(beta_cfg_dict.get("beta_0", 0.1)),
        m=float(beta_cfg_dict.get("m", 0.9)),
        rho=float(beta_cfg_dict.get("rho", 0.8)),
        alpha=float(beta_cfg_dict.get("alpha", 0.6)),
        min_beta=float(beta_cfg_dict.get("min_beta", 1e-3)),
        eps=float(beta_cfg_dict.get("eps", 1e-6)),
        log_margins=bool(beta_cfg_dict.get("log_margins", True)),
        log_dir=str(beta_cfg_dict.get("log_dir", "logs/beta_dpo_margins")),
    )

    # Optional wandb init
    wandb_project = dpo_train_args.get("wandb_project") or dpo_train_args.get("report")
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
                name=dpo_train_args.get("run_name", "beta-dpo"),
                config=config,
            )

    trainer = BetaDPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        beta_dpo_cfg=beta_cfg,
        processing_class=None,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))

    # Push to HuggingFace Hub if configured
    hub_model_id = dpo_train_args.get("hub_model_id")
    if hub_model_id:
        import torch.distributed as dist

        is_main = (
            (not dist.is_available())
            or (not dist.is_initialized())
            or (dist.get_rank() == 0)
        )
        if is_main:
            print(f"\nPushing model to HuggingFace Hub: {hub_model_id}")
            trainer.model.push_to_hub(hub_model_id)
            print(
                f"Model uploaded successfully to: https://huggingface.co/{hub_model_id}"
            )
