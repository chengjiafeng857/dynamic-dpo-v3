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
from .trainers.sft_trainer import run_sft_training




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


