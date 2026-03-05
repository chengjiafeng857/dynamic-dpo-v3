"""Run a single SFT job and optionally push the trained model to Hugging Face Hub."""

import argparse
from typing import Any, Dict, List

from src.config.loader import load_yaml
from src.trainers.sft_trainer import run_sft_training


def _is_main_process() -> bool:
    try:
        import torch.distributed as dist

        return (
            (not dist.is_available())
            or (not dist.is_initialized())
            or (dist.get_rank() == 0)
        )
    except Exception:
        return True


def _distributed_barrier() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        return


def _require_simpo_runtime_settings(config: Dict[str, Any]) -> None:
    sft_cfg = config.get("sft_training", {})
    fsdp_cfg = sft_cfg.get("fsdp")

    errors: List[str] = []
    if not bool(sft_cfg.get("packing", False)):
        errors.append("sft_training.packing must be true.")

    packing_strategy = str(sft_cfg.get("packing_strategy", "bfd")).lower()
    if packing_strategy != "bfd":
        errors.append("sft_training.packing_strategy must be 'bfd'.")

    attn_implementation = str(sft_cfg.get("attn_implementation", "")).lower()
    if attn_implementation not in {"flash_attention_2", "flash_attention_3"}:
        errors.append(
            "sft_training.attn_implementation must be flash_attention_2 or flash_attention_3."
        )

    if not isinstance(fsdp_cfg, dict) or not bool(fsdp_cfg.get("enabled", False)):
        errors.append("sft_training.fsdp.enabled must be true.")
    else:
        fsdp_mode_tokens = {
            token.strip().lower() for token in str(fsdp_cfg.get("mode", "")).split() if token.strip()
        }
        if "full_shard" not in fsdp_mode_tokens:
            errors.append("sft_training.fsdp.mode must include full_shard.")
        if "auto_wrap" not in fsdp_mode_tokens:
            errors.append("sft_training.fsdp.mode must include auto_wrap.")

    if sft_cfg.get("max_length") is None:
        errors.append("sft_training.max_length must be configured.")

    if errors:
        raise ValueError("SIMPO runtime setting validation failed:\n- " + "\n- ".join(errors))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one SFT job and optionally push to Hub.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=-1)
    parser.add_argument(
        "--push",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to push trained weights to the Hub.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Optional Hub repo override for this run.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name override for this run.",
    )
    parser.add_argument(
        "--require-simpo-settings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate packing+bfd+flash attention+FSDP runtime settings before training.",
    )
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    sft_cfg = config.setdefault("sft_training", {})

    if args.run_name:
        sft_cfg["run_name"] = str(args.run_name)
    if args.hub_model_id:
        sft_cfg["hub_model_id"] = str(args.hub_model_id)
    sft_cfg["push_to_hub"] = False

    if args.require_simpo_settings:
        _require_simpo_runtime_settings(config)

    trainer = run_sft_training(config)

    if args.push:
        hub_model_id = sft_cfg.get("hub_model_id")
        if not hub_model_id:
            raise ValueError(
                "Cannot push without sft_training.hub_model_id. Set it in config or pass --hub-model-id."
            )
        if _is_main_process():
            print(f"[SFT-SINGLE] Uploading model to {hub_model_id}")
            trainer.push_to_hub()
            print(f"[SFT-SINGLE] Upload complete: {hub_model_id}")
    _distributed_barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
