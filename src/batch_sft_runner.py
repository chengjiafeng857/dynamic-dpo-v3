"""Batch SFT orchestration: train, push, clean outputs, delete caches."""

import argparse
import copy
import gc
import shutil
from pathlib import Path
from typing import Any, List, TypedDict

from .batch_runner_utils import (
    clear_cuda_memory,
    delete_hf_cache_entries,
    distributed_barrier,
    finish_wandb_run,
    gather_error_messages,
    is_main_process,
    resolve_cache_cleanup_flags,
    resolve_config_path,
)
from .config.loader import load_yaml
from .trainers.sft_trainer import run_sft_training


class SFTBatchRunPlan(TypedDict):
    base_config_path: str
    dataset_slug: str
    model_slug: str
    policy_name: str
    hub_model_id: str
    config: dict[str, Any]


def _build_run_config(
    base_config: dict[str, Any],
    *,
    dataset_slug: str,
    model_entry: dict[str, Any],
    hf_username: str,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)

    model_slug = str(model_entry["model_slug"])
    policy_name = str(model_entry["policy_name"])
    hub_model_id = f"{hf_username}/{dataset_slug}-{model_slug}-sft"
    run_name = f"{dataset_slug}-{model_slug}-sft"

    config["policy_name"] = policy_name
    if "ref_name" in config:
        config["ref_name"] = str(model_entry.get("ref_name", policy_name))

    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg["chat_template_name"] = str(model_entry["chat_template_name"])

    sft_cfg = config.setdefault("sft_training", {})
    sft_cfg["run_name"] = run_name
    sft_cfg["save_dir"] = str(Path("batch_sft_runs") / run_name)
    sft_cfg["hub_model_id"] = hub_model_id
    sft_cfg["push_to_hub"] = False

    fsdp_cfg = sft_cfg.get("fsdp")
    if isinstance(fsdp_cfg, dict) and bool(fsdp_cfg.get("enabled", False)):
        fsdp_cfg["transformer_layer_cls_to_wrap"] = str(
            model_entry["fsdp_transformer_layer_cls_to_wrap"]
        )

    return config


def build_run_matrix(batch_config: dict[str, Any], *, config_dir: Path) -> List[SFTBatchRunPlan]:
    """Expand the batch config into concrete per-run configs."""
    execution_order = str(batch_config.get("execution_order", "model_major"))
    if execution_order not in {"dataset_major", "model_major"}:
        raise ValueError(
            "execution_order must be either 'dataset_major' or 'model_major'."
        )

    hf_username = str(batch_config["hf_username"])
    datasets = batch_config.get("datasets", [])
    models = batch_config.get("models", [])
    dataset_specs: List[dict[str, Any]] = []
    for dataset_entry in datasets:
        base_config_path = resolve_config_path(
            str(dataset_entry["base_config"]), config_dir
        )
        dataset_specs.append(
            {
                "base_config_path": str(base_config_path),
                "base_config": load_yaml(str(base_config_path)),
                "dataset_slug": str(dataset_entry["dataset_slug"]),
            }
        )

    if execution_order == "dataset_major":
        pairings = [
            (dataset_spec, model_entry)
            for dataset_spec in dataset_specs
            for model_entry in models
        ]
    else:
        pairings = [
            (dataset_spec, model_entry)
            for model_entry in models
            for dataset_spec in dataset_specs
        ]

    run_plans: List[SFTBatchRunPlan] = []
    for dataset_spec, model_entry in pairings:
        run_config = _build_run_config(
            dataset_spec["base_config"],
            dataset_slug=dataset_spec["dataset_slug"],
            model_entry=model_entry,
            hf_username=hf_username,
        )
        run_plans.append(
            {
                "base_config_path": dataset_spec["base_config_path"],
                "dataset_slug": dataset_spec["dataset_slug"],
                "model_slug": str(model_entry["model_slug"]),
                "policy_name": str(model_entry["policy_name"]),
                "hub_model_id": str(run_config["sft_training"]["hub_model_id"]),
                "config": run_config,
            }
        )
    return run_plans


def _build_run_label(run_plan: SFTBatchRunPlan, *, index: int, total_runs: int) -> str:
    return f"{run_plan['dataset_slug']} / {run_plan['model_slug']} ({index}/{total_runs})"


def _run_training_for_plan(run_plan: SFTBatchRunPlan) -> tuple[Any, str | None]:
    try:
        return run_sft_training(run_plan["config"]), None
    except Exception as exc:
        return None, str(exc)


def _push_model_for_run(trainer: Any) -> str | None:
    try:
        trainer.push_to_hub()
    except Exception as exc:
        return str(exc)
    return None


def _phase_error_message(error: Any) -> str | None:
    return error if isinstance(error, str) and error else None


def cleanup_run_artifacts(
    *,
    trainer: Any,
    run_config: dict[str, Any],
    cleanup_config: dict[str, Any],
) -> str | None:
    finish_wandb_run()

    if trainer is not None:
        try:
            trainer.model = None
        except Exception:
            pass
    gc.collect()
    clear_cuda_memory()

    if bool(cleanup_config.get("delete_run_output", True)):
        save_dir = Path(str(run_config["sft_training"]["save_dir"]))
        try:
            shutil.rmtree(save_dir, ignore_errors=True)
        except Exception as exc:
            return str(exc)

    delete_policy_model_cache, delete_dataset_cache, _ = resolve_cache_cleanup_flags(
        cleanup_config
    )
    repo_ids: list[str] = []

    if delete_policy_model_cache:
        policy_name = str(run_config["policy_name"])
        ref_name = run_config.get("ref_name")
        if ref_name is not None and str(ref_name) == policy_name:
            print(
                "[SFT-BATCH] Skipping policy model cache cleanup because "
                "ref_name matches policy_name; Hugging Face cache is shared by repo_id."
            )
        else:
            repo_ids.append(policy_name)

    if delete_dataset_cache:
        repo_ids.append(str(run_config["dataset"]["dataset_name"]))

    if repo_ids:
        delete_hf_cache_entries(repo_ids, log_prefix="SFT-BATCH")
    return None


def cleanup_successful_run(
    *,
    trainer: Any,
    run_config: dict[str, Any],
    cleanup_config: dict[str, Any],
) -> str | None:
    return cleanup_run_artifacts(
        trainer=trainer,
        run_config=run_config,
        cleanup_config=cleanup_config,
    )


def _has_future_same_policy(
    run_plan: SFTBatchRunPlan,
    run_plans: List[SFTBatchRunPlan],
    *,
    index: int,
) -> bool:
    return any(
        future_run_plan["policy_name"] == run_plan["policy_name"]
        for future_run_plan in run_plans[index:]
    )


def delete_completed_policy_cache_if_last_use(
    *,
    run_plan: SFTBatchRunPlan,
    run_plans: List[SFTBatchRunPlan],
    index: int,
    cleanup_config: dict[str, Any],
) -> bool:
    _, _, delete_completed_policy_model_cache = resolve_cache_cleanup_flags(
        cleanup_config
    )
    if (
        not delete_completed_policy_model_cache
        or _has_future_same_policy(run_plan, run_plans, index=index)
    ):
        return False

    cleanup_completed_policy_cache(
        completed_policy_name=str(run_plan["policy_name"]),
        cleanup_config=cleanup_config,
    )
    return True


def cleanup_completed_policy_cache(
    *, completed_policy_name: str, cleanup_config: dict[str, Any]
) -> None:
    _, _, delete_completed_policy_model_cache = resolve_cache_cleanup_flags(
        cleanup_config
    )
    if not delete_completed_policy_model_cache:
        return
    delete_hf_cache_entries([completed_policy_name], log_prefix="SFT-BATCH")


def run_batch_sft(batch_config: dict[str, Any], *, config_dir: Path) -> int:
    """Run the full SFT matrix sequentially and fail fast on errors."""
    cleanup_config = batch_config.get("cleanup", {})
    run_plans = build_run_matrix(batch_config, config_dir=config_dir)
    main_process = is_main_process()

    for index, run_plan in enumerate(run_plans, start=1):
        run_label = _build_run_label(run_plan, index=index, total_runs=len(run_plans))
        if main_process:
            print(
                f"[SFT-BATCH] Starting {run_label} "
                f"from {run_plan['base_config_path']} -> {run_plan['hub_model_id']}"
            )

        trainer, train_error = _run_training_for_plan(run_plan)
        train_errors = gather_error_messages(_phase_error_message(train_error))
        if train_errors:
            if main_process:
                print(f"[SFT-BATCH] Failed {run_label}: {train_errors[0]}")
            distributed_barrier()
            return 1

        push_error = None
        if main_process:
            print(f"[SFT-BATCH] Training complete for {run_label}")
            push_error = _push_model_for_run(trainer)
            if push_error is None:
                print(f"[SFT-BATCH] Upload complete for {run_label}")

        push_errors = gather_error_messages(_phase_error_message(push_error))
        if push_errors:
            if main_process:
                print(f"[SFT-BATCH] Failed {run_label}: {push_errors[0]}")
            distributed_barrier()
            return 1

        cleanup_error = None
        deleted_completed_policy_cache = False
        if main_process:
            cleanup_error = cleanup_successful_run(
                trainer=trainer,
                run_config=run_plan["config"],
                cleanup_config=cleanup_config,
            )
            if _phase_error_message(cleanup_error) is None:
                print(f"[SFT-BATCH] Cleanup complete for {run_label}")
                deleted_completed_policy_cache = delete_completed_policy_cache_if_last_use(
                    run_plan=run_plan,
                    run_plans=run_plans,
                    index=index,
                    cleanup_config=cleanup_config,
                )

        cleanup_errors = gather_error_messages(_phase_error_message(cleanup_error))
        if cleanup_errors:
            if main_process:
                print(f"[SFT-BATCH] Failed {run_label}: {cleanup_errors[0]}")
            distributed_barrier()
            return 1

        if main_process and deleted_completed_policy_cache:
            print(
                "[SFT-BATCH] Deleted completed policy model cache for "
                f"{run_plan['policy_name']}"
            )

        barrier_error = distributed_barrier()
        if isinstance(barrier_error, str) and barrier_error:
            if main_process:
                print(f"[SFT-BATCH] Failed {run_label}: {barrier_error}")
            return 1

    if main_process:
        print(f"[SFT-BATCH] All {len(run_plans)} SFT runs completed successfully.")
    return 0


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for the batch SFT runner."""
    parser = argparse.ArgumentParser(description="Run the six-run SFT matrix")
    parser.add_argument("--config", type=str, default="config_sft_batch.yaml")
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=-1)
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    batch_config = load_yaml(str(config_path))
    return run_batch_sft(batch_config, config_dir=config_path.parent)


if __name__ == "__main__":
    raise SystemExit(main())
