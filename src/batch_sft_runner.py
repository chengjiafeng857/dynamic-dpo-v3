"""Batch SFT orchestration for a fixed dataset/model run matrix."""

import argparse
import copy
import gc
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config.loader import load_yaml
from .trainers.sft_trainer import run_sft_training

try:
    from huggingface_hub import scan_cache_dir
except ImportError:  # pragma: no cover - transformers normally brings this in.
    scan_cache_dir = None


def _resolve_config_path(config_path: str, base_dir: Path) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _build_run_config(
    base_config: Dict[str, Any],
    *,
    dataset_slug: str,
    model_entry: Dict[str, Any],
    hf_username: str,
) -> Dict[str, Any]:
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


def build_run_matrix(batch_config: Dict[str, Any], *, config_dir: Path) -> List[Dict[str, Any]]:
    """Expand the batch config into concrete per-run configs."""
    execution_order = str(batch_config.get("execution_order", "model_major"))
    if execution_order not in {"dataset_major", "model_major"}:
        raise ValueError(
            "execution_order must be either 'dataset_major' or 'model_major'."
        )

    hf_username = str(batch_config["hf_username"])
    datasets = batch_config.get("datasets", [])
    models = batch_config.get("models", [])
    dataset_specs: List[Dict[str, Any]] = []
    for dataset_entry in datasets:
        base_config_path = _resolve_config_path(str(dataset_entry["base_config"]), config_dir)
        dataset_specs.append(
            {
                "base_config_path": str(base_config_path),
                "base_config": load_yaml(str(base_config_path)),
                "dataset_slug": str(dataset_entry["dataset_slug"]),
            }
        )

    run_plans: List[Dict[str, Any]] = []
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


def _finish_wandb_run() -> None:
    try:
        import wandb

        wandb.finish()
    except Exception:
        return


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


def _gather_error_messages(local_error: str | None) -> List[str]:
    try:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return [local_error] if local_error else []

        gathered: List[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_error)
        return [message for message in gathered if message]
    except Exception:
        return [local_error] if local_error else []


def _clear_cuda_memory() -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    except Exception:
        return


def _collect_revision_hashes(cache_info: Any, repo_ids: Iterable[str]) -> List[str]:
    repo_id_set = {str(repo_id) for repo_id in repo_ids if repo_id}
    repos = getattr(cache_info, "repos", None)
    if repos is None:
        repos = getattr(cache_info, "cached_repos", [])

    revision_hashes: List[str] = []
    for repo in repos or []:
        if getattr(repo, "repo_id", None) not in repo_id_set:
            continue
        for revision in getattr(repo, "revisions", []) or []:
            commit_hash = getattr(revision, "commit_hash", None)
            if commit_hash:
                revision_hashes.append(str(commit_hash))
    return revision_hashes


def _delete_hf_cache_entries(repo_ids: Iterable[str]) -> None:
    if scan_cache_dir is None:
        print("[SFT-BATCH] Hugging Face cache scan unavailable; skipping cache cleanup.")
        return

    try:
        cache_info = scan_cache_dir()
        revision_hashes = _collect_revision_hashes(cache_info, repo_ids)
        if not revision_hashes:
            return
        delete_strategy = cache_info.delete_revisions(*revision_hashes)
        delete_strategy.execute()
    except Exception as exc:
        print(f"[SFT-BATCH] Warning: failed to clean Hugging Face cache: {exc}")


def _resolve_cache_cleanup_flags(cleanup_config: Dict[str, Any]) -> tuple[bool, bool, bool]:
    legacy_delete_hf_cache = bool(cleanup_config.get("delete_hf_download_cache", False))
    delete_policy_model_cache = bool(
        cleanup_config.get("delete_policy_model_cache", legacy_delete_hf_cache)
    )
    delete_dataset_cache = bool(
        cleanup_config.get("delete_dataset_cache", legacy_delete_hf_cache)
    )
    delete_completed_policy_model_cache = bool(
        cleanup_config.get("delete_completed_policy_model_cache", False)
    )
    return (
        delete_policy_model_cache,
        delete_dataset_cache,
        delete_completed_policy_model_cache,
    )


def cleanup_run_artifacts(
    *,
    trainer: Any,
    run_config: Dict[str, Any],
    cleanup_config: Dict[str, Any],
) -> None:
    """Release process memory and remove local/HF cache artifacts after a successful run."""
    _finish_wandb_run()

    if trainer is not None:
        try:
            trainer.model = None
        except Exception:
            pass
    gc.collect()
    _clear_cuda_memory()

    if bool(cleanup_config.get("delete_run_output", True)):
        save_dir = Path(str(run_config["sft_training"]["save_dir"]))
        shutil.rmtree(save_dir, ignore_errors=True)

    (
        delete_policy_model_cache,
        delete_dataset_cache,
        _,
    ) = _resolve_cache_cleanup_flags(cleanup_config)
    repo_ids: List[str] = []

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
        _delete_hf_cache_entries(repo_ids)


def cleanup_completed_policy_cache(
    *,
    completed_policy_name: str,
    cleanup_config: Dict[str, Any],
) -> None:
    """Delete a model cache only after its last run in the batch has completed."""
    (
        _delete_policy_model_cache,
        _delete_dataset_cache,
        delete_completed_policy_model_cache,
    ) = _resolve_cache_cleanup_flags(cleanup_config)
    if not delete_completed_policy_model_cache:
        return

    _delete_hf_cache_entries([completed_policy_name])


def run_batch_sft(batch_config: Dict[str, Any], *, config_dir: Path) -> int:
    """Run the full SFT matrix sequentially and fail fast on errors."""
    cleanup_config = batch_config.get("cleanup", {})
    run_plans = build_run_matrix(batch_config, config_dir=config_dir)
    is_main_process = _is_main_process()

    for index, run_plan in enumerate(run_plans, start=1):
        run_config = run_plan["config"]
        run_label = (
            f"{run_plan['dataset_slug']} / {run_plan['model_slug']} "
            f"({index}/{len(run_plans)})"
        )
        if is_main_process:
            print(
                f"[SFT-BATCH] Starting {run_label} "
                f"from {run_plan['base_config_path']} -> {run_plan['hub_model_id']}"
            )
        trainer = None
        train_error = None
        try:
            trainer = run_sft_training(run_config)
        except Exception as exc:
            train_error = str(exc)

        train_errors = _gather_error_messages(train_error)
        if train_errors:
            if is_main_process:
                print(f"[SFT-BATCH] Failed {run_label}: {train_errors[0]}")
            _distributed_barrier()
            return 1

        if is_main_process:
            print(f"[SFT-BATCH] Training complete for {run_label}")

        push_error = None
        if is_main_process:
            try:
                trainer.push_to_hub()
                print(f"[SFT-BATCH] Upload complete for {run_label}")
            except Exception as exc:
                push_error = str(exc)

        push_errors = _gather_error_messages(push_error)
        if push_errors:
            if is_main_process:
                print(f"[SFT-BATCH] Failed {run_label}: {push_errors[0]}")
            _distributed_barrier()
            return 1

        if is_main_process:
            cleanup_run_artifacts(
                trainer=trainer,
                run_config=run_config,
                cleanup_config=cleanup_config,
            )
            print(f"[SFT-BATCH] Cleanup complete for {run_label}")
            has_future_same_policy = any(
                future_run_plan["policy_name"] == run_plan["policy_name"]
                for future_run_plan in run_plans[index:]
            )
            if not has_future_same_policy:
                cleanup_completed_policy_cache(
                    completed_policy_name=str(run_plan["policy_name"]),
                    cleanup_config=cleanup_config,
                )
                if bool(cleanup_config.get("delete_completed_policy_model_cache", False)):
                    print(
                        "[SFT-BATCH] Deleted completed policy model cache for "
                        f"{run_plan['policy_name']}"
                    )
        _distributed_barrier()

    if is_main_process:
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
