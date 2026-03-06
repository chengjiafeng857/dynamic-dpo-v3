"""Batch DPO orchestration for an explicit HH run matrix."""

import argparse
import copy
import errno
import gc
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .cli import run_beta_dpo_training, run_margin_dpo_training
from .config.loader import load_yaml

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
    trainer_type: str,
    trainer_slug: str,
    run_entry: Dict[str, Any],
    hf_username: str,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)

    dataset_slug = str(run_entry["dataset_slug"])
    model_slug = str(run_entry["model_slug"])
    policy_name = str(run_entry["policy_name"])
    ref_name = str(run_entry.get("ref_name", policy_name))
    run_name = f"{dataset_slug}-{model_slug}-{trainer_slug}"

    config["policy_name"] = policy_name
    config["ref_name"] = ref_name

    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg["data_dir"] = str(run_entry["dataset_data_dir"])
    dataset_cfg["chat_template"] = True
    dataset_cfg["chat_template_name"] = str(run_entry["chat_template_name"])

    dpo_cfg = config.setdefault("dpo_training", {})
    dpo_cfg["run_name"] = run_name
    dpo_cfg["save_dir"] = str(Path("batch_dpo_runs") / run_name)
    dpo_cfg["hub_model_id"] = f"{hf_username}/{run_name}"

    fsdp_cfg = dpo_cfg.get("fsdp")
    if isinstance(fsdp_cfg, dict) and bool(fsdp_cfg.get("enabled", False)):
        fsdp_cfg["transformer_layer_cls_to_wrap"] = str(
            run_entry["fsdp_transformer_layer_cls_to_wrap"]
        )

    if trainer_type == "margin":
        margin_cfg = config.setdefault("margin_log", {})
        margin_cfg["log_dir"] = str(Path("logs") / f"{run_name}-margins")
        margin_cfg["archive_after_run"] = True
        margin_cfg["archive_backend"] = "wandb_then_hf"
        margin_cfg["delete_local_after_archive"] = True
        margin_cfg["wandb_artifact_name"] = f"{run_name}-margin-logs"
        margin_cfg["hf_dataset_repo_id"] = f"{hf_username}/{run_name}-margin-logs"
        margin_cfg["hf_dataset_private"] = False

    return config


def build_run_matrix(batch_config: Dict[str, Any], *, config_dir: Path) -> List[Dict[str, Any]]:
    """Expand the batch config into concrete per-run DPO jobs."""
    execution_order = str(batch_config.get("execution_order", "run_major"))
    if execution_order not in {"run_major", "trainer_major"}:
        raise ValueError(
            "execution_order must be either 'run_major' or 'trainer_major'."
        )

    hf_username = str(batch_config["hf_username"])
    trainer_specs: List[Dict[str, Any]] = []
    for trainer_entry in batch_config.get("trainers", []):
        trainer_type = str(trainer_entry["trainer_type"])
        if trainer_type not in {"beta", "margin"}:
            raise ValueError("trainer_type must be either 'beta' or 'margin'.")
        base_config_path = _resolve_config_path(str(trainer_entry["base_config"]), config_dir)
        trainer_specs.append(
            {
                "trainer_type": trainer_type,
                "trainer_slug": str(trainer_entry["trainer_slug"]),
                "base_config_path": str(base_config_path),
                "base_config": load_yaml(str(base_config_path)),
            }
        )

    run_entries = batch_config.get("runs", [])
    if execution_order == "run_major":
        pairings = [
            (run_entry, trainer_spec)
            for run_entry in run_entries
            for trainer_spec in trainer_specs
        ]
    else:
        pairings = [
            (run_entry, trainer_spec)
            for trainer_spec in trainer_specs
            for run_entry in run_entries
        ]

    run_plans: List[Dict[str, Any]] = []
    for run_entry, trainer_spec in pairings:
        run_config = _build_run_config(
            trainer_spec["base_config"],
            trainer_type=trainer_spec["trainer_type"],
            trainer_slug=trainer_spec["trainer_slug"],
            run_entry=run_entry,
            hf_username=hf_username,
        )
        run_name = str(run_config["dpo_training"]["run_name"])
        run_plans.append(
            {
                "trainer_type": trainer_spec["trainer_type"],
                "trainer_slug": trainer_spec["trainer_slug"],
                "base_config_path": trainer_spec["base_config_path"],
                "dataset_slug": str(run_entry["dataset_slug"]),
                "model_slug": str(run_entry["model_slug"]),
                "policy_name": str(run_config["policy_name"]),
                "ref_name": str(run_config["ref_name"]),
                "hub_model_id": str(run_config["dpo_training"]["hub_model_id"]),
                "output_dir": str(Path("batch_dpo_outputs") / run_name),
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


def _distributed_barrier() -> str | None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        return "Distributed synchronization failed at barrier."
    return None


def _gather_error_messages(local_error: str | None) -> List[str]:
    try:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return [local_error] if local_error else []

        gathered: List[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_error)
        return [message for message in gathered if message]
    except Exception as exc:
        messages = [local_error] if local_error else []
        messages.append(
            "Distributed synchronization failed while collecting errors: "
            f"{exc}"
        )
        return messages


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
        print("[DPO-BATCH] Hugging Face cache scan unavailable; skipping cache cleanup.")
        return

    try:
        cache_info = scan_cache_dir()
        revision_hashes = _collect_revision_hashes(cache_info, repo_ids)
        if not revision_hashes:
            return
        delete_strategy = cache_info.delete_revisions(*revision_hashes)
        delete_strategy.execute()
    except Exception as exc:
        print(f"[DPO-BATCH] Warning: failed to clean Hugging Face cache: {exc}")


def _format_run_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if isinstance(exc, OSError):
        if exc.errno == errno.ENOSPC or "no space left on device" in lowered:
            return f"Disk full: {message}"
        if exc.errno == errno.EDQUOT or "disk quota exceeded" in lowered:
            return f"Disk quota exceeded: {message}"
    if exc.__class__.__name__ == "OutOfMemoryError" or "out of memory" in lowered:
        return f"OOM: {message}"
    return message


def _archive_margin_logs_to_wandb(
    *,
    log_dir: Path,
    artifact_name: str,
) -> bool:
    try:
        import wandb
    except Exception as exc:
        print(f"[DPO-BATCH] Warning: W&B unavailable for margin log archival: {exc}")
        return False

    if wandb.run is None:
        print("[DPO-BATCH] Warning: no active W&B run for margin log archival.")
        return False

    try:
        artifact = wandb.Artifact(artifact_name, type="dataset")
        artifact.add_dir(str(log_dir), name=log_dir.name)
        logged_artifact = wandb.run.log_artifact(artifact)
        wait = getattr(logged_artifact, "wait", None)
        if callable(wait):
            wait()
        return True
    except Exception as exc:
        print(f"[DPO-BATCH] Warning: failed to archive margin logs to W&B: {exc}")
        return False


def _archive_margin_logs_to_hf_dataset(
    *,
    log_dir: Path,
    repo_id: str,
    private: bool,
    run_name: str,
) -> bool:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        print(
            "[DPO-BATCH] Warning: Hugging Face Hub unavailable for margin log archival: "
            f"{exc}"
        )
        return False

    try:
        api = HfApi()
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(log_dir),
            path_in_repo=log_dir.name,
            commit_message=f"Upload margin logs for {run_name}",
        )
        return True
    except Exception as exc:
        print(
            "[DPO-BATCH] Warning: failed to archive margin logs to Hugging Face: "
            f"{exc}"
        )
        return False


def _delete_local_margin_logs(log_dir: Path) -> None:
    if not log_dir.exists():
        return
    shutil.rmtree(log_dir)


def archive_margin_logs(run_plan: Dict[str, Any]) -> str | None:
    if run_plan["trainer_type"] != "margin":
        return None

    margin_cfg = run_plan["config"].get("margin_log", {})
    if not bool(margin_cfg.get("archive_after_run", False)):
        return None

    log_dir = Path(str(margin_cfg.get("log_dir", "")))
    if not log_dir.exists():
        return f"Margin log directory missing after successful run: {log_dir}"

    archive_backend = str(margin_cfg.get("archive_backend", "wandb_then_hf"))
    if archive_backend != "wandb_then_hf":
        return f"Unsupported margin_log.archive_backend: {archive_backend}"

    dpo_cfg = run_plan["config"].get("dpo_training", {})
    wandb_project = dpo_cfg.get("wandb_project") or dpo_cfg.get("report")
    archived = False
    if wandb_project:
        archived = _archive_margin_logs_to_wandb(
            log_dir=log_dir,
            artifact_name=str(
                margin_cfg.get(
                    "wandb_artifact_name",
                    f"{dpo_cfg.get('run_name', 'dpo')}-margin-logs",
                )
            ),
        )

    if not archived:
        archived = _archive_margin_logs_to_hf_dataset(
            log_dir=log_dir,
            repo_id=str(margin_cfg["hf_dataset_repo_id"]),
            private=bool(margin_cfg.get("hf_dataset_private", False)),
            run_name=str(dpo_cfg.get("run_name", "dpo")),
        )

    if not archived:
        return f"Failed to archive margin logs for {dpo_cfg.get('run_name', 'dpo')}."

    if bool(margin_cfg.get("delete_local_after_archive", False)):
        try:
            _delete_local_margin_logs(log_dir)
        except Exception as exc:
            return f"Failed to delete local margin logs after archival: {_format_run_error(exc)}"

    return None


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


def cleanup_run_artifacts(*, run_plan: Dict[str, Any], cleanup_config: Dict[str, Any]) -> None:
    """Release process memory and remove local/HF cache artifacts after a successful run."""
    _finish_wandb_run()
    gc.collect()
    _clear_cuda_memory()

    if bool(cleanup_config.get("delete_run_output", False)):
        save_dir = Path(str(run_plan["config"]["dpo_training"]["save_dir"]))
        shutil.rmtree(save_dir, ignore_errors=True)
        shutil.rmtree(Path(str(run_plan["output_dir"])), ignore_errors=True)

    (
        delete_policy_model_cache,
        delete_dataset_cache,
        _,
    ) = _resolve_cache_cleanup_flags(cleanup_config)
    repo_ids: List[str] = []

    if delete_policy_model_cache:
        policy_name = str(run_plan["policy_name"])
        ref_name = str(run_plan["ref_name"])
        if ref_name == policy_name:
            print(
                "[DPO-BATCH] Skipping policy model cache cleanup because "
                "ref_name matches policy_name; Hugging Face cache is shared by repo_id."
            )
        else:
            repo_ids.append(policy_name)

    if delete_dataset_cache:
        repo_ids.append(str(run_plan["config"]["dataset"]["dataset_name"]))

    if repo_ids:
        _delete_hf_cache_entries(repo_ids)


def cleanup_completed_policy_cache(
    *,
    completed_policy_name: str,
    cleanup_config: Dict[str, Any],
) -> None:
    """Delete a model cache only after its last batch run has completed."""
    (
        _delete_policy_model_cache,
        _delete_dataset_cache,
        delete_completed_policy_model_cache,
    ) = _resolve_cache_cleanup_flags(cleanup_config)
    if not delete_completed_policy_model_cache:
        return

    _delete_hf_cache_entries([completed_policy_name])


def _run_training_job(run_plan: Dict[str, Any]) -> None:
    config = run_plan["config"]
    output_dir = str(run_plan["output_dir"])
    if run_plan["trainer_type"] == "beta":
        run_beta_dpo_training(config, output_dir=output_dir)
        return
    if run_plan["trainer_type"] == "margin":
        run_margin_dpo_training(config, output_dir=output_dir)
        return
    raise ValueError(f"Unknown trainer_type: {run_plan['trainer_type']}")


def run_batch_dpo(batch_config: Dict[str, Any], *, config_dir: Path) -> int:
    """Run the full DPO matrix sequentially and fail fast on errors."""
    cleanup_config = batch_config.get("cleanup", {})
    run_plans = build_run_matrix(batch_config, config_dir=config_dir)
    is_main_process = _is_main_process()

    for index, run_plan in enumerate(run_plans, start=1):
        run_label = (
            f"{run_plan['dataset_slug']} / {run_plan['model_slug']} / "
            f"{run_plan['trainer_slug']} ({index}/{len(run_plans)})"
        )
        if is_main_process:
            print(
                f"[DPO-BATCH] Starting {run_label} "
                f"from {run_plan['base_config_path']} -> {run_plan['hub_model_id']}"
            )

        train_error = None
        try:
            _run_training_job(run_plan)
        except Exception as exc:
            train_error = _format_run_error(exc)

        train_errors = _gather_error_messages(train_error)
        if train_errors:
            if is_main_process:
                print(f"[DPO-BATCH] Failed {run_label}: {train_errors[0]}")
            _distributed_barrier()
            return 1

        post_run_error = None
        if is_main_process:
            print(f"[DPO-BATCH] Completed {run_label}")
            try:
                archive_error = archive_margin_logs(run_plan)
                if archive_error:
                    post_run_error = archive_error
                else:
                    cleanup_run_artifacts(run_plan=run_plan, cleanup_config=cleanup_config)
                    print(f"[DPO-BATCH] Cleanup complete for {run_label}")
                    has_future_same_policy = any(
                        future_run_plan["policy_name"] == run_plan["policy_name"]
                        for future_run_plan in run_plans[index:]
                    )
                    if not has_future_same_policy:
                        cleanup_completed_policy_cache(
                            completed_policy_name=str(run_plan["policy_name"]),
                            cleanup_config=cleanup_config,
                        )
                        if bool(
                            cleanup_config.get(
                                "delete_completed_policy_model_cache", False
                            )
                        ):
                            print(
                                "[DPO-BATCH] Deleted completed policy model cache for "
                                f"{run_plan['policy_name']}"
                            )
            except Exception as exc:
                post_run_error = _format_run_error(exc)

        post_run_errors = _gather_error_messages(post_run_error)
        if post_run_errors:
            if is_main_process:
                print(f"[DPO-BATCH] Failed {run_label}: {post_run_errors[0]}")
            _distributed_barrier()
            return 1

        barrier_error = _distributed_barrier()
        if isinstance(barrier_error, str) and barrier_error:
            if is_main_process:
                print(f"[DPO-BATCH] Failed {run_label}: {barrier_error}")
            return 1

    if is_main_process:
        print(f"[DPO-BATCH] All {len(run_plans)} DPO runs completed successfully.")
    return 0


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for the batch DPO runner."""
    parser = argparse.ArgumentParser(description="Run the eight-run HH DPO matrix")
    parser.add_argument("--config", type=str, default="config_dpo_batch.yaml")
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=-1)
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    batch_config = load_yaml(str(config_path))
    return run_batch_dpo(batch_config, config_dir=config_path.parent)


if __name__ == "__main__":
    raise SystemExit(main())
