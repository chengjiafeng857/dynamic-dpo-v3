"""Batch DPO orchestration: train, archive margin logs, clean outputs, delete caches."""

import argparse
import copy
import errno
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
from .cli import run_beta_dpo_training, run_margin_dpo_training
from .config.loader import load_yaml


class DPOBatchRunPlan(TypedDict):
    trainer_type: str
    trainer_slug: str
    base_config_path: str
    dataset_slug: str
    model_slug: str
    policy_name: str
    ref_name: str
    hub_model_id: str
    output_dir: str
    config: dict[str, Any]


def _build_run_config(
    base_config: dict[str, Any],
    *,
    trainer_type: str,
    trainer_slug: str,
    run_entry: dict[str, Any],
    hf_username: str,
) -> dict[str, Any]:
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
        margin_cfg["delete_local_after_archive"] = True
        margin_cfg["hf_dataset_repo_id"] = f"{hf_username}/{run_name}-margin-logs"
        margin_cfg["hf_dataset_private"] = False

    return config


def build_run_matrix(batch_config: dict[str, Any], *, config_dir: Path) -> List[DPOBatchRunPlan]:
    """Expand the batch config into concrete per-run DPO jobs."""
    execution_order = str(batch_config.get("execution_order", "run_major"))
    if execution_order not in {"run_major", "trainer_major"}:
        raise ValueError(
            "execution_order must be either 'run_major' or 'trainer_major'."
        )

    hf_username = str(batch_config["hf_username"])
    trainer_specs: List[dict[str, Any]] = []
    for trainer_entry in batch_config.get("trainers", []):
        trainer_type = str(trainer_entry["trainer_type"])
        if trainer_type not in {"beta", "margin"}:
            raise ValueError("trainer_type must be either 'beta' or 'margin'.")
        base_config_path = resolve_config_path(
            str(trainer_entry["base_config"]), config_dir
        )
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

    run_plans: List[DPOBatchRunPlan] = []
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


def _build_run_label(run_plan: DPOBatchRunPlan, *, index: int, total_runs: int) -> str:
    return (
        f"{run_plan['dataset_slug']} / {run_plan['model_slug']} / "
        f"{run_plan['trainer_slug']} ({index}/{total_runs})"
    )


def _run_training_job(run_plan: DPOBatchRunPlan) -> None:
    config = run_plan["config"]
    output_dir = str(run_plan["output_dir"])
    if run_plan["trainer_type"] == "beta":
        run_beta_dpo_training(config, output_dir=output_dir)
        return
    if run_plan["trainer_type"] == "margin":
        run_margin_dpo_training(config, output_dir=output_dir)
        return
    raise ValueError(f"Unknown trainer_type: {run_plan['trainer_type']}")


def _run_training_for_plan(run_plan: DPOBatchRunPlan) -> str | None:
    try:
        _run_training_job(run_plan)
    except Exception as exc:
        return _format_run_error(exc)
    return None


def _phase_error_message(error: Any) -> str | None:
    return error if isinstance(error, str) and error else None


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


def archive_margin_logs_for_run(run_plan: DPOBatchRunPlan) -> str | None:
    if run_plan["trainer_type"] != "margin":
        return None

    margin_cfg = run_plan["config"].get("margin_log", {})
    if not bool(margin_cfg.get("archive_after_run", False)):
        return None

    log_dir = Path(str(margin_cfg.get("log_dir", "")))
    if not log_dir.exists():
        return f"Margin log directory missing after successful run: {log_dir}"

    repo_id = margin_cfg.get("hf_dataset_repo_id")
    if not repo_id:
        return "margin_log.hf_dataset_repo_id must be set when archive_after_run=true."

    archived = _archive_margin_logs_to_hf_dataset(
        log_dir=log_dir,
        repo_id=str(repo_id),
        private=bool(margin_cfg.get("hf_dataset_private", False)),
        run_name=str(run_plan["config"]["dpo_training"].get("run_name", "dpo")),
    )
    if not archived:
        return (
            "Failed to archive margin logs for "
            f"{run_plan['config']['dpo_training'].get('run_name', 'dpo')}."
        )

    if bool(margin_cfg.get("delete_local_after_archive", False)):
        try:
            shutil.rmtree(log_dir)
        except Exception as exc:
            return (
                "Failed to delete local margin logs after archival: "
                f"{_format_run_error(exc)}"
            )

    return None


def cleanup_run_artifacts(
    *, run_plan: DPOBatchRunPlan, cleanup_config: dict[str, Any]
) -> str | None:
    finish_wandb_run()
    gc.collect()
    clear_cuda_memory()

    if bool(cleanup_config.get("delete_run_output", False)):
        save_dir = Path(str(run_plan["config"]["dpo_training"]["save_dir"]))
        try:
            shutil.rmtree(save_dir, ignore_errors=True)
            shutil.rmtree(Path(str(run_plan["output_dir"])), ignore_errors=True)
        except Exception as exc:
            return _format_run_error(exc)

    delete_policy_model_cache, delete_dataset_cache, _ = resolve_cache_cleanup_flags(
        cleanup_config
    )
    repo_ids: list[str] = []

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
        delete_hf_cache_entries(repo_ids, log_prefix="DPO-BATCH")
    return None


def cleanup_successful_run(
    *, run_plan: DPOBatchRunPlan, cleanup_config: dict[str, Any]
) -> str | None:
    archive_error = archive_margin_logs(run_plan)
    if archive_error:
        return archive_error
    return cleanup_run_artifacts(run_plan=run_plan, cleanup_config=cleanup_config)


def archive_margin_logs(run_plan: DPOBatchRunPlan) -> str | None:
    return archive_margin_logs_for_run(run_plan)


def _has_future_same_policy(
    run_plan: DPOBatchRunPlan,
    run_plans: List[DPOBatchRunPlan],
    *,
    index: int,
) -> bool:
    return any(
        future_run_plan["policy_name"] == run_plan["policy_name"]
        for future_run_plan in run_plans[index:]
    )


def delete_completed_policy_cache_if_last_use(
    *,
    run_plan: DPOBatchRunPlan,
    run_plans: List[DPOBatchRunPlan],
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
    delete_hf_cache_entries([completed_policy_name], log_prefix="DPO-BATCH")


def run_batch_dpo(batch_config: dict[str, Any], *, config_dir: Path) -> int:
    """Run the full DPO matrix sequentially and fail fast on errors."""
    cleanup_config = batch_config.get("cleanup", {})
    run_plans = build_run_matrix(batch_config, config_dir=config_dir)
    main_process = is_main_process()

    for index, run_plan in enumerate(run_plans, start=1):
        run_label = _build_run_label(run_plan, index=index, total_runs=len(run_plans))
        if main_process:
            print(
                f"[DPO-BATCH] Starting {run_label} "
                f"from {run_plan['base_config_path']} -> {run_plan['hub_model_id']}"
            )

        train_errors = gather_error_messages(
            _phase_error_message(_run_training_for_plan(run_plan))
        )
        if train_errors:
            if main_process:
                print(f"[DPO-BATCH] Failed {run_label}: {train_errors[0]}")
            distributed_barrier()
            return 1

        post_run_error = None
        deleted_completed_policy_cache = False
        if main_process:
            print(f"[DPO-BATCH] Completed {run_label}")
            post_run_error = cleanup_successful_run(
                run_plan=run_plan,
                cleanup_config=cleanup_config,
            )
            if _phase_error_message(post_run_error) is None:
                print(f"[DPO-BATCH] Cleanup complete for {run_label}")
                deleted_completed_policy_cache = delete_completed_policy_cache_if_last_use(
                    run_plan=run_plan,
                    run_plans=run_plans,
                    index=index,
                    cleanup_config=cleanup_config,
                )

        post_run_errors = gather_error_messages(_phase_error_message(post_run_error))
        if post_run_errors:
            if main_process:
                print(f"[DPO-BATCH] Failed {run_label}: {post_run_errors[0]}")
            distributed_barrier()
            return 1

        if main_process and deleted_completed_policy_cache:
            print(
                "[DPO-BATCH] Deleted completed policy model cache for "
                f"{run_plan['policy_name']}"
            )

        barrier_error = distributed_barrier()
        if isinstance(barrier_error, str) and barrier_error:
            if main_process:
                print(f"[DPO-BATCH] Failed {run_label}: {barrier_error}")
            return 1

    if main_process:
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
