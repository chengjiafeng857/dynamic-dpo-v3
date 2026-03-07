"""Shared helpers for batch training runners."""

from pathlib import Path
from typing import Any, Iterable, List

try:
    from huggingface_hub import scan_cache_dir
except ImportError:  # pragma: no cover - transformers normally brings this in.
    scan_cache_dir = None


def resolve_config_path(config_path: str, base_dir: Path) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = base_dir / path
    return path


def finish_wandb_run() -> None:
    try:
        import wandb

        wandb.finish()
    except Exception:
        return


def is_main_process() -> bool:
    try:
        import torch.distributed as dist

        return (
            (not dist.is_available())
            or (not dist.is_initialized())
            or (dist.get_rank() == 0)
        )
    except Exception:
        return True


def distributed_barrier() -> str | None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        return "Distributed synchronization failed at barrier."
    return None


def gather_error_messages(local_error: str | None) -> List[str]:
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


def clear_cuda_memory() -> None:
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


def collect_revision_hashes(cache_info: Any, repo_ids: Iterable[str]) -> List[str]:
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


def delete_hf_cache_entries(repo_ids: Iterable[str], *, log_prefix: str) -> None:
    if scan_cache_dir is None:
        print(f"[{log_prefix}] Hugging Face cache scan unavailable; skipping cache cleanup.")
        return

    try:
        cache_info = scan_cache_dir()
        revision_hashes = collect_revision_hashes(cache_info, repo_ids)
        if not revision_hashes:
            return
        delete_strategy = cache_info.delete_revisions(*revision_hashes)
        delete_strategy.execute()
    except Exception as exc:
        print(f"[{log_prefix}] Warning: failed to clean Hugging Face cache: {exc}")


def resolve_cache_cleanup_flags(cleanup_config: dict[str, Any]) -> tuple[bool, bool, bool]:
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
