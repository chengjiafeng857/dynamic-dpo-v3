"""Unit tests for shared batch runner utilities."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.batch_runner_utils import (
    collect_revision_hashes,
    delete_hf_cache_entries,
    resolve_cache_cleanup_flags,
    resolve_config_path,
)


class _DeleteStrategy:
    def __init__(self, log: list[tuple[str, tuple[str, ...]]], hashes: tuple[str, ...]):
        self._log = log
        self._hashes = hashes

    def execute(self):
        self._log.append(("execute", self._hashes))


class _CacheInfo:
    def __init__(self, log: list[tuple[str, tuple[str, ...]]]):
        self._log = log
        self.repos = [
            SimpleNamespace(
                repo_id="repo/model",
                revisions=[SimpleNamespace(commit_hash="model-hash")],
            ),
            SimpleNamespace(
                repo_id="repo/dataset",
                revisions=[SimpleNamespace(commit_hash="dataset-hash")],
            ),
        ]

    def delete_revisions(self, *hashes: str):
        self._log.append(("delete", hashes))
        return _DeleteStrategy(self._log, hashes)


class BatchRunnerUtilsTest(unittest.TestCase):
    def test_resolve_config_path_resolves_relative_paths(self):
        resolved = resolve_config_path("config.yaml", Path("/tmp/project"))
        self.assertEqual(resolved, Path("/tmp/project/config.yaml"))

    def test_resolve_config_path_preserves_absolute_paths(self):
        resolved = resolve_config_path("/tmp/config.yaml", Path("/tmp/project"))
        self.assertEqual(resolved, Path("/tmp/config.yaml"))

    def test_resolve_cache_cleanup_flags_honors_explicit_keys(self):
        flags = resolve_cache_cleanup_flags(
            {
                "delete_policy_model_cache": True,
                "delete_dataset_cache": False,
                "delete_completed_policy_model_cache": True,
            }
        )
        self.assertEqual(flags, (True, False, True))

    def test_resolve_cache_cleanup_flags_uses_legacy_flag(self):
        flags = resolve_cache_cleanup_flags({"delete_hf_download_cache": True})
        self.assertEqual(flags, (True, True, False))

    def test_collect_revision_hashes_filters_repo_ids(self):
        cache_info = _CacheInfo([])
        hashes = collect_revision_hashes(cache_info, ["repo/dataset"])
        self.assertEqual(hashes, ["dataset-hash"])

    def test_delete_hf_cache_entries_executes_delete_strategy(self):
        log: list[tuple[str, tuple[str, ...]]] = []
        with patch(
            "src.batch_runner_utils.scan_cache_dir",
            return_value=_CacheInfo(log),
        ):
            delete_hf_cache_entries(
                ["repo/model", "repo/dataset"],
                log_prefix="TEST",
            )

        self.assertEqual(log[0], ("delete", ("model-hash", "dataset-hash")))
        self.assertEqual(log[1], ("execute", ("model-hash", "dataset-hash")))


if __name__ == "__main__":
    unittest.main()
