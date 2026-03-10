from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from typing import Dict, Iterable, Tuple

import yaml

try:
    from huggingface_hub import HfApi
except ImportError as exc:
    raise ImportError(
        "huggingface_hub is required to upload files. Install it with `pip install huggingface_hub`."
    ) from exc


DEFAULT_JUDGED_FILENAME = "rollout_judged.jsonl"
SPECIAL_TOKEN_RE = re.compile(r"<\\|[^>]+?\\|>")
ESCAPED_WHITESPACE_RE = re.compile(r"(?:\\[nrt])+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload rollout_judged.jsonl from run_rollout.py to the Hugging Face Hub."
    )
    parser.add_argument("--config", type=str, default="config_dpo.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--judged_path", type=str, default=None)
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--repo_type", type=str, default="dataset", choices=["dataset", "model"])
    parser.add_argument("--path_in_repo", type=str, default=None)
    parser.add_argument("--private", action="store_true", default=False)
    parser.add_argument("--commit_message", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--token", type=str, default=None)
    return parser.parse_args()


def resolve_judged_path(config: Dict, args: argparse.Namespace) -> str:
    if args.judged_path:
        return args.judged_path
    rollout_cfg = config.get("rollout", {})
    output_dir = args.output_dir or rollout_cfg.get("output_dir", "rollout_output")
    return os.path.join(output_dir, DEFAULT_JUDGED_FILENAME)


def resolve_repo_id(config: Dict, args: argparse.Namespace) -> str:
    repo_id = args.repo_id
    if repo_id:
        return repo_id
    rollout_cfg = config.get("rollout", {})
    repo_id = rollout_cfg.get("upload_repo_id")
    if repo_id:
        return repo_id
    raise ValueError("Missing repo id. Set --repo_id or rollout.upload_repo_id in config.")


def resolve_token(args: argparse.Namespace) -> str | None:
    if args.token:
        return args.token
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def _ensure_file(path: str) -> str:
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Judged rollout file not found: {resolved}")
    return resolved


def _normalize_text(text: object) -> str:
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def _is_effectively_empty(text: object) -> bool:
    stripped = _normalize_text(text).strip()
    if not stripped:
        return True
    stripped = ESCAPED_WHITESPACE_RE.sub("", stripped)
    stripped = SPECIAL_TOKEN_RE.sub("", stripped).strip()
    return not stripped


def _iter_message_contents(value: object) -> Iterable[str]:
    if isinstance(value, str) or value is None:
        yield _normalize_text(value)
        return
    if isinstance(value, dict):
        yield _normalize_text(value.get("content", ""))
        return
    if isinstance(value, list):
        for msg in value:
            if isinstance(msg, dict):
                yield _normalize_text(msg.get("content", ""))
            elif isinstance(msg, str) or msg is None:
                yield _normalize_text(msg)


def _has_nonempty_text(value: object) -> bool:
    found = False
    for content in _iter_message_contents(value):
        found = True
        if not _is_effectively_empty(content):
            return True
    return False if found else False


def filter_judged_rollout(input_path: str, output_path: str) -> Tuple[int, int]:
    total = 0
    kept = 0
    with open(input_path, "r", encoding="utf-8") as src, open(
        output_path, "w", encoding="utf-8"
    ) as dst:
        for line_num, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {input_path}") from exc
            if not _has_nonempty_text(record.get("chosen")):
                continue
            if not _has_nonempty_text(record.get("rejected")):
                continue
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    return total, kept


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    judged_path = _ensure_file(resolve_judged_path(config, args))
    repo_id = resolve_repo_id(config, args)
    path_in_repo = args.path_in_repo or os.path.basename(judged_path)
    commit_message = args.commit_message or f"Upload {path_in_repo}"

    token = resolve_token(args)
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
        token=token,
    )
    filtered_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp:
            filtered_path = tmp.name
        total, kept = filter_judged_rollout(judged_path, filtered_path)
        api.upload_file(
            path_or_fileobj=filtered_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=args.repo_type,
            token=token,
            commit_message=commit_message,
            revision=args.revision,
        )
        print(
            f"Uploaded {kept}/{total} filtered rows from {judged_path} to "
            f"{repo_id}/{path_in_repo} ({args.repo_type})."
        )
    finally:
        if filtered_path and os.path.exists(filtered_path):
            os.remove(filtered_path)


if __name__ == "__main__":
    main()
