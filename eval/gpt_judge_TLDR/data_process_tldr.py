"""
Prepare TLDR evaluation data for model-judge evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_field(
    value: str | None, candidates: list[str], available: list[str], field_name: str
) -> str:
    if value:
        if value not in available:
            raise ValueError(
                f"Configured {field_name}='{value}' not found in dataset columns: {available}"
            )
        return value

    for candidate in candidates:
        if candidate in available:
            return candidate

    raise ValueError(
        f"Could not infer {field_name} from dataset columns: {available}. "
        f"Set dataset.{field_name} in config."
    )


def _resolve_optional_field(
    value: str | None, candidates: list[str], available: list[str]
) -> str | None:
    if value:
        return value if value in available else None
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _word_count(text: str) -> int:
    return len(text.split())


def _summarize_lengths(values: list[str]) -> dict[str, float]:
    if not values:
        return {"avg_words": 0.0, "avg_chars": 0.0}
    total_words = sum(_word_count(value) for value in values)
    total_chars = sum(len(value) for value in values)
    count = len(values)
    return {
        "avg_words": total_words / count,
        "avg_chars": total_chars / count,
    }


def load_tldr_test_set(
    dataset_name: str,
    split: str,
    max_samples: int | None,
    seed: int,
    post_field: str | None,
    summary_field: str | None,
    id_field: str | None,
    meta_fields: list[str] | None,
) -> list[dict[str, Any]]:
    dataset = load_dataset(dataset_name, split=split)
    available = list(dataset.column_names)

    post_key = _resolve_field(
        post_field,
        ["post", "prompt", "content", "text", "article", "document", "body"],
        available,
        "post_field",
    )
    summary_key = _resolve_field(
        summary_field,
        ["summary", "reference_summary", "completion", "tldr", "highlights"],
        available,
        "summary_field",
    )
    id_key = _resolve_optional_field(
        id_field,
        ["post_id", "id", "reddit_id"],
        available,
    )

    if max_samples and max_samples > 0 and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    rows: list[dict[str, Any]] = []
    meta_fields = meta_fields or []

    for idx, row in enumerate(dataset):
        record = {
            "post_id": str(row.get(id_key, idx)) if id_key else str(idx),
            "post": row.get(post_key, ""),
            "reference_summary": row.get(summary_key, ""),
        }
        if meta_fields:
            record["metadata"] = {
                field: row.get(field) for field in meta_fields if field in row
            }
        rows.append(record)

    return rows


def save_evaluation_set(data: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TLDR evaluation dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="test/gpt_judge_TLDR/config_evaluation.yaml",
        help="Path to evaluation config YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output path for evaluation JSON",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    output_cfg = config.get("output", {})
    output_path = args.output or str(
        Path(output_cfg.get("data_dir", "test/gpt_judge_TLDR/data"))
        / "tldr_eval_set.json"
    )

    data = load_tldr_test_set(
        dataset_name=dataset_cfg.get("name", "trl-lib/tldr"),
        split=dataset_cfg.get("split", "test"),
        max_samples=dataset_cfg.get("max_samples", 500),
        seed=dataset_cfg.get("seed", 42),
        post_field=dataset_cfg.get("post_field"),
        summary_field=dataset_cfg.get("summary_field"),
        id_field=dataset_cfg.get("id_field"),
        meta_fields=dataset_cfg.get("meta_fields"),
    )

    posts = [row.get("post", "") for row in data]
    summaries = [row.get("reference_summary", "") for row in data]
    post_stats = _summarize_lengths(posts)
    summary_stats = _summarize_lengths(summaries)

    print("Prepared evaluation dataset")
    print(f"Examples: {len(data)}")
    print(
        "Post length avg words/avg chars: "
        f"{post_stats['avg_words']:.1f}/{post_stats['avg_chars']:.1f}"
    )
    print(
        "Summary length avg words/avg chars: "
        f"{summary_stats['avg_words']:.1f}/{summary_stats['avg_chars']:.1f}"
    )

    save_evaluation_set(data, output_path)
    print(f"Saved evaluation set to {output_path}")


if __name__ == "__main__":
    main()
