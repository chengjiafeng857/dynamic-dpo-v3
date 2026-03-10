"""
Sample qualitative examples from GPT-4 oracle evaluations.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml


COMPARISON_MAP = {
    "sft_vs_standard_dpo": ("sft", "standard_dpo"),
    "sft_vs_beta_dpo": ("sft", "beta_dpo"),
    "sft_vs_dynamic_beta_dpo": ("sft", "dynamic_beta_dpo"),
}


def _load_json(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _sample(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows
    return rng.sample(rows, count)


def _format_example(row: dict[str, Any]) -> str:
    return (
        f"**Post ID:** {row.get('post_id')}\n\n"
        f"**Post:**\n{row.get('post')}\n\n"
        f"**Summary A ({row.get('summary_a_model')}):**\n{row.get('summary_a')}\n\n"
        f"**Summary B ({row.get('summary_b_model')}):**\n{row.get('summary_b')}\n\n"
        f"**GPT-4 Preference:** {row.get('gpt4_preference')}\n\n"
        f"**GPT-4 Explanation:**\n{row.get('gpt4_explanation')}\n\n"
        "---\n"
    )


def generate_qualitative_report(
    results_dir: str | Path,
    output_path: str | Path,
    num_examples: int,
    seed: int,
) -> None:
    results_dir = Path(results_dir)
    output_path = Path(output_path)
    rng = random.Random(seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# GPT-4 Oracle Qualitative Analysis\n\n")

        for comparison, (sft_model, dpo_model) in COMPARISON_MAP.items():
            result_path = results_dir / f"{comparison}.json"
            if not result_path.exists():
                continue

            rows = _load_json(result_path)
            dpo_wins = [row for row in rows if row.get("winner") == dpo_model]
            sft_wins = [row for row in rows if row.get("winner") == sft_model]
            ties = [row for row in rows if row.get("winner") == "tie"]

            handle.write(f"## {comparison}\n\n")
            handle.write(f"### {dpo_model} wins\n\n")
            for row in _sample(dpo_wins, num_examples, rng):
                handle.write(_format_example(row))

            handle.write(f"### {sft_model} wins\n\n")
            for row in _sample(sft_wins, num_examples, rng):
                handle.write(_format_example(row))

            if ties:
                handle.write("### Ties\n\n")
                for row in _sample(ties, num_examples, rng):
                    handle.write(_format_example(row))

    print(f"Saved qualitative analysis to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qualitative analysis report")
    parser.add_argument(
        "--config",
        type=str,
        default="test/gpt_judge_TLDR/config_evaluation.yaml",
        help="Path to evaluation config YAML",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory with evaluation results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown path",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of examples per category",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    output_cfg = config.get("output", {})
    results_dir = args.results_dir or output_cfg.get("results_dir", "test/gpt_judge_TLDR/results")
    output_path = args.output or str(Path(results_dir) / "qualitative_examples.md")

    generate_qualitative_report(
        results_dir=results_dir,
        output_path=output_path,
        num_examples=args.num_examples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
