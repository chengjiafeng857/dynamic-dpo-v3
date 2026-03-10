"""
Analyze GPT-4 oracle evaluation results and compute win rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from scipy.stats import beta

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


def compute_confidence_interval(
    wins: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    alpha = 1.0 - confidence
    if wins == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha / 2, wins, total - wins + 1)
    if wins == total:
        upper = 1.0
    else:
        upper = beta.ppf(1 - alpha / 2, wins + 1, total - wins)
    return lower, upper


def calculate_win_rate(
    evaluations: list[dict[str, Any]],
    dpo_model: str,
    sft_model: str = "sft",
) -> dict[str, Any]:
    wins = 0
    losses = 0
    ties = 0

    for row in evaluations:
        winner = row.get("winner")
        if winner == dpo_model:
            wins += 1
        elif winner == sft_model:
            losses += 1
        else:
            ties += 1

    total = wins + losses + ties
    win_rate = wins / total if total else 0.0
    ci_lower, ci_upper = compute_confidence_interval(wins, total)

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "total": total,
        "win_rate": win_rate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def create_comparison_table(results_dir: str | Path) -> pd.DataFrame:
    rows = []
    results_dir = Path(results_dir)

    for comparison, (sft_model, dpo_model) in COMPARISON_MAP.items():
        result_path = results_dir / f"{comparison}.json"
        if not result_path.exists():
            continue
        evaluations = _load_json(result_path)
        stats = calculate_win_rate(
            evaluations, dpo_model=dpo_model, sft_model=sft_model
        )
        rows.append(
            {
                "Model Pair": f"{sft_model} vs {dpo_model}",
                "Win Rate": f"{stats['win_rate'] * 100:.2f}%",
                "Wins": stats["wins"],
                "Losses": stats["losses"],
                "Ties": stats["ties"],
                "Total": stats["total"],
                "95% CI": f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]",
            }
        )

    return pd.DataFrame(rows)


def generate_report(results_dir: str | Path, output_path: str | Path) -> None:
    results_dir = Path(results_dir)
    output_path = Path(output_path)
    table = create_comparison_table(results_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# GPT-4 Oracle Evaluation Report\n\n")
        if table.empty:
            handle.write("No evaluation results found.\n")
            return
        handle.write(table.to_markdown(index=False))
        handle.write("\n")

    csv_path = results_dir / "win_rates_summary.csv"
    table.to_csv(csv_path, index=False)

    print(f"Saved report to {output_path}")
    print(f"Saved summary CSV to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GPT-4 oracle results")
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
        help="Output path for markdown report",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    output_cfg = config.get("output", {})
    results_dir = args.results_dir or output_cfg.get(
        "results_dir", "test/gpt_judge_TLDR/results"
    )
    output_path = args.output or str(Path(results_dir) / "evaluation_report.md")

    generate_report(results_dir, output_path)


if __name__ == "__main__":
    main()
