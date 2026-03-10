"""
Visualize GPT-4 oracle evaluation results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
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


def _word_count(text: str | None) -> int:
    if not text:
        return 0
    return len(text.split())


def _collect_stats(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    length_rows = []

    for comparison, (sft_model, dpo_model) in COMPARISON_MAP.items():
        path = results_dir / f"{comparison}.json"
        if not path.exists():
            continue
        rows = _load_json(path)

        wins = sum(1 for row in rows if row.get("winner") == dpo_model)
        losses = sum(1 for row in rows if row.get("winner") == sft_model)
        ties = sum(1 for row in rows if row.get("winner") == "tie")
        total = wins + losses + ties
        win_rate = wins / total if total else 0.0

        summary_rows.append(
            {
                "comparison": comparison,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "total": total,
                "win_rate": win_rate,
            }
        )

        for row in rows:
            summary_a_model = row.get("summary_a_model")
            summary_b_model = row.get("summary_b_model")
            summary_a = row.get("summary_a")
            summary_b = row.get("summary_b")

            sft_summary = summary_a if summary_a_model == sft_model else summary_b
            dpo_summary = summary_a if summary_a_model == dpo_model else summary_b

            length_rows.append(
                {
                    "comparison": comparison,
                    "winner": row.get("winner"),
                    "sft_length": _word_count(sft_summary),
                    "dpo_length": _word_count(dpo_summary),
                    "length_diff": _word_count(dpo_summary)
                    - _word_count(sft_summary),
                    "subreddit": (row.get("metadata") or {}).get("subreddit"),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(length_rows)


def _plot_win_rates(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = range(len(summary_df))
    ax.bar(positions, summary_df["win_rate"], color="#1f77b4")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1)
    ax.set_title("GPT-4 Win Rates")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(summary_df["comparison"], rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "win_rates_chart.png")
    plt.close(fig)


def _plot_win_loss(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = range(len(summary_df))
    ax.bar(positions, summary_df["wins"], label="DPO wins")
    ax.bar(
        positions,
        summary_df["losses"],
        bottom=summary_df["wins"],
        label="SFT wins",
    )
    ax.bar(
        positions,
        summary_df["ties"],
        bottom=summary_df["wins"] + summary_df["losses"],
        label="Ties",
    )
    ax.set_ylabel("Count")
    ax.set_title("Win/Loss/Tie Breakdown")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(summary_df["comparison"], rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "win_loss_breakdown.png")
    plt.close(fig)


def _plot_length_analysis(length_df: pd.DataFrame, output_dir: Path) -> None:
    if length_df.empty:
        return
    for comparison in length_df["comparison"].unique():
        subset = length_df[length_df["comparison"] == comparison]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        dpo_mask = subset["winner"].str.contains("dpo", na=False)
        ax.boxplot(
            [
                subset[subset["winner"] == "sft"]["length_diff"].tolist(),
                subset[dpo_mask]["length_diff"].tolist(),
                subset[subset["winner"] == "tie"]["length_diff"].tolist(),
            ],
            labels=["SFT wins", "DPO wins", "Ties"],
            showfliers=False,
        )
        ax.set_ylabel("Length diff (DPO - SFT, words)")
        ax.set_title(f"Length Difference: {comparison}")
        fig.tight_layout()
        fig.savefig(output_dir / f"length_analysis_{comparison}.png")
        plt.close(fig)


def _plot_category_heatmap(length_df: pd.DataFrame, output_dir: Path) -> None:
    if length_df.empty or "subreddit" not in length_df.columns:
        return
    df = length_df.dropna(subset=["subreddit"])
    if df.empty:
        return

    df = df.copy()
    df["dpo_win"] = df["winner"].str.contains("dpo", na=False)
    grouped = df.groupby(["subreddit", "comparison"])["dpo_win"].mean().reset_index()

    top_subreddits = (
        df["subreddit"].value_counts().head(10).index.tolist()
    )
    filtered = grouped[grouped["subreddit"].isin(top_subreddits)]
    if filtered.empty:
        return

    pivot = filtered.pivot(index="subreddit", columns="comparison", values="dpo_win")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.fillna(0).values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("DPO Win Rate by Subreddit")
    fig.colorbar(im, ax=ax, label="Win rate")
    fig.tight_layout()
    fig.savefig(output_dir / "category_heatmap.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize GPT-4 oracle results")
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
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for visualizations",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    output_cfg = config.get("output", {})
    results_dir = Path(args.results_dir or output_cfg.get("results_dir", "test/gpt_judge_TLDR/results"))
    output_dir = Path(
        args.output_dir or output_cfg.get("visualizations_dir", "test/gpt_judge_TLDR/visualizations")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df, length_df = _collect_stats(results_dir)
    _plot_win_rates(summary_df, output_dir)
    _plot_win_loss(summary_df, output_dir)
    _plot_length_analysis(length_df, output_dir)
    _plot_category_heatmap(length_df, output_dir)

    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
