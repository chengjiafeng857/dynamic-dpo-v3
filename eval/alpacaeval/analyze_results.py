"""
Analyze AlpacaEval 2.0 evaluation results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os


def _safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _print_leaderboard(leaderboard_path: str, model_name: str) -> None:
    if not os.path.exists(leaderboard_path):
        print(f"Leaderboard not found at {leaderboard_path}")
        return

    with open(leaderboard_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("Leaderboard is empty.")
        return

    model_row = None
    for row in rows:
        if model_name in (row.get("name") or ""):
            model_row = row
            break

    print("\n" + "=" * 60)
    print("ALPACAEVAL 2.0 EVALUATION RESULTS")
    print("=" * 60)

    if model_row:
        print(f"\nModel: {model_row.get('name')}")
        win_rate = _safe_float(model_row.get("win_rate"))
        if win_rate is not None:
            print(f"Win Rate: {win_rate:.2%}")

        lc_win = _safe_float(model_row.get("length_controlled_winrate"))
        if lc_win is not None:
            print(f"Length-Controlled Win Rate: {lc_win:.2%}")

        avg_length = _safe_float(model_row.get("avg_length"))
        if avg_length is not None:
            print(f"Average Output Length: {avg_length:.0f} characters")
    else:
        print("\nModel not found in leaderboard. Showing full table.")

    print("\nTop 10 Models (by file order):")
    for row in rows[:10]:
        name = row.get("name", "unknown")
        win_rate = _safe_float(row.get("win_rate"))
        win_rate_str = f"{win_rate:.2%}" if win_rate is not None else "n/a"
        print(f"- {name}: {win_rate_str}")


def _print_annotations(annotations_path: str) -> None:
    if not os.path.exists(annotations_path):
        print(f"\nAnnotations not found at {annotations_path}")
        return

    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    if not annotations:
        print("\nAnnotations file is empty.")
        return

    preferences = [item.get("preference", 0) for item in annotations]
    wins = preferences.count(2)
    losses = preferences.count(1)
    draws = preferences.count(0)
    total = len(preferences)

    print(f"\nTotal Evaluations: {total}")
    print(f"Wins: {wins} ({wins / total * 100:.1f}%)")
    print(f"Losses: {losses} ({losses / total * 100:.1f}%)")
    print(f"Draws: {draws} ({draws / total * 100:.1f}%)")
    print("\n" + "=" * 60 + "\n")


def analyze_results(results_dir: str, model_name: str) -> None:
    leaderboard_path = os.path.join(results_dir, "leaderboard.csv")
    annotations_path = os.path.join(results_dir, "annotations.json")
    _print_leaderboard(leaderboard_path, model_name)
    _print_annotations(annotations_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze AlpacaEval results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="test/alpacaeval/results",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="hh-llama32-1b-sft",
        help="Model name to locate in leaderboard",
    )
    args = parser.parse_args()
    analyze_results(args.results_dir, args.model_name)


if __name__ == "__main__":
    main()
