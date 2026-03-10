"""
Evaluate summary pairs using GPT-4 as an oracle judge.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError as exc:
    raise RuntimeError(
        "openai package is required for GPT-4 oracle evaluation. Install it first."
    ) from exc


COMPARISON_MAP = {
    "sft_vs_standard_dpo": ("sft", "standard_dpo"),
    "sft_vs_beta_dpo": ("sft", "beta_dpo"),
    "sft_vs_dynamic_beta_dpo": ("sft", "dynamic_beta_dpo"),
    # "base_vs_sft": ("base_model", "sft"),
    # "base_vs_standard_dpo": ("base_model", "standard_dpo"),
    # "base_vs_beta_dpo": ("base_model", "beta_dpo"),
    # "base_vs_dynamic_beta_dpo": ("base_model", "dynamic_beta_dpo"),
}


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_json(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(data: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)


def create_evaluation_prompt(
    template: str, post: str, summary_a: str, summary_b: str
) -> str:
    return template.format(post=post, summary_a=summary_a, summary_b=summary_b)


def _parse_preference(text: str) -> str | None:
    if re.search(r"\b(tie|equal)\b", text, flags=re.IGNORECASE):
        return "TIE"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        match = re.search(r"summary\s*([ab])", line, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
        if line.lower().startswith(("preference", "choice", "winner")):
            match = re.search(r"\b([ab])\b", line, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()

    match = re.findall(r"summary\s*([ab])", text, flags=re.IGNORECASE)
    if match:
        return match[-1].upper()

    return None


def call_gpt4_oracle(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
) -> tuple[str, dict[str, Any]]:
    attempt = 0
    backoff = initial_backoff
    last_error: Exception | None = None

    while attempt <= max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage else {}
            return text, usage
        except Exception as exc:  # Broad to handle API/network errors.
            last_error = exc
            attempt += 1
            if attempt > max_retries:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    raise RuntimeError(f"OpenAI API failed after {max_retries} retries") from last_error


def evaluate_comparison(
    summaries_path: str,
    comparison_type: str,
    output_path: str,
    oracle_cfg: dict[str, Any],
    resume: bool,
    max_examples: int | None,
) -> None:
    if comparison_type not in COMPARISON_MAP:
        raise ValueError(f"Unknown comparison_type: {comparison_type}")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    summaries = _load_json(summaries_path)
    results: list[dict[str, Any]] = []
    if resume and Path(output_path).exists():
        results = _load_json(output_path)

    completed_ids = {row.get("post_id") for row in results}

    rng = random.Random(oracle_cfg.get("seed", 42))
    randomize = oracle_cfg.get("randomize_order", True)
    prompt_template = oracle_cfg.get("prompt_template")
    if not prompt_template:
        raise ValueError("gpt4_oracle.prompt_template must be set in config.")

    client = OpenAI()

    sft_key, dpo_key = COMPARISON_MAP[comparison_type]
    sft_summary_key = f"{sft_key}_summary"
    dpo_summary_key = f"{dpo_key}_summary"

    pending = [row for row in summaries if row.get("post_id") not in completed_ids]
    if max_examples is not None:
        pending = pending[:max_examples]

    for idx, row in enumerate(tqdm(pending, desc="GPT-4 eval"), start=1):
        post = row.get("post", "")
        metadata = row.get("metadata")
        summary_a = row.get(sft_summary_key)
        summary_b = row.get(dpo_summary_key)
        summary_a_model = sft_key
        summary_b_model = dpo_key

        if summary_a is None or summary_b is None:
            raise ValueError(
                f"Missing summaries for post_id {row.get('post_id')}: "
                f"{sft_summary_key} or {dpo_summary_key}"
            )

        if randomize and rng.random() < 0.5:
            summary_a, summary_b = summary_b, summary_a
            summary_a_model, summary_b_model = summary_b_model, summary_a_model

        prompt = create_evaluation_prompt(prompt_template, post, summary_a, summary_b)
        text, usage = call_gpt4_oracle(
            client=client,
            prompt=prompt,
            model=oracle_cfg.get("model", "gpt-4-turbo"),
            temperature=oracle_cfg.get("temperature", 0.3),
            max_tokens=oracle_cfg.get("max_tokens", 500),
            max_retries=oracle_cfg.get("max_retries", 5),
            initial_backoff=oracle_cfg.get("initial_backoff", 1.0),
            max_backoff=oracle_cfg.get("max_backoff", 60.0),
        )
        preference = _parse_preference(text)
        winner = None
        if preference == "A":
            winner = summary_a_model
        elif preference == "B":
            winner = summary_b_model
        elif preference == "TIE":
            winner = "tie"

        results.append(
            {
                "post_id": row.get("post_id"),
                "post": post,
                "metadata": metadata,
                "summary_a": summary_a,
                "summary_b": summary_b,
                "summary_a_model": summary_a_model,
                "summary_b_model": summary_b_model,
                "gpt4_preference": preference,
                "gpt4_explanation": text,
                "winner": winner,
                "comparison_type": comparison_type,
                "usage": usage,
            }
        )

        if idx % oracle_cfg.get("batch_size", 10) == 0:
            print(f"Evaluated {idx}/{len(pending)}")
            _save_json(results, output_path)

    _save_json(results, output_path)
    print(f"Saved results to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT-4 oracle evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="test/gpt_judge_TLDR/config_evaluation.yaml",
        help="Path to evaluation config YAML",
    )
    parser.add_argument(
        "--summaries",
        type=str,
        required=True,
        help="Path to summaries JSON",
    )
    parser.add_argument(
        "--comparison",
        type=str,
        required=True,
        choices=sorted(COMPARISON_MAP.keys()),
        help="Comparison type",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output JSON path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Limit number of evaluations",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    oracle_cfg = config.get("gpt4_oracle", {})
    output_cfg = config.get("output", {})
    output_path = args.output or str(
        Path(output_cfg.get("results_dir", "test/gpt_judge_TLDR/results"))
        / f"{args.comparison}.json"
    )

    evaluate_comparison(
        summaries_path=args.summaries,
        comparison_type=args.comparison,
        output_path=output_path,
        oracle_cfg=oracle_cfg,
        resume=args.resume,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
