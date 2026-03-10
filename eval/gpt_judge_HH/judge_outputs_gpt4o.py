"""
Directly judge three output files with GPT-4o using a config-driven prompt.
"""

from __future__ import annotations

import argparse
import hashlib
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
    raise RuntimeError("openai package is required to run this script.") from exc


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_outputs(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of outputs in {path}")
    outputs = {}
    for row in data:
        instruction = row.get("instruction")
        output = row.get("output")
        if not instruction or output is None:
            continue
        if instruction not in outputs:
            outputs[instruction] = output
    return outputs


def _intersection_keys(*maps: dict[str, str]) -> list[str]:
    if not maps:
        return []
    keys = set(maps[0].keys())
    for m in maps[1:]:
        keys &= set(m.keys())
    return sorted(keys)


def _build_prompt(
    template: str, instruction: str, labeled_outputs: dict[str, str]
) -> str:
    return template.format(
        instruction=instruction,
        output_a=labeled_outputs["A"],
        output_b=labeled_outputs["B"],
        output_c=labeled_outputs["C"],
    )


def _normalize_winner(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().strip('"').strip().upper()
    if cleaned in {"A", "B", "C", "TIE"}:
        return cleaned
    return None


def _parse_response(text: str) -> tuple[str | None, str | None]:
    text = text.strip()
    if not text:
        return None, None

    comparison: str | None = None
    winner: str | None = None

    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            raw_comparison = payload.get("comparison") or payload.get("Comparison")
            if isinstance(raw_comparison, str):
                comparison = raw_comparison.strip()
            raw_winner = payload.get("winner") or payload.get("Winner")
            if isinstance(raw_winner, str):
                winner = _normalize_winner(raw_winner)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        lower = line.lower()
        if lower.startswith("comparison:"):
            comparison = line.split(":", 1)[1].strip()
        elif lower.startswith("winner:"):
            winner = _normalize_winner(line.split(":", 1)[1])

    if winner is None:
        match = re.search(r'"winner"\s*:\s*"(A|B|C|TIE)"', text, re.IGNORECASE)
        if match:
            winner = match.group(1).upper()

    if winner is None:
        match = re.search(r"\b(A|B|C|TIE)\b", text, re.IGNORECASE)
        if match:
            winner = match.group(1).upper()

    return comparison, winner


def _init_counts() -> dict[str, int]:
    return {"sft": 0, "og_dpo": 0, "dpo": 0, "TIE": 0}


def _record_count(counts: dict[str, int], winner_key: str | None) -> None:
    if not winner_key:
        counts["TIE"] += 1
        return
    if winner_key.upper() == "TIE":
        counts["TIE"] += 1
        return
    if winner_key not in counts:
        counts[winner_key] = 0
    counts[winner_key] += 1


def _summarize(counts: dict[str, int], total: int) -> dict[str, Any]:
    if total == 0:
        return {"total": 0, "counts": counts, "win_rates": {}}
    win_rates = {key: value / total for key, value in counts.items()}
    return {"total": total, "counts": counts, "win_rates": win_rates}


def _seed_for_instruction(seed: int | None, instruction: str) -> int | None:
    if seed is None:
        return None
    digest = hashlib.sha256(f"{seed}-{instruction}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _call_gpt4_oracle(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    system_prompt: str | None = None,
) -> tuple[str, dict[str, Any]]:
    attempt = 0
    backoff = initial_backoff
    last_error: Exception | None = None

    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    while attempt <= max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            usage = {}
            if response.usage:
                usage = (
                    response.usage.model_dump()
                    if hasattr(response.usage, "model_dump")
                    else dict(response.usage)
                )
            return text, usage
        except Exception as exc:  # Broad to handle API/network errors.
            last_error = exc
            attempt += 1
            if attempt > max_retries:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    raise RuntimeError(f"OpenAI API failed after {max_retries} retries") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use GPT-4o to directly judge three output files."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="test/gpt_judge/config_evaluation.yaml",
        help="Path to evaluation config YAML.",
    )
    parser.add_argument(
        "--sft",
        type=str,
        default=None,
        help="Override path to SFT outputs JSON.",
    )
    parser.add_argument(
        "--og_dpo",
        type=str,
        default=None,
        help="Override path to original DPO outputs JSON.",
    )
    parser.add_argument(
        "--dpo",
        type=str,
        default=None,
        help="Override path to DPO outputs JSON.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override GPT-4o model name for judging.",
    )
    parser.add_argument(
        "--max_instances",
        "--max_examples",
        dest="max_examples",
        type=int,
        default=None,
        help="Limit the number of instructions to judge.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed for output order randomization.",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Override path to write per-example judgments.",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default=None,
        help="Override path to write the summary JSON.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip instructions already present in the results file.",
    )

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    config = _load_config(args.config)
    oracle_cfg = config.get("gpt4_oracle", {})
    inputs_cfg = config.get("inputs", {})
    output_cfg = config.get("output", {})

    prompt_template = oracle_cfg.get("prompt_template")
    if not prompt_template:
        raise ValueError("gpt4_oracle.prompt_template must be set in config.")

    sft_path = args.sft or inputs_cfg.get("sft")
    og_dpo_path = args.og_dpo or inputs_cfg.get("og_dpo")
    dpo_path = args.dpo or inputs_cfg.get("dpo")
    if not sft_path or not og_dpo_path or not dpo_path:
        raise ValueError("inputs.sft, inputs.og_dpo, and inputs.dpo must be set.")

    results_path = Path(
        args.results_file
        or output_cfg.get("results_file", "test/gpt_judge/results/gpt4o_judgments.jsonl")
    )
    summary_path = Path(
        args.summary_file
        or output_cfg.get("summary_file", "test/gpt_judge/results/summary.json")
    )

    model_name = args.model or oracle_cfg.get("model", "gpt-4o-2024-08-06")
    temperature = oracle_cfg.get("temperature", 0.0)
    max_tokens = oracle_cfg.get("max_tokens", 256)
    seed = args.seed if args.seed is not None else oracle_cfg.get("seed", 42)
    max_retries = oracle_cfg.get("max_retries", 5)
    initial_backoff = oracle_cfg.get("initial_backoff", 1.0)
    max_backoff = oracle_cfg.get("max_backoff", 60.0)
    system_prompt = oracle_cfg.get("system_prompt")

    max_examples = (
        args.max_examples
        if args.max_examples is not None
        else oracle_cfg.get("max_examples")
    )

    sft_map = _load_outputs(Path(sft_path))
    og_dpo_map = _load_outputs(Path(og_dpo_path))
    dpo_map = _load_outputs(Path(dpo_path))

    instructions = _intersection_keys(sft_map, og_dpo_map, dpo_map)
    if max_examples is not None:
        instructions = instructions[:max_examples]

    seen = set()
    counts = _init_counts()
    if args.resume and results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instruction = row.get("instruction")
                if not instruction or instruction in seen:
                    continue
                seen.add(instruction)
                winner_key = row.get("winner_key") or row.get("winner")
                _record_count(counts, winner_key)

    client = OpenAI()

    results_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(counts.values())

    with results_path.open("a", encoding="utf-8") as out_f:
        for instruction in tqdm(instructions, desc="Judging", unit="examples"):
            outputs = [
                ("sft", sft_map[instruction]),
                ("og_dpo", og_dpo_map[instruction]),
                ("dpo", dpo_map[instruction]),
            ]
            instruction_seed = _seed_for_instruction(seed, instruction)
            rng = random.Random(instruction_seed)
            rng.shuffle(outputs)
            label_map = {"A": outputs[0][0], "B": outputs[1][0], "C": outputs[2][0]}
            labeled_outputs = {
                "A": outputs[0][1],
                "B": outputs[1][1],
                "C": outputs[2][1],
            }

            if instruction in seen:
                continue

            prompt = _build_prompt(prompt_template, instruction, labeled_outputs)
            content, usage = _call_gpt4_oracle(
                client=client,
                prompt=prompt,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                system_prompt=system_prompt,
            )
            comparison, winner = _parse_response(content)
            if winner is None:
                winner = "TIE"

            winner_key = label_map.get(winner, winner)
            _record_count(counts, winner_key)
            total += 1

            out_f.write(
                json.dumps(
                    {
                        "instruction": instruction,
                        "comparison": comparison,
                        "winner": winner,
                        "winner_key": winner_key,
                        "labels": label_map,
                        "model": model_name,
                        "raw_response": content,
                        "usage": usage,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out_f.flush()

    summary = _summarize(counts, total)
    summary["model"] = model_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print(f"Judged {total} instructions.")
    print(summary)


if __name__ == "__main__":
    main()
