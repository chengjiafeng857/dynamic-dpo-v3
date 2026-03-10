"""Shared helpers for the staged AlpacaEval pipeline."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List

import torch
import yaml
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR / "configs"
TEMPLATE_DIR = SCRIPT_DIR / "templates"
RUNS_DIR = SCRIPT_DIR / "runs"
DEFAULT_DATASET_REPO = "tatsu-lab/alpaca_eval"
SIMPO_PARITY_VERSION = "0.6.2"

PRESET_REQUIRED_KEYS = (
    "model_name",
    "pretty_name",
    "prompt_template",
    "bos_mode",
    "generation",
    "stop_token_ids",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sanitize_generator_name(value: str) -> str:
    text = str(value)
    return text.replace("\\", "_").replace("/", "_")


def _load_dataset_from_file(data_file: str) -> List[dict[str, Any]]:
    suffix = Path(data_file).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        dataset = load_dataset("json", data_files=data_file)
    elif suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=data_file)
    else:
        raise ValueError(f"Unsupported data file format: {data_file}")

    split = "train" if "train" in dataset else next(iter(dataset.keys()))
    return list(dataset[split])


def _select_dataset_file(repo_id: str) -> str:
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:
        raise RuntimeError(
            "Failed to list AlpacaEval dataset files. Pass --data_file with a local path."
        ) from exc

    candidates = [
        file for file in files if file.lower().endswith((".json", ".jsonl", ".parquet"))
    ]
    if not candidates:
        raise RuntimeError("No JSON/Parquet files found in the AlpacaEval dataset repo.")

    preferred = []
    for file in candidates:
        lowered = file.lower()
        if "alpaca_eval" in lowered and "annotation" not in lowered and "leaderboard" not in lowered:
            preferred.append(file)

    return sorted(preferred or candidates)[0]


def load_alpacaeval_dataset(
    repo_id: str = DEFAULT_DATASET_REPO,
    data_file: str | None = None,
) -> List[dict[str, Any]]:
    resolved_file = data_file or os.getenv("ALPACAEVAL_DATA_FILE")
    if resolved_file is None:
        filename = _select_dataset_file(repo_id)
        resolved_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
    return _load_dataset_from_file(resolved_file)


def resolve_preset_path(preset_name_or_path: str) -> Path:
    candidate = Path(preset_name_or_path)
    if candidate.exists():
        return candidate.resolve()

    if candidate.suffix:
        resolved = CONFIG_DIR / candidate.name
    else:
        resolved = CONFIG_DIR / f"{candidate.name}.yaml"
    if not resolved.exists():
        raise FileNotFoundError(f"Preset not found: {preset_name_or_path}")
    return resolved.resolve()


def load_preset(preset_name_or_path: str) -> dict[str, Any]:
    preset_path = resolve_preset_path(preset_name_or_path)
    with open(preset_path, "r", encoding="utf-8") as handle:
        preset = yaml.safe_load(handle) or {}

    missing = [key for key in PRESET_REQUIRED_KEYS if key not in preset]
    if missing:
        raise ValueError(f"Preset {preset_path.name} missing required keys: {', '.join(missing)}")

    prompt_template = str(preset["prompt_template"])
    template_path = (preset_path.parent / prompt_template).resolve()
    if not template_path.exists():
        template_path = (SCRIPT_DIR / prompt_template).resolve()
    if not template_path.exists():
        raise FileNotFoundError(
            f"Prompt template {prompt_template!r} referenced by {preset_path.name} was not found."
        )

    generation = dict(preset.get("generation") or {})
    normalized = dict(preset)
    normalized["preset_name"] = preset_path.stem
    normalized["preset_path"] = str(preset_path)
    normalized["prompt_template"] = str(prompt_template)
    normalized["prompt_template_path"] = str(template_path)
    normalized["prompt_template_text"] = template_path.read_text(encoding="utf-8")
    normalized["generation"] = generation
    normalized["stop_token_ids"] = list(preset.get("stop_token_ids") or [])
    return normalized


def ensure_run_dir(run_name: str, output_root: str | None = None, *, force: bool = False) -> Path:
    root = Path(output_root).resolve() if output_root else RUNS_DIR
    run_dir = root / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    return run_dir


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_manifest(run_dir: str | Path) -> dict[str, Any]:
    return load_json(Path(run_dir) / "manifest.json")


def write_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> None:
    write_json(Path(run_dir) / "manifest.json", manifest)


def _normalize_instruction(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "") or "").strip()
    input_text = str(row.get("input", "") or "").strip()
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def render_prompt(
    *,
    template_text: str,
    row: dict[str, Any],
    bos_mode: str,
    tokenizer: Any,
) -> str:
    normalized_instruction = _normalize_instruction(row)
    context = {
        "instruction": normalized_instruction,
        "raw_instruction": str(row.get("instruction", "") or ""),
        "input": str(row.get("input", "") or ""),
    }
    prompt = str(template_text).format(**context)
    bos_token = getattr(tokenizer, "bos_token", None) or ""

    if bos_mode == "single":
        if bos_token:
            while prompt.startswith(bos_token):
                prompt = prompt[len(bos_token) :]
            prompt = f"{bos_token}{prompt}"
        return prompt

    if bos_mode == "none":
        if bos_token:
            while prompt.startswith(bos_token):
                prompt = prompt[len(bos_token) :]
        return prompt

    if bos_mode == "tokenizer":
        if bos_token and not prompt.startswith(bos_token):
            return f"{bos_token}{prompt}"
        return prompt

    raise ValueError(f"Unsupported bos_mode: {bos_mode}")


def _resolve_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def resolve_device(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if device == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _lookup_token_id(tokenizer: Any, token: str) -> int | None:
    vocab = tokenizer.get_vocab()
    if token in vocab:
        return vocab[token]
    added_vocab = tokenizer.get_added_vocab()
    if token in added_vocab:
        return added_vocab[token]
    if token in tokenizer.all_special_tokens:
        return tokenizer.convert_tokens_to_ids(token)
    return None


def resolve_eos_token_id(tokenizer: Any, stop_token_ids: Iterable[int] | None = None) -> int | list[int] | None:
    eos_token_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(int(tokenizer.eos_token_id))
    elif tokenizer.eos_token:
        token_id = _lookup_token_id(tokenizer, tokenizer.eos_token)
        if token_id is not None:
            eos_token_ids.append(token_id)

    special_map = tokenizer.special_tokens_map or {}
    eot_tokens: list[str] = []
    if "eot_token" in special_map:
        eot_tokens.append(special_map["eot_token"])
    additional = special_map.get("additional_special_tokens") or []
    for token in additional:
        lowered = token.lower()
        if "eot" in lowered or "end_of_turn" in lowered:
            eot_tokens.append(token)
    for token in tokenizer.all_special_tokens:
        lowered = token.lower()
        if "eot" in lowered or "end_of_turn" in lowered:
            eot_tokens.append(token)

    for token in dict.fromkeys(eot_tokens):
        token_id = _lookup_token_id(tokenizer, token)
        if token_id is not None and token_id not in eos_token_ids:
            eos_token_ids.append(token_id)

    if not eot_tokens:
        token_id = _lookup_token_id(tokenizer, "<|eot_id|>")
        if token_id is not None and token_id not in eos_token_ids:
            eos_token_ids.append(token_id)

    for token_id in stop_token_ids or []:
        integer_id = int(token_id)
        if integer_id not in eos_token_ids:
            eos_token_ids.append(integer_id)

    if not eos_token_ids:
        return None
    if len(eos_token_ids) == 1:
        return eos_token_ids[0]
    return eos_token_ids


def _batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_prompt_rows(
    dataset_rows: list[dict[str, Any]],
    *,
    preset: dict[str, Any],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    prompt_rows: list[dict[str, Any]] = []
    for index, row in enumerate(dataset_rows):
        prompt = render_prompt(
            template_text=str(preset["prompt_template_text"]),
            row=row,
            bos_mode=str(preset["bos_mode"]),
            tokenizer=tokenizer,
        )
        prompt_rows.append(
            {
                "index": index,
                "instruction": str(row.get("instruction", "") or ""),
                "input": str(row.get("input", "") or ""),
                "prompt": prompt,
            }
        )
    return prompt_rows


def generate_texts_from_prompts(
    prompts: list[str],
    *,
    model_name: str,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    seed: int | None,
    stop_token_ids: list[int] | None,
) -> list[str]:
    resolved_device = resolve_device(device)
    dtype = _resolve_dtype(resolved_device)

    if seed is not None:
        set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_token_id = resolve_eos_token_id(tokenizer, stop_token_ids)

    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if resolved_device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if resolved_device != "cuda":
        model = model.to(resolved_device)
    model.eval()

    outputs: list[str] = []
    do_sample = temperature > 0

    for batch in _batched(prompts, batch_size):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        padded_input_length = inputs["input_ids"].shape[1] if tokenizer.padding_side == "left" else None

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if eos_token_id is not None:
            gen_kwargs["eos_token_id"] = eos_token_id
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)

        for index in range(len(batch)):
            prompt_length = padded_input_length if padded_input_length is not None else input_lengths[index]
            output_ids = generated[index][prompt_length:]
            text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            outputs.append(text)

    return outputs


def serialize_model_outputs(
    dataset_rows: list[dict[str, Any]],
    texts: list[str],
    *,
    generator_name: str,
) -> list[dict[str, Any]]:
    if len(dataset_rows) != len(texts):
        raise ValueError("dataset_rows and texts must have the same length.")

    sanitized_generator = sanitize_generator_name(generator_name)
    outputs: list[dict[str, Any]] = []
    for row, text in zip(dataset_rows, texts):
        output_row = {
            "instruction": str(row.get("instruction", "") or ""),
            "output": str(text),
            "generator": sanitized_generator,
        }
        input_text = str(row.get("input", "") or "")
        if input_text:
            output_row["input"] = input_text
        outputs.append(output_row)
    return outputs


def prepare_run(
    *,
    preset_name: str,
    run_name: str,
    output_root: str | None = None,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    data_file: str | None = None,
    max_instances: int | None = None,
    force: bool = False,
) -> Path:
    preset = load_preset(preset_name)
    dataset_rows = load_alpacaeval_dataset(repo_id=dataset_repo, data_file=data_file)
    if max_instances is not None:
        dataset_rows = dataset_rows[:max_instances]

    run_dir = ensure_run_dir(run_name, output_root, force=force)
    write_json(run_dir / "dataset.json", dataset_rows)
    manifest = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "preset_name": preset["preset_name"],
        "preset": {
            "model_name": preset["model_name"],
            "pretty_name": preset["pretty_name"],
            "prompt_template": preset["prompt_template"],
            "prompt_template_path": preset["prompt_template_path"],
            "bos_mode": preset["bos_mode"],
            "generation": preset["generation"],
            "stop_token_ids": preset["stop_token_ids"],
        },
        "dataset": {
            "repo_id": dataset_repo,
            "data_file": data_file,
            "num_rows": len(dataset_rows),
        },
        "paths": {
            "dataset_file": str(run_dir / "dataset.json"),
            "prompts_file": str(run_dir / "prompts.jsonl"),
            "outputs_file": str(run_dir / "outputs" / "model_outputs.json"),
            "results_dir": str(run_dir / "results"),
        },
        "status": {
            "prepared_at": utc_now_iso(),
            "generated_at": None,
            "judged_at": None,
        },
    }
    write_manifest(run_dir, manifest)
    return run_dir


def generate_run(
    run_dir: str | Path,
    *,
    device: str = "cuda",
    seed: int | None = 42,
    max_input_tokens: int = 2048,
    model_name: str | None = None,
    batch_size: int | None = None,
) -> Path:
    run_path = Path(run_dir).resolve()
    manifest = load_manifest(run_path)
    preset = load_preset(str(manifest["preset_name"]))
    dataset_rows = load_json(run_path / "dataset.json")

    resolved_model_name = str(model_name or preset["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, use_fast=True)
    prompt_rows = build_prompt_rows(dataset_rows, preset=preset, tokenizer=tokenizer)
    prompts_path = run_path / "prompts.jsonl"
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompts_path, "w", encoding="utf-8") as handle:
        for row in prompt_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    generation_cfg = dict(preset["generation"])
    resolved_batch_size = int(batch_size or generation_cfg.get("batch_size", 1))
    texts = generate_texts_from_prompts(
        [row["prompt"] for row in prompt_rows],
        model_name=resolved_model_name,
        batch_size=resolved_batch_size,
        max_input_tokens=max_input_tokens,
        max_new_tokens=int(generation_cfg.get("max_new_tokens", 512)),
        temperature=float(generation_cfg.get("temperature", 0.0)),
        top_p=float(generation_cfg.get("top_p", 1.0)),
        device=device,
        seed=seed,
        stop_token_ids=list(preset.get("stop_token_ids") or []),
    )
    outputs = serialize_model_outputs(
        dataset_rows,
        texts,
        generator_name=str(preset.get("pretty_name") or resolved_model_name),
    )
    outputs_file = run_path / "outputs" / "model_outputs.json"
    write_json(outputs_file, outputs)

    manifest["resolved_model_name"] = resolved_model_name
    manifest["generation_runtime"] = {
        "device": device,
        "seed": seed,
        "max_input_tokens": max_input_tokens,
        "batch_size": resolved_batch_size,
    }
    manifest["status"]["generated_at"] = utc_now_iso()
    write_manifest(run_path, manifest)
    return outputs_file


def parse_leaderboard(results_dir: str | Path, model_name: str) -> dict[str, Any]:
    leaderboard_path = Path(results_dir) / "leaderboard.csv"
    if not leaderboard_path.exists():
        return {}

    with open(leaderboard_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        if model_name in (row.get("name") or ""):
            return dict(row)
    return {}


def build_report(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    manifest = load_manifest(run_path)
    results_dir = Path(manifest["paths"]["results_dir"])
    pretty_name = str(manifest["preset"]["pretty_name"])
    leaderboard_row = parse_leaderboard(results_dir, pretty_name)
    annotations_path = results_dir / "annotations.json"
    annotations = load_json(annotations_path) if annotations_path.exists() else []

    report = {
        "run_name": manifest["run_name"],
        "preset_name": manifest["preset_name"],
        "resolved_model_name": manifest.get("resolved_model_name", manifest["preset"]["model_name"]),
        "dataset_rows": manifest["dataset"]["num_rows"],
        "generated_at": manifest["status"]["generated_at"],
        "judged_at": manifest["status"]["judged_at"],
        "leaderboard": leaderboard_row,
        "num_annotations": len(annotations),
    }
    return report


def format_report(report: dict[str, Any]) -> str:
    leaderboard = report.get("leaderboard") or {}
    lines = [
        f"run_name: {report['run_name']}",
        f"preset_name: {report['preset_name']}",
        f"resolved_model_name: {report['resolved_model_name']}",
        f"dataset_rows: {report['dataset_rows']}",
        f"generated_at: {report['generated_at']}",
        f"judged_at: {report['judged_at']}",
        f"num_annotations: {report['num_annotations']}",
    ]
    if leaderboard:
        lines.append(f"name: {leaderboard.get('name', '')}")
        lines.append(f"win_rate: {leaderboard.get('win_rate', '')}")
        lines.append(f"length_controlled_winrate: {leaderboard.get('length_controlled_winrate', '')}")
        lines.append(f"avg_length: {leaderboard.get('avg_length', '')}")
    return "\n".join(lines)


def get_installed_alpaca_eval_version() -> str | None:
    try:
        return importlib.metadata.version("alpaca-eval")
    except importlib.metadata.PackageNotFoundError:
        return None

