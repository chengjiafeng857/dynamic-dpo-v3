"""YAML config loading utilities."""

from typing import Any, Dict, Optional

import torch
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping at the top level.")
    return data


def resolve_torch_dtype(precision: Any) -> Optional[torch.dtype]:
    """Map repo precision strings to torch dtypes for model loading."""
    normalized = str(precision).lower()
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return None
