"""YAML config loading utilities."""

from typing import Any, Dict

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

