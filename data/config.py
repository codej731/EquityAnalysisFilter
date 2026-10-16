"""Load config.yaml and expose its hash for run_log."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass(frozen=True)
class Config:
    raw: dict
    path: Path
    hash: str

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Nested lookup: cfg.get("validation.staleness_months")."""
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def resolve_path(self, dotted_key: str) -> Path:
        """Paths in config are relative to the config file's directory."""
        p = Path(self.get(dotted_key))
        return p if p.is_absolute() else self.path.parent / p


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    content = path.read_bytes()
    return Config(
        raw=yaml.safe_load(content) or {},
        path=path.resolve(),
        hash=hashlib.sha256(content).hexdigest()[:16],
    )
