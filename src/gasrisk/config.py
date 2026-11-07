"""Configuration loading.

The configuration lives in ``config/config.yaml``. Paths in it are relative to
the repository root and are resolved here, so no module hardcodes a location.
Set the ``GASRISK_CONFIG`` environment variable to use a different file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def find_repo_root(start: Path | None = None) -> Path:
    """Return the first parent directory that contains ``config/config.yaml``."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config" / "config.yaml").is_file():
            return candidate
    raise FileNotFoundError("Could not locate repository root (config/config.yaml not found).")


@dataclass(frozen=True)
class Config:
    """Parsed configuration with resolved paths."""

    raw: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, key: str) -> Path:
        """Absolute path for an entry of the ``paths`` section, created if missing."""
        p = self.root / self.raw["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML configuration.

    Args:
        path: Optional explicit path. Defaults to ``$GASRISK_CONFIG`` or
            ``<repo root>/config/config.yaml``.
    """
    env = os.environ.get("GASRISK_CONFIG")
    if path is None and env:
        path = env
    if path is None:
        root = find_repo_root()
        path = root / "config" / "config.yaml"
    else:
        path = Path(path).resolve()
        root = find_repo_root(path.parent)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=root)
