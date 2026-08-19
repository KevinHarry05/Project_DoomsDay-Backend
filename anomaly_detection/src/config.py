"""Configuration loader - identical pattern to the two forecasting packages."""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _PACKAGE_ROOT / "config" / "config.yaml"


class Config:
    def __init__(self, data: Dict[str, Any], path: Path):
        self._data = data
        self.path = path

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return copy.deepcopy(node)

    def __getitem__(self, key: str) -> Any:
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(f"Missing config key: {key}")
        return value

    @property
    def package_root(self) -> Path:
        return _PACKAGE_ROOT

    @property
    def output_dir(self) -> Path:
        raw = Path(self.get("output.dir", "outputs")).expanduser()
        return raw if raw.is_absolute() else _PACKAGE_ROOT / raw


class _Missing:
    pass


_MISSING = _Missing()


def load_config(path=None) -> Config:
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data, cfg_path)


def setup_logging(cfg: Config) -> None:
    logging.basicConfig(
        level=getattr(logging, str(cfg.get("logging.level", "INFO")).upper(), logging.INFO),
        format=cfg.get("logging.format", "%(asctime)s | %(levelname)s | %(message)s"),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
