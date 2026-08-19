"""
Configuration loader.

Single source of truth for every tunable in the LightGBM track. Nothing in this
package reads a magic number from anywhere else.
"""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _PACKAGE_ROOT / "config" / "config.yaml"


class Config:
    """Thin dotted-access wrapper over the YAML config."""

    def __init__(self, data: Dict[str, Any], path: Path):
        self._data = data
        self.path = path

    # -- access -------------------------------------------------------------
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
    def raw(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    # -- convenience --------------------------------------------------------
    @property
    def package_root(self) -> Path:
        return _PACKAGE_ROOT

    @property
    def raw_data_dir(self) -> Path:
        return Path(self["paths.raw_data_dir"]).expanduser()

    @property
    def output_dir(self) -> Path:
        raw = Path(self["paths.output_dir"]).expanduser()
        return raw if raw.is_absolute() else _PACKAGE_ROOT / raw

    @property
    def all_regions(self) -> List[str]:
        return list(self["regions.all"])

    @property
    def default_regions(self) -> List[str]:
        return list(self["regions.default_run"])

    @property
    def horizons(self) -> List[int]:
        return sorted(int(h) for h in self["horizons.supported"])


class _Missing:
    pass


_MISSING = _Missing()


def load_config(path: str | os.PathLike | None = None) -> Config:
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
    logging.getLogger("lightgbm").setLevel(logging.WARNING)
