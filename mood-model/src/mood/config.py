"""Đọc cấu hình module tâm trạng từ configs/mood_config.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Thư mục gốc của module (mood-model/)
ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs" / "mood_config.yaml"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

# Mã 5 nhóm tâm trạng, dùng thống nhất toàn module
GROUPS = ["vui", "buon", "thu_gian", "nang_dong", "cang_thang"]

SEED = 42


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Trả về toàn bộ cấu hình dưới dạng dict (có cache)."""
    cfg_path = Path(path) if path is not None else CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("groups") != GROUPS:
        raise ValueError(
            f"groups trong config ({cfg.get('groups')}) khác GROUPS ({GROUPS})"
        )
    return cfg
