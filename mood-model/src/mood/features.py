"""Ma trận đặc trưng âm thanh cho bộ phân loại tâm trạng (task T3 / M3).

QUY TẮC: chỉ dùng đặc trưng âm thanh Spotify. KHÔNG dùng tags (nguồn sinh nhãn →
rò rỉ), không dùng genre, không dùng artist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config

# key được mã hóa sin/cos nên không nằm trong danh sách cột số thẳng
CYCLIC_FEATURE = "key"
N_KEYS = 12


def feature_columns(cfg: dict | None = None) -> list[str]:
    """Tên các cột của ma trận đặc trưng, theo đúng thứ tự dựng."""
    cfg = cfg or load_config()
    cols = [c for c in cfg["classifier"]["features"] if c != CYCLIC_FEATURE]
    if CYCLIC_FEATURE in cfg["classifier"]["features"]:
        cols += ["key_sin", "key_cos"]
    return cols


def build_features(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Dựng ma trận đặc trưng âm thanh.

    `key` (0-11) là biến vòng → mã hóa sin/cos để khoảng cách giữa key 11 và key 0
    bằng khoảng cách giữa key 0 và key 1.
    """
    cfg = cfg or load_config()
    feats = cfg["classifier"]["features"]

    forbidden = {"tags", "tags_list", "genre", "artist", "track_id", "name"}
    assert not (set(feats) & forbidden), "Đặc trưng đầu vào không được chứa tag/genre/artist"

    out = pd.DataFrame(index=df.index)
    for col in feats:
        if col == CYCLIC_FEATURE:
            angle = 2 * np.pi * df[col].astype(float) / N_KEYS
            out["key_sin"] = np.sin(angle)
            out["key_cos"] = np.cos(angle)
        else:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    # Điền thiếu bằng trung vị (rất ít: xem T0)
    out = out.fillna(out.median(numeric_only=True))
    return out[feature_columns(cfg)]


def training_set(df: pd.DataFrame, labels: pd.DataFrame, cfg: dict | None = None):
    """Tập huấn luyện: chỉ bài `label_status == 'labeled'`.

    Bài `conflict`, `suspect` (nghi ghép nhầm phiên bản, gồm cả duration_outlier)
    và `no_emotion_tag` đều bị loại.
    Trả về (X, y, groups, track_ids) với groups = artist để chia fold.
    """
    cfg = cfg or load_config()
    merged = df.merge(labels[["track_id", "mood_label", "label_status"]], on="track_id")
    merged = merged[merged["label_status"] == "labeled"].reset_index(drop=True)

    X = build_features(merged, cfg)
    y = merged["mood_label"].astype(str)
    groups = merged["artist"].astype(str)
    return X, y, groups, merged["track_id"]
