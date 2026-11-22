"""Hàm công khai cho TV2 (task T5 / M5).

Chữ ký hàm theo docs/data_contracts.md — không đổi khi chưa hỏi người dùng.

Ví dụ:
    from src.mood import api
    v, e = api.quick_mood("vui")            # nút chọn nhanh -> điểm thanh kéo
    api.slider_to_mood(v, e)                # điểm thanh kéo -> xác suất 5 nhóm
    api.mood_scores(track_ids, v, e)        # độ hợp tâm trạng [0, 1] cho từng bài
"""

from __future__ import annotations

import json
import warnings
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd

from .config import GROUPS, load_config
from .mood_space import CENTROIDS_JSON, group_probabilities
from .predict import PROB_COLS, TRACK_MOOD_PARQUET

__all__ = ["GROUPS", "load_mood_profiles", "quick_mood", "slider_to_mood",
           "mood_scores", "user_mood_target", "mood_journey"]


@lru_cache(maxsize=1)
def load_mood_profiles() -> pd.DataFrame:
    """Đọc track_mood.parquet, index = track_id."""
    if not TRACK_MOOD_PARQUET.exists():
        raise FileNotFoundError(
            f"Chưa có {TRACK_MOOD_PARQUET}. Chạy: python -m src.mood.predict")
    return pd.read_parquet(TRACK_MOOD_PARQUET).set_index("track_id")


@lru_cache(maxsize=1)
def load_centroids() -> dict:
    """Đọc mood_centroids.json (tâm và hiệp phương sai học được ở T4)."""
    if not CENTROIDS_JSON.exists():
        raise FileNotFoundError(
            f"Chưa có {CENTROIDS_JSON}. Chạy: python -m src.mood.mood_space")
    with open(CENTROIDS_JSON, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _profile_arrays() -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Bảng tra nhanh: track_id -> dòng, ma trận xác suất, ma trận (valence, energy)."""
    df = load_mood_profiles()
    index = {tid: i for i, tid in enumerate(df.index)}
    proba = df[PROB_COLS].to_numpy(dtype=float)
    ve = df[["valence", "energy"]].to_numpy(dtype=float)
    return index, proba, ve


def quick_mood(code: str) -> tuple[float, float]:
    """Nút chọn nhanh -> tọa độ (valence, energy) = tâm học được của nhóm."""
    if code not in GROUPS:
        raise ValueError(f"Mã nhóm không hợp lệ: {code!r}. Hợp lệ: {GROUPS}")
    mean = load_centroids()["centroids"][code]["mean"]
    return float(mean[0]), float(mean[1])


def slider_to_mood(valence: float, energy: float) -> dict[str, float]:
    """Điểm thanh kéo -> xác suất thuộc từng nhóm (tổng = 1)."""
    return group_probabilities(float(valence), float(energy), load_centroids())


def mood_scores(track_ids: Sequence[str], valence: float, energy: float,
                lam: float | None = None) -> np.ndarray:
    """Độ hợp tâm trạng [0, 1] cho từng bài, cùng thứ tự với track_ids.

    mood_score = lam * p_target + (1 - lam) * prox, trong đó:
      - p_target = Σ_m w_m · P(m | bài), w_m từ slider_to_mood (M4),
      - prox = exp(-d² / 2σ²), d = khoảng cách Euclid từ (valence, energy) của bài
        tới điểm thanh kéo.
    Bài không có trong hồ sơ trả về 0 kèm warning.
    """
    cfg = load_config()
    lam = float(cfg["scoring"]["lam"]) if lam is None else float(lam)
    sigma = float(cfg["scoring"]["sigma"])

    index, proba, ve = _profile_arrays()
    rows = np.fromiter((index.get(t, -1) for t in track_ids), dtype=np.int64,
                       count=len(track_ids))
    known = rows >= 0
    n_missing = int((~known).sum())
    if n_missing:
        warnings.warn(f"{n_missing}/{len(rows)} bài không có trong hồ sơ tâm trạng "
                      "-> điểm = 0", stacklevel=2)

    slider = slider_to_mood(valence, energy)   # gọi 1 lần, không gọi lại cho từng nhóm
    w = np.array([slider[g] for g in GROUPS])
    scores = np.zeros(len(rows), dtype=float)
    if known.any():
        idx = rows[known]
        p_target = proba[idx] @ w
        d2 = ((ve[idx] - np.array([valence, energy])) ** 2).sum(axis=1)
        prox = np.exp(-d2 / (2 * sigma ** 2))
        scores[known] = lam * p_target + (1 - lam) * prox
    return np.clip(scores, 0.0, 1.0)


def user_mood_target(user_track_ids: Sequence[str], code: str,
                     weights: Sequence[float] | None = None) -> tuple[float, float]:
    """Tâm trạng mục tiêu cá nhân hóa (M6) cho user và nhóm `code`.

    Lấy các bài user đã nghe có P(code | bài) >= `personalization.min_prob`, tính tâm
    (valence, energy) có trọng số, rồi co về tâm chung theo số bài:

        mu' = (n * mu_user + k * mu_chung) / (n + k),  k = personalization.shrinkage_k

    User không có bài nào thuộc nhóm -> trả về đúng tâm chung.
    """
    if code not in GROUPS:
        raise ValueError(f"Mã nhóm không hợp lệ: {code!r}. Hợp lệ: {GROUPS}")
    cfg = load_config()
    k = float(cfg["personalization"]["shrinkage_k"])
    min_prob = float(cfg["personalization"]["min_prob"])
    global_mean = np.array(quick_mood(code), dtype=float)

    index, proba, ve = _profile_arrays()
    rows = np.fromiter((index.get(t, -1) for t in user_track_ids), dtype=np.int64,
                       count=len(user_track_ids))
    known = rows >= 0
    if not known.any():
        return float(global_mean[0]), float(global_mean[1])

    idx = rows[known]
    w = (np.ones(len(idx)) if weights is None
         else np.asarray(weights, dtype=float)[known])
    g = GROUPS.index(code)
    belongs = proba[idx, g] >= min_prob
    if not belongs.any():
        return float(global_mean[0]), float(global_mean[1])

    idx, w = idx[belongs], w[belongs]
    if w.sum() <= 0:
        w = np.ones(len(idx))
    user_mean = (w[:, None] * ve[idx]).sum(axis=0) / w.sum()

    n = float(len(idx))
    mu = (n * user_mean + k * global_mean) / (n + k)
    return float(mu[0]), float(mu[1])


def mood_journey(start: tuple[float, float], end: tuple[float, float],
                 n_steps: int) -> list[tuple[float, float]]:
    """(M7) Danh sách `n_steps` điểm mục tiêu đi từ `start` tới `end`.

    Nội suy tuyến tính trên mặt phẳng (valence, energy), gồm cả hai đầu mút.
    Dùng để tạo playlist chuyển dần tâm trạng: mỗi điểm chọn một bài có
    `mood_scores` cao nhất chưa được chọn (xem src/mood/journey.py).
    """
    if n_steps < 2:
        raise ValueError("n_steps phải >= 2")
    for name, point in (("start", start), ("end", end)):
        if len(point) != 2:
            raise ValueError(f"{name} phải là (valence, energy)")
    ts = np.linspace(0.0, 1.0, n_steps)
    return [(float(start[0] + (end[0] - start[0]) * t),
             float(start[1] + (end[1] - start[1]) * t)) for t in ts]
