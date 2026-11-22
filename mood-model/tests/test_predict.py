"""Test cho src/mood/predict.py và src/mood/mood_space.py (hợp đồng T4)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.mood.config import GROUPS, load_config
from src.mood.io import load_tracks
from src.mood.mood_space import (CENTROIDS_JSON, build_centroids, group_probabilities,
                                 weighted_gaussian)
from src.mood.predict import PROB_COLS, TRACK_MOOD_PARQUET, blend


@pytest.fixture(scope="module")
def track_mood() -> pd.DataFrame:
    if not TRACK_MOOD_PARQUET.exists():
        pytest.skip("chưa chạy python -m src.mood.predict")
    return pd.read_parquet(TRACK_MOOD_PARQUET)


@pytest.fixture(scope="module")
def centroids() -> dict:
    if not CENTROIDS_JSON.exists():
        pytest.skip("chưa chạy python -m src.mood.mood_space")
    with open(CENTROIDS_JSON, encoding="utf-8") as f:
        return json.load(f)


def test_contract_columns_and_rows(track_mood):
    """track_mood.parquet phải đúng cột và mỗi bài đúng 1 dòng."""
    expected = {"track_id", *PROB_COLS, "mood_pred", "mood_confidence",
                "mood_source", "valence", "energy"}
    assert set(track_mood.columns) == expected
    assert not track_mood["track_id"].duplicated().any()
    assert len(track_mood) == len(load_tracks())


def test_probabilities_sum_to_one(track_mood):
    p = track_mood[PROB_COLS].to_numpy()
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all() and (p <= 1).all()


def test_pred_matches_argmax(track_mood):
    pred = np.array(GROUPS)[track_mood[PROB_COLS].to_numpy().argmax(axis=1)]
    assert (pred == track_mood["mood_pred"].to_numpy()).all()
    assert np.allclose(track_mood["mood_confidence"],
                       track_mood[PROB_COLS].to_numpy().max(axis=1))


def test_mood_source_values(track_mood):
    assert set(track_mood["mood_source"]) <= {"tag", "model"}
    assert (track_mood["mood_source"] == "tag").sum() > 0


def test_blend_weights():
    """alpha = 1 -> hoàn toàn theo tag; alpha = 0 -> hoàn toàn theo mô hình."""
    tag = np.array([[1.0, 0, 0, 0, 0]])
    model = np.array([[0, 1.0, 0, 0, 0]])
    assert np.allclose(blend(tag, model, 1.0), tag)
    assert np.allclose(blend(tag, model, 0.0), model)
    mix = blend(tag, model, 0.7)
    assert mix[0, 0] == pytest.approx(0.7) and mix[0, 1] == pytest.approx(0.3)
    assert np.allclose(mix.sum(axis=1), 1.0)


def test_weighted_gaussian_matches_unweighted():
    """Trọng số bằng nhau -> trùng với trung bình và hiệp phương sai thường."""
    rng = np.random.default_rng(42)
    pts = rng.normal(size=(500, 2))
    mean, cov = weighted_gaussian(pts, np.ones(len(pts)))
    assert np.allclose(mean, pts.mean(axis=0))
    assert np.allclose(cov, np.cov(pts.T, ddof=1), atol=1e-6)


def test_centroids_contract(centroids):
    assert centroids["groups"] == GROUPS
    for g in GROUPS:
        c = centroids["centroids"][g]
        assert len(c["mean"]) == 2
        assert np.array(c["cov"]).shape == (2, 2)
        assert c["n"] > 0
        # hiệp phương sai phải xác định dương
        assert np.all(np.linalg.eigvals(np.array(c["cov"])) > 0)
    assert set(centroids["manual_baseline"]) == set(GROUPS)


def test_slider_probabilities(centroids):
    """Điểm thanh kéo -> phân phối xác suất hợp lệ; tại tâm nhóm nào thì nhóm đó cao."""
    for g in GROUPS:
        v, e = centroids["centroids"][g]["mean"]
        probs = group_probabilities(v, e, centroids)
        assert sum(probs.values()) == pytest.approx(1.0)
        assert all(0 <= p <= 1 for p in probs.values())
        assert max(probs, key=probs.get) == g, f"tại tâm {g} phải nghiêng về {g}"
