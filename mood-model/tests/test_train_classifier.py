"""Test cho các hàm phụ của src/mood/train_classifier.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mood.config import GROUPS, load_config
from src.mood.train_classifier import (ManualCentroidClassifier,
                                       expected_calibration_error, score_table)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_manual_centroid_predicts_nearest(cfg):
    """Bài nằm đúng tại tọa độ tự đặt của nhóm nào thì phải được gán nhóm đó."""
    centroids = cfg["manual_centroids"]
    clf = ManualCentroidClassifier(centroids, cfg["scoring"]["sigma"]).fit(None)
    X = pd.DataFrame([centroids[g] for g in GROUPS], columns=["valence", "energy"])
    assert list(clf.predict(X)) == GROUPS


def test_manual_centroid_proba_is_distribution(cfg):
    clf = ManualCentroidClassifier(cfg["manual_centroids"], cfg["scoring"]["sigma"]).fit(None)
    X = pd.DataFrame({"valence": [0.1, 0.5, 0.9], "energy": [0.2, 0.5, 0.8]})
    p = clf.predict_proba(X)
    assert p.shape == (3, len(GROUPS))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all() and (p <= 1).all()


def test_score_table_perfect_prediction():
    y = pd.Series(GROUPS)
    proba = np.eye(len(GROUPS))
    row = score_table(y, proba)
    assert row["macro_f1"] == pytest.approx(1.0)
    assert row["accuracy"] == pytest.approx(1.0)
    for g in GROUPS:
        assert row[f"f1_{g}"] == pytest.approx(1.0)


def test_ece_zero_when_perfectly_calibrated():
    """Dự đoán đúng với xác suất 1.0 -> ECE = 0."""
    y = pd.Series(GROUPS)
    proba = np.eye(len(GROUPS))
    assert expected_calibration_error(y, proba) == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    """Luôn tự tin 100% nhưng chỉ đúng một nửa -> ECE khoảng 0.5."""
    n = 100
    y = pd.Series([GROUPS[0]] * (n // 2) + [GROUPS[1]] * (n // 2))
    proba = np.zeros((n, len(GROUPS)))
    proba[:, 0] = 1.0
    assert expected_calibration_error(y, proba) == pytest.approx(0.5, abs=0.01)


def test_log_loss_column_order():
    """log_loss phải khớp đúng cột xác suất với nhãn (sklearn sắp nhãn theo chữ cái)."""
    from sklearn.metrics import log_loss as sk_log_loss

    y = pd.Series(GROUPS)
    proba = np.full((len(GROUPS), len(GROUPS)), 0.02)
    np.fill_diagonal(proba, 1 - 0.02 * (len(GROUPS) - 1))
    row = score_table(y, proba)
    # dự đoán gần như hoàn hảo -> log-loss phải rất nhỏ
    assert row["log_loss"] < 0.15
    order = np.argsort(GROUPS)
    assert row["log_loss"] == pytest.approx(
        sk_log_loss(y, proba[:, order], labels=sorted(GROUPS)))
