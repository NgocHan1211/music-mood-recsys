"""Test cho src/mood/features.py: không rò rỉ tag, mã hóa key, lọc tập huấn luyện."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mood.config import load_config
from src.mood.features import build_features, feature_columns, training_set
from src.mood.io import load_tracks
from src.mood.labeling import make_labels


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def data(cfg):
    df = load_tracks(sample=True)
    return df, make_labels(df, cfg)


def test_no_tag_or_genre_leakage(cfg, data):
    """Đặc trưng đầu vào không được chứa tag, genre hay artist (quy tắc CLAUDE.md)."""
    df, _ = data
    X = build_features(df, cfg)
    forbidden = {"tags", "tags_list", "n_tags", "genre", "artist", "track_id", "name", "year"}
    assert not (set(X.columns) & forbidden)
    assert set(X.columns) == set(feature_columns(cfg))


def test_key_is_cyclic(cfg, data):
    """key 11 và key 0 phải gần nhau như key 0 và key 1."""
    df, _ = data
    sub = df.head(1).copy()
    def enc(k):
        s = sub.copy(); s["key"] = k
        row = build_features(s, cfg).iloc[0]
        return np.array([row["key_sin"], row["key_cos"]])
    d_11_0 = np.linalg.norm(enc(11) - enc(0))
    d_0_1 = np.linalg.norm(enc(0) - enc(1))
    assert d_11_0 == pytest.approx(d_0_1, rel=1e-9)


def test_no_nan_and_numeric(cfg, data):
    df, _ = data
    X = build_features(df, cfg)
    assert not X.isna().any().any()
    assert all(np.issubdtype(t, np.number) for t in X.dtypes)
    assert len(X) == len(df)


def test_training_set_only_labeled(cfg, data):
    """Tập huấn luyện chỉ gồm bài labeled; conflict/suspect/no_emotion_tag bị loại."""
    df, labels = data
    X, y, groups, ids = training_set(df, labels, cfg)
    n_labeled = int((labels["label_status"] == "labeled").sum())
    assert len(X) == len(y) == len(groups) == len(ids) == n_labeled
    assert y.notna().all()
    excluded = set(labels.loc[labels["label_status"] != "labeled", "track_id"])
    assert not (set(ids) & excluded)


def test_groups_are_artists(cfg, data):
    """groups dùng để chia fold phải là artist, và có nhiều hơn số fold."""
    df, labels = data
    _, _, groups, _ = training_set(df, labels, cfg)
    assert groups.nunique() > cfg["classifier"]["n_splits"]
