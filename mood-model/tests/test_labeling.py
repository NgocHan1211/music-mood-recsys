"""Test cho src/mood/labeling.py: nhãn mềm, mâu thuẫn, nhãn cứng."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mood.config import GROUPS, load_config
from src.mood.io import load_tracks
from src.mood.labeling import SOFT_COLS, build_tag_weights, make_labels, soft_labels


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def labeled(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_tracks(sample=True)
    return df, make_labels(df, cfg)


def test_weights_are_real_tags_and_normalized(cfg, labeled):
    """Mọi tag trong bảng trọng số phải có thật trong dữ liệu (quy tắc 'không bịa tag')."""
    df, _ = labeled
    weights = build_tag_weights(cfg)
    lt = cfg["labeling_tags"]
    for tag, w in weights.items():
        assert set(w) <= set(GROUPS), tag
    # tổng trọng số gốc của mỗi tag = 1 (tag hỗ trợ bị nhân support_weight)
    for tag, w in lt["direct"].items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, tag
    for tag, w in lt["support"].items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, tag
        assert abs(sum(weights[tag].values()) - lt["support_weight"]) < 1e-9, tag


def test_soft_labels_sum_to_one(labeled):
    _, labels = labeled
    has = labels[SOFT_COLS].notna().all(axis=1)
    sums = labels.loc[has, SOFT_COLS].sum(axis=1)
    assert np.allclose(sums, 1.0)
    # không có bài nào NaN một phần
    partial = labels[SOFT_COLS].isna().any(axis=1) & labels[SOFT_COLS].notna().any(axis=1)
    assert not partial.any()


def test_no_emotion_tag_is_nan(labeled):
    """Bài không có tag nào trong bảng trọng số phải có nhãn mềm NaN."""
    _, labels = labeled
    no_tag = labels["label_status"] == "no_emotion_tag"
    assert labels.loc[no_tag, SOFT_COLS].isna().all().all()
    assert labels.loc[no_tag, "mood_label"].isna().all()
    has = labels[SOFT_COLS].notna().all(axis=1)
    assert not (no_tag & has).any()


def test_soft_labels_manual_case(cfg):
    """Ví dụ tính tay: chill (thu_gian 1.0) + metal (cang_thang .7, nang_dong .3) x 0.5."""
    weights = build_tag_weights(cfg)
    s = soft_labels(pd.Series([["chill", "metal"]]), weights).iloc[0]
    sw = cfg["labeling_tags"]["support_weight"]
    total = 1.0 + sw
    assert s["y_thu_gian"] == pytest.approx(1.0 / total)
    assert s["y_cang_thang"] == pytest.approx(0.7 * sw / total)
    assert s["y_nang_dong"] == pytest.approx(0.3 * sw / total)
    assert s["y_vui"] == 0.0 and s["y_buon"] == 0.0


def test_conflict_and_confidence_rules(cfg):
    """Hai nhóm ngang nhau -> conflict; nhóm trội rõ -> labeled."""
    lab = cfg["labeling"]
    tie = pd.DataFrame({"track_id": ["t1"], "tags_list": [["beautiful", "chill"]],
                        "spotify_id_dup": [False], "duration_outlier": [False],
                        "name": ["x"], "artist": ["a"], "year": [2000]})
    out = make_labels(tie, cfg)
    gap = abs(out.loc[0, "y_buon"] - out.loc[0, "y_thu_gian"])
    if gap < lab["conflict_margin"]:
        assert out.loc[0, "label_status"] == "conflict"
        assert out.loc[0, "mood_label"] is None

    clear = tie.copy(); clear["tags_list"] = [["chill", "chillout"]]
    out2 = make_labels(clear, cfg)
    assert out2.loc[0, "mood_label"] == "thu_gian"
    assert out2.loc[0, "label_status"] == "labeled"
    assert out2.loc[0, "y_thu_gian"] >= lab["min_label_confidence"]


def test_label_status_values_and_uniqueness(labeled):
    df, labels = labeled
    assert set(labels["label_status"]) <= {"labeled", "conflict", "suspect", "no_emotion_tag"}
    assert len(labels) == len(df)
    assert not labels["track_id"].duplicated().any()
    # mood_label chỉ nhận mã nhóm hợp lệ
    assert set(labels["mood_label"].dropna()) <= set(GROUPS)


def test_min_tracks_per_group(cfg):
    """Trên toàn bộ dữ liệu, mỗi nhóm phải đạt min_tracks_per_group."""
    df = load_tracks()
    labels = make_labels(df, cfg)
    counts = labels.loc[labels["label_status"] == "labeled", "mood_label"].value_counts()
    for g in GROUPS:
        assert counts.get(g, 0) >= cfg["labeling"]["min_tracks_per_group"], g


def test_labeled_always_has_mood_label(labeled):
    """Mọi bài label_status = 'labeled' phải có mood_label (và ngược lại)."""
    _, labels = labeled
    is_labeled = labels["label_status"] == "labeled"
    assert labels.loc[is_labeled, "mood_label"].notna().all()
    assert labels.loc[labels["mood_label"].notna(), "label_status"].isin(
        {"labeled", "suspect"}
    ).all()
