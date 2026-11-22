"""Test cho src/mood/evaluate_mood.py và user_mood_target (T6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mood import api
from src.mood.config import GROUPS, load_config
from src.mood.evaluate_mood import (hide_tracks, ndcg_at_k, recall_at_k, split_users,
                                    target_point)


@pytest.fixture(scope="module")
def profiles():
    try:
        return api.load_mood_profiles()
    except FileNotFoundError:
        pytest.skip("chưa chạy python -m src.mood.predict")


# --- chỉ số xếp hạng ---

def test_recall_at_k():
    ranked = ["a", "b", "c", "d", "e"]
    assert recall_at_k(ranked, {"a", "b"}, 2) == pytest.approx(1.0)
    assert recall_at_k(ranked, {"a", "e"}, 2) == pytest.approx(0.5)
    assert recall_at_k(ranked, {"z"}, 5) == pytest.approx(0.0)
    # nhiều bài liên quan hơn k -> mẫu số là k
    assert recall_at_k(ranked, {"a", "b", "c"}, 2) == pytest.approx(1.0)


def test_ndcg_at_k():
    ranked = ["a", "b", "c"]
    assert ndcg_at_k(ranked, {"a", "b"}, 3) == pytest.approx(1.0)
    # xếp đúng thứ tự tốt hơn xếp sai
    assert ndcg_at_k(["a", "x", "y"], {"a"}, 3) > ndcg_at_k(["x", "y", "a"], {"a"}, 3)
    assert ndcg_at_k(ranked, set(), 3) is np.nan or np.isnan(ndcg_at_k(ranked, set(), 3))


# --- chia dữ liệu ---

def test_split_users_is_reproducible():
    hist = pd.DataFrame({"user_id": [f"u{i}" for i in range(100)],
                         "track_id": ["t"] * 100, "playcount": [1] * 100})
    a = split_users(hist, 0.2)
    b = split_users(hist, 0.2)
    assert len(a) == 20
    assert set(a) == set(b)


def test_hide_tracks_ratio_and_disjoint():
    rng = np.random.default_rng(42)
    tracks = np.array([f"t{i}" for i in range(20)])
    visible, hidden = hide_tracks(tracks, 0.3, rng)
    assert len(hidden) == 6 and len(visible) == 14
    assert not set(visible) & set(hidden)
    assert set(visible) | set(hidden) == set(tracks)


# --- cá nhân hóa (M6) ---

def test_user_mood_target_shrinks_to_global(profiles):
    """Càng ít bài thì tâm cá nhân càng gần tâm chung."""
    cfg = load_config()
    g = np.array(api.quick_mood("cang_thang"))
    pool = profiles[profiles["mood_pred"] == "cang_thang"].index.tolist()
    d_few = np.linalg.norm(np.array(api.user_mood_target(pool[:2], "cang_thang")) - g)
    d_many = np.linalg.norm(np.array(api.user_mood_target(pool[:200], "cang_thang")) - g)
    assert d_few < d_many


def test_user_mood_target_falls_back_to_global(profiles):
    """User không có bài nào thuộc nhóm -> trả về đúng tâm chung."""
    pool = profiles[profiles["mood_pred"] == "thu_gian"].index[:50].tolist()
    assert api.user_mood_target(pool, "cang_thang") == api.quick_mood("cang_thang")
    assert api.user_mood_target([], "vui") == api.quick_mood("vui")


def test_user_mood_target_respects_weights(profiles):
    """Trọng số playcount phải đổi được tâm cá nhân.

    Chú ý: chỉ các bài có P(nhóm | bài) >= min_prob mới được tính, nên phải dồn
    trọng số vào một bài thật sự thuộc nhóm.
    """
    min_prob = load_config()["personalization"]["min_prob"]
    pool = profiles[profiles["mood_pred"] == "vui"].index[:60].tolist()
    qualifying = [t for t in pool if profiles.loc[t, "p_vui"] >= min_prob]
    assert qualifying, "cần ít nhất 1 bài vượt min_prob"
    weights = [100.0 if t == qualifying[0] else 1.0 for t in pool]
    assert api.user_mood_target(pool, "vui") != api.user_mood_target(pool, "vui", weights=weights)


def test_user_mood_target_ignores_low_probability_tracks(profiles):
    """Bài dưới min_prob không ảnh hưởng tới tâm cá nhân."""
    min_prob = load_config()["personalization"]["min_prob"]
    pool = profiles[profiles["mood_pred"] == "vui"].index[:60].tolist()
    weak = [t for t in pool if profiles.loc[t, "p_vui"] < min_prob]
    if not weak:
        pytest.skip("không có bài dưới ngưỡng trong mẫu này")
    weights = [100.0 if t == weak[0] else 1.0 for t in pool]
    assert api.user_mood_target(pool, "vui") == api.user_mood_target(pool, "vui", weights=weights)


def test_user_mood_target_rejects_bad_code(profiles):
    with pytest.raises(ValueError):
        api.user_mood_target(["x"], "happy")


# --- ba cách đặt mục tiêu ---

def test_target_point_kinds_differ(profiles):
    cfg = load_config()
    pool = np.array(profiles[profiles["mood_pred"] == "vui"].index[:50])
    w = np.ones(len(pool))
    manual = target_point("manual", "vui", pool, w, cfg)
    learned = target_point("learned", "vui", pool, w, cfg)
    personal = target_point("personal", "vui", pool, w, cfg)
    assert manual == tuple(cfg["manual_centroids"]["vui"])
    assert learned == api.quick_mood("vui")
    assert manual != learned and learned != personal
    with pytest.raises(ValueError):
        target_point("khong_co", "vui", pool, w, cfg)
