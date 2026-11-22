"""Test cho src/mood/api.py — hợp đồng hàm công khai cho TV2 (T5)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.mood import api
from src.mood.config import GROUPS, load_config


@pytest.fixture(scope="module")
def profiles():
    try:
        return api.load_mood_profiles()
    except FileNotFoundError:
        pytest.skip("chưa chạy python -m src.mood.predict")


def test_groups_matches_contract():
    assert api.GROUPS == ["vui", "buon", "thu_gian", "nang_dong", "cang_thang"]


def test_load_mood_profiles_indexed_by_track_id(profiles):
    assert profiles.index.name == "track_id"
    assert not profiles.index.duplicated().any()
    assert {"mood_pred", "mood_confidence", "mood_source", "valence", "energy"} <= set(profiles.columns)


def test_quick_mood_returns_learned_centroid(profiles):
    """quick_mood phải trả về đúng tâm học được trong mood_centroids.json."""
    centroids = api.load_centroids()
    for g in GROUPS:
        v, e = api.quick_mood(g)
        assert (v, e) == tuple(centroids["centroids"][g]["mean"])
        assert 0.0 <= v <= 1.0 and 0.0 <= e <= 1.0


def test_quick_mood_rejects_bad_code(profiles):
    with pytest.raises(ValueError):
        api.quick_mood("happy")


def test_slider_to_mood_is_distribution(profiles):
    for v, e in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.8, 0.2)]:
        probs = api.slider_to_mood(v, e)
        assert set(probs) == set(GROUPS)
        assert sum(probs.values()) == pytest.approx(1.0)
        assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_slider_at_centroid_favours_that_group(profiles):
    for g in GROUPS:
        probs = api.slider_to_mood(*api.quick_mood(g))
        assert max(probs, key=probs.get) == g


def test_mood_scores_range_and_order(profiles):
    ids = profiles.index[:500].tolist()
    s = api.mood_scores(ids, *api.quick_mood("vui"))
    assert isinstance(s, np.ndarray) and s.shape == (500,)
    assert ((s >= 0) & (s <= 1)).all()
    # đảo thứ tự đầu vào -> điểm đảo theo đúng thứ tự đó
    s_rev = api.mood_scores(ids[::-1], *api.quick_mood("vui"))
    assert np.allclose(s_rev, s[::-1])


def test_mood_scores_prefers_matching_tracks(profiles):
    """Bài được dự đoán là cang_thang phải có điểm cao hơn khi thanh kéo ở cang_thang."""
    target = api.quick_mood("cang_thang")
    tense = profiles[profiles["mood_pred"] == "cang_thang"].index[:300].tolist()
    calm = profiles[profiles["mood_pred"] == "thu_gian"].index[:300].tolist()
    assert api.mood_scores(tense, *target).mean() > api.mood_scores(calm, *target).mean()


def test_mood_scores_lambda_effect(profiles):
    """lam = 1 -> chỉ dùng xác suất nhóm; lam = 0 -> chỉ dùng độ gần trên thanh kéo."""
    ids = profiles.index[:200].tolist()
    v, e = api.quick_mood("thu_gian")
    only_p = api.mood_scores(ids, v, e, lam=1.0)
    only_prox = api.mood_scores(ids, v, e, lam=0.0)
    default = api.mood_scores(ids, v, e)
    lam = load_config()["scoring"]["lam"]
    assert np.allclose(default, lam * only_p + (1 - lam) * only_prox)


def test_mood_scores_unknown_track_is_zero_with_warning(profiles):
    ids = [profiles.index[0], "KHONG_CO_TRACK_NAY"]
    with pytest.warns(UserWarning):
        s = api.mood_scores(ids, 0.5, 0.5)
    assert s[1] == 0.0 and s[0] > 0.0


def test_mood_scores_speed_20k(profiles):
    """Yêu cầu hợp đồng: 20.000 bài trong dưới 1 giây."""
    ids = profiles.index[:20000].tolist()
    api.mood_scores(ids[:10], 0.5, 0.5)  # làm nóng cache
    start = time.perf_counter()
    api.mood_scores(ids, 0.5, 0.5)
    assert time.perf_counter() - start < 1.0


def test_user_mood_target_implemented(profiles):
    """T6 đã cài: trả về điểm hợp lệ trên thanh kéo."""
    v, e = api.user_mood_target(profiles.index[:50].tolist(), "vui")
    assert 0.0 <= v <= 1.0 and 0.0 <= e <= 1.0


def test_mood_journey_implemented():
    """T7 đã cài: trả về đúng số điểm, đi từ start tới end."""
    pts = api.mood_journey((0.2, 0.3), (0.8, 0.7), 5)
    assert len(pts) == 5
    assert pts[0] == pytest.approx((0.2, 0.3)) and pts[-1] == pytest.approx((0.8, 0.7))
