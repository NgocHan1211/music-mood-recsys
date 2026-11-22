"""Test cho mood_journey và src/mood/journey.py (T7 / M7)."""

from __future__ import annotations

import numpy as np
import pytest

from src.mood import api
from src.mood.journey import build_playlist


def test_journey_endpoints_and_length():
    pts = api.mood_journey((0.2, 0.3), (0.8, 0.7), 5)
    assert len(pts) == 5
    assert pts[0] == pytest.approx((0.2, 0.3))
    assert pts[-1] == pytest.approx((0.8, 0.7))


def test_journey_is_evenly_spaced():
    pts = np.array(api.mood_journey((0.0, 0.0), (1.0, 0.5), 6))
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert np.allclose(steps, steps[0])


def test_journey_monotonic_between_endpoints():
    pts = api.mood_journey((0.31, 0.59), (0.65, 0.64), 8)
    v = [p[0] for p in pts]
    assert v == sorted(v)
    assert all(0.31 <= x <= 0.65 for x in v)


def test_journey_rejects_too_few_steps():
    with pytest.raises(ValueError):
        api.mood_journey((0.2, 0.3), (0.8, 0.7), 1)


def test_playlist_has_no_duplicates_and_moves_toward_target():
    try:
        profiles = api.load_mood_profiles()
    except FileNotFoundError:
        pytest.skip("chưa chạy python -m src.mood.predict")
    candidates = profiles.sample(3000, random_state=42).index.tolist()
    start, end = api.quick_mood("buon"), api.quick_mood("vui")
    pl = build_playlist(start, end, 8, candidates)

    assert len(pl) == 8
    assert not pl["track_id"].duplicated().any()
    # bài chọn cho mỗi bước phải gần điểm mục tiêu của bước đó
    chosen = profiles.loc[pl["track_id"]]
    d = np.hypot(chosen["valence"].to_numpy() - pl["target_valence"].to_numpy(),
                 chosen["energy"].to_numpy() - pl["target_energy"].to_numpy())
    assert d.max() < 0.15
    # valence của bài cuối phải cao hơn bài đầu (đi từ buồn sang vui)
    assert chosen["valence"].iloc[-1] > chosen["valence"].iloc[0]
