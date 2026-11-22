"""Kiểm thử cho A2 (bốn tập), A1 (nhãn bằng embedding) và A3 (mục tiêu từ bài ví dụ)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mood import api, taste
from src.mood.config import GROUPS, load_config
from src.mood.evaluate_four_sets import SETS, build_candidates
from src.mood.evaluate_mood import hidden_for_user, seed_target
from src.mood.labeling import build_tag_weights
from src.mood.labeling_embed import (GROUP_TEXT, audio_match, hand_plus_embed,
                                     weights_from_z)
from src.mood.splits import CSV, als_frame, hidden_map, user_views


# --------------------------------------------------------------------------- #
# Tầng nối sang taste-model
# --------------------------------------------------------------------------- #

def test_taste_package_loads_without_clashing_with_mood_src():
    """Nạp được gói của TV2 mà không ghi đè gói `src` của mood-model."""
    models = taste.taste_models()
    assert hasattr(models, "ALS") and hasattr(models, "Popularity")
    import src.mood.config as mood_config  # vẫn là module của mood-model
    assert mood_config.GROUPS == GROUPS


def test_combine_scores_requires_same_candidate_set():
    """Hợp đồng của TV2: hai bảng điểm phải phủ đúng cùng tập bài."""
    combine = taste.taste_hybrid().combine_scores
    with pytest.raises(ValueError):
        combine({"a": 1.0, "b": 0.5}, {"a": 1.0}, taste_weight=0.5)


def test_train_frame_removes_hidden_tracks():
    """Bài bị che phải bị loại khỏi dữ liệu huấn luyện gu (chống rò rỉ)."""
    hist = pd.DataFrame({"user_id": ["u1", "u1", "u2"],
                         "track_id": ["t1", "t2", "t1"],
                         "playcount": [3, 1, 5]})
    out = taste.train_frame(hist, np.array(["u1", "u2"]), {"u1": {"t2"}})
    assert set(zip(out["user_id"], out["track_id"])) == {("u1", "t1"), ("u2", "t1")}


# --------------------------------------------------------------------------- #
# Bộ chia tái lập được
# --------------------------------------------------------------------------- #

def test_hidden_split_is_order_independent():
    """Cùng user -> cùng bộ chia, bất kể thứ tự truyền vào hay thứ tự duyệt."""
    tracks = np.array([f"t{i}" for i in range(20)])
    v1, h1 = hidden_for_user("userA", tracks, 0.3)
    v2, h2 = hidden_for_user("userA", tracks[::-1], 0.3)
    assert set(h1) == set(h2) and set(v1) == set(v2)
    assert len(h1) == 6 and len(set(h1) & set(v1)) == 0


def test_hidden_split_differs_between_users():
    tracks = np.array([f"t{i}" for i in range(20)])
    _, ha = hidden_for_user("userA", tracks, 0.3)
    _, hb = hidden_for_user("userB", tracks, 0.3)
    assert set(ha) != set(hb)


# --------------------------------------------------------------------------- #
# A2 — bốn tập ứng viên
# --------------------------------------------------------------------------- #

def _fake_universe(mood: str, n: int = 40):
    other = [g for g in GROUPS if g != mood][0]
    return {mood: np.array([f"m{i}" for i in range(n)]),
            "_other_" + mood: np.array([f"o{i}" for i in range(n)]),
            "_pred": {**{f"m{i}": mood for i in range(n)},
                      **{f"o{i}": other for i in range(n)}}}


def test_four_sets_are_disjoint_and_sized():
    mood = GROUPS[0]
    other = [g for g in GROUPS if g != mood][0]
    hidden = np.array([f"h{i}" for i in range(10)])
    hidden_moods = np.array([mood] * 5 + [other] * 5)
    uni = _fake_universe(mood)
    sets = build_candidates(hidden, hidden_moods, set(hidden), mood, None, uni, 3,
                            np.random.default_rng(0))
    assert sets is not None
    assert all(len(sets[k]) == 3 for k in SETS)
    flat = [t for k in SETS for t in sets[k]]
    assert len(flat) == len(set(flat)) == 12


def test_four_sets_returns_none_when_not_enough_hidden():
    mood = GROUPS[0]
    hidden = np.array(["h0", "h1"])
    hidden_moods = np.array([mood, mood])
    sets = build_candidates(hidden, hidden_moods, set(hidden), mood, None,
                            _fake_universe(mood), 3, np.random.default_rng(0))
    assert sets is None


def test_four_sets_excludes_listened_tracks_from_set3_set4():
    """set3 và set4 chỉ được chứa bài user CHƯA từng nghe."""
    mood = GROUPS[0]
    other = [g for g in GROUPS if g != mood][0]
    hidden = np.array([f"h{i}" for i in range(8)])
    hidden_moods = np.array([mood] * 4 + [other] * 4)
    uni = _fake_universe(mood)
    # giả vờ user đã nghe một nửa kho nhạc
    listened = set(hidden) | {f"m{i}" for i in range(20)} | {f"o{i}" for i in range(20)}
    sets = build_candidates(hidden, hidden_moods, listened, mood, None, uni, 3,
                            np.random.default_rng(1))
    assert sets is not None
    assert not (set(sets[SETS[2]]) & listened)
    assert not (set(sets[SETS[3]]) & listened)


def test_four_sets_is_reproducible():
    mood = GROUPS[0]
    other = [g for g in GROUPS if g != mood][0]
    hidden = np.array([f"h{i}" for i in range(10)])
    hidden_moods = np.array([mood] * 5 + [other] * 5)
    uni = _fake_universe(mood)
    a = build_candidates(hidden, hidden_moods, set(hidden), mood, None, uni, 3,
                         np.random.default_rng(7))
    b = build_candidates(hidden, hidden_moods, set(hidden), mood, None, uni, 3,
                         np.random.default_rng(7))
    assert a == b


# --------------------------------------------------------------------------- #
# A1 — nhãn tag bằng embedding
# --------------------------------------------------------------------------- #

def _z_table():
    return pd.DataFrame(
        [[2.5, -0.4, -0.3, -0.9, -0.9],    # tag rõ nhóm 1
         [0.2, 0.1, 0.1, -0.2, -0.2],      # tag mờ, phải bị loại ở z_min = 1
         [1.2, 1.1, -0.5, -0.9, -0.9]],    # tag trỏ hai nhóm
        index=["tag_manh", "tag_mo", "tag_doi"],
        columns=[f"z_{g}" for g in GROUPS])


def test_weights_argmax_keeps_only_strong_tags():
    w = weights_from_z(_z_table(), z_min=1.0, mode="argmax")
    assert set(w) == {"tag_manh", "tag_doi"}
    assert w["tag_manh"] == {GROUPS[0]: 1.0}


def test_weights_soft_spreads_over_positive_groups_and_sums_to_one():
    w = weights_from_z(_z_table(), z_min=1.0, mode="soft")
    assert set(w["tag_doi"]) == {GROUPS[0], GROUPS[1]}
    assert sum(w["tag_doi"].values()) == pytest.approx(1.0)


def test_z_min_controls_coverage():
    table = _z_table()
    assert len(weights_from_z(table, 0.0)) >= len(weights_from_z(table, 1.0))


def test_hand_plus_embed_never_overwrites_hand_weights():
    cfg = load_config()
    hand = build_tag_weights(cfg)
    a_hand_tag = next(iter(hand))
    emb = {a_hand_tag: {GROUPS[-1]: 1.0}, "tag_moi": {GROUPS[0]: 1.0}}
    merged = hand_plus_embed(hand, emb, support_weight=0.5)
    assert merged[a_hand_tag] == hand[a_hand_tag]       # tag cũ giữ nguyên
    assert merged["tag_moi"] == {GROUPS[0]: 0.5}        # tag mới nhân support_weight


def test_audio_match_counts_argmax_against_audio_position():
    pos = pd.DataFrame({"nhom_theo_amthanh": [GROUPS[0], GROUPS[1]]},
                       index=["t1", "t2"])
    w = {"t1": {GROUPS[0]: 1.0}, "t2": {GROUPS[0]: 1.0}}
    assert audio_match(w, pos) == pytest.approx(0.5)


def test_group_text_covers_all_groups():
    assert set(GROUP_TEXT) == set(GROUPS)


# --------------------------------------------------------------------------- #
# A3 — mục tiêu từ bài ví dụ
# --------------------------------------------------------------------------- #

def test_seed_target_falls_back_to_global_when_too_few_seeds():
    cfg = load_config()
    point, n = seed_target("u1", "vui", np.array(["khong_ton_tai"]), np.array([1.0]),
                           cfg, n_seeds=3)
    assert n == 0
    assert point == api.quick_mood("vui")


def test_seed_target_is_reproducible_and_inside_unit_square():
    cfg = load_config()
    profiles = api.load_mood_profiles()
    mood = "thu_gian"
    g = GROUPS.index(mood)
    col = f"p_{mood}"
    ids = profiles.index[profiles[col] >= 0.6][:30].tolist()
    w = np.ones(len(ids))
    a, na = seed_target("u42", mood, np.array(ids), w, cfg, n_seeds=3)
    b, nb = seed_target("u42", mood, np.array(ids), w, cfg, n_seeds=3)
    assert a == b and na == nb >= 3
    assert 0.0 <= a[0] <= 1.0 and 0.0 <= a[1] <= 1.0


def test_seed_target_uses_only_tracks_of_that_mood():
    """Bài có P(nhóm) dưới ngưỡng không được tham gia tính tâm."""
    cfg = load_config()
    profiles = api.load_mood_profiles()
    mood = "cang_thang"
    strong = profiles.index[profiles[f"p_{mood}"] >= 0.8][:3].tolist()
    weak = profiles.index[profiles[f"p_{mood}"] <= 0.1][:20].tolist()
    only_strong, n1 = seed_target("uX", mood, np.array(strong), np.ones(3), cfg, n_seeds=3)
    mixed, n2 = seed_target("uX", mood, np.array(strong + weak),
                            np.ones(len(strong) + len(weak)), cfg, n_seeds=3)
    assert n1 == 3 and n2 == 3
    assert only_strong == mixed


def test_seed_target_differs_from_personal_target():
    """seed3 không co về tâm chung nên phải khác `user_mood_target`."""
    cfg = load_config()
    profiles = api.load_mood_profiles()
    mood = "buon"
    ids = profiles.index[profiles[f"p_{mood}"] >= 0.6][:25].tolist()
    w = np.ones(len(ids))
    seed_point, n = seed_target("uY", mood, np.array(ids), w, cfg, n_seeds=3)
    personal = api.user_mood_target(ids, mood, weights=w)
    assert n >= 3
    assert seed_point != personal


# --------------------------------------------------------------------------- #
# Kiểm tra đầu-cuối: mô hình gu đã huấn luyện không được thấy phần bị che
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not taste.ALS_NPZ.exists(),
                    reason="chưa chạy python -m src.mood.taste")
def test_trained_als_has_not_seen_hidden_tracks():
    """Rò rỉ = 0: không bài bị che nào nằm trong lịch sử huấn luyện của ALS."""
    from src.mood.config import SEED, load_config
    from src.mood.io import load_history

    cfg = load_config()
    ev = cfg["evaluation"]
    als = taste.load_als("local")
    cohort = pd.read_csv(taste.COHORT_CSV, dtype={"user_id": str})
    test_users = cohort.loc[cohort["is_test"], "user_id"].to_numpy()

    profiles = api.load_mood_profiles()
    hist = load_history(min_user_tracks=int(ev["min_user_tracks"]))
    hist = hist[hist["track_id"].isin(set(profiles.index))]

    sample = np.random.default_rng(SEED).choice(np.sort(test_users), size=40, replace=False)
    sub = hist[hist["user_id"].isin(set(sample))]
    n_hidden = 0
    for user_id, g in sub.groupby("user_id"):
        tracks = g["track_id"].to_numpy()
        if len(tracks) < int(ev["min_user_tracks"]):
            continue
        _, hidden = hidden_for_user(user_id, tracks, float(ev["hidden_ratio"]))
        n_hidden += len(hidden)
        assert not (set(hidden) & als.seen.get(user_id, set()))
        assert user_id in als.user_index
    assert n_hidden > 0


# --------------------------------------------------------------------------- #
# Bộ chia train/test chính thức của nhóm (src/mood/splits.py)
# --------------------------------------------------------------------------- #

def _split_frames():
    train = pd.DataFrame({"user_id": ["u1", "u1", "u1", "u2", "u2"],
                          "track_id": ["t1", "t2", "t3", "t1", "t4"],
                          "playcount": [5, 1, 2, 7, 1],
                          "engagement_score": [0.9, 0.2, 0.4, 0.8, 0.15]})
    test = pd.DataFrame({"user_id": ["u1", "u2"],
                         "track_id": ["t9", "t8"],
                         "playcount": [3, 2],
                         "engagement_score": [0.5, 0.3]})
    return train, test


def test_user_views_splits_visible_from_hidden():
    train, test = _split_frames()
    views = user_views(train, test, np.array(["u1", "u2"]))
    vis, w, hid = views["u1"]
    assert set(vis) == {"t1", "t2", "t3"} and set(hid) == {"t9"}
    assert not (set(vis) & set(hid))
    assert len(w) == len(vis) and w.sum() == 8.0


def test_user_views_skips_users_without_hidden_part():
    train, test = _split_frames()
    train = pd.concat([train, pd.DataFrame({"user_id": ["u3"], "track_id": ["t5"],
                                            "playcount": [1], "engagement_score": [0.1]})])
    views = user_views(train, test, np.array(["u1", "u2", "u3"]))
    assert "u3" not in views


def test_user_views_can_weight_by_engagement():
    train, test = _split_frames()
    views = user_views(train, test, np.array(["u1"]), weight_col="engagement_score")
    _, w, _ = views["u1"]
    assert w.sum() == pytest.approx(1.5)


def test_hidden_map_matches_test_rows():
    _, test = _split_frames()
    assert hidden_map(test) == {"u1": {"t9"}, "u2": {"t8"}}


def test_als_frame_renames_weight_to_playcount():
    train, _ = _split_frames()
    f = als_frame(train, "engagement_score")
    assert list(f.columns) == ["user_id", "track_id", "playcount"]
    assert f["playcount"].iloc[0] == pytest.approx(0.9 * 60.0)
    with pytest.raises(ValueError):
        als_frame(train, "khong_ton_tai")


@pytest.mark.skipif(not CSV["test"].exists(), reason="chưa có data/ của nhóm")
def test_official_split_has_no_overlapping_pairs():
    """Bộ chia của TV1: không cặp (user, bài) nào vừa ở train vừa ở test."""
    from src.mood.splits import official_split
    train, test = official_split()
    pairs = set(map(tuple, test[["user_id", "track_id"]].to_numpy()))
    sample = train.sample(n=200_000, random_state=42)[["user_id", "track_id"]]
    assert not (set(map(tuple, sample.to_numpy())) & pairs)
    assert not (set(test["track_id"]) - set(train["track_id"]))
