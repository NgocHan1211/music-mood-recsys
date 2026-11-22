"""Kiểm thử L1 (gán nhãn bằng LLM) — các phần không cần gọi API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mood.config import GROUPS
from src.mood.labeling_llm import (GROUP_DESC, SCHEMA, UNKNOWN, VARIANTS, audio_group,
                                   audio_match, build_prompt, cache_csv,
                                   expanded_labels, popularity_bins)


def _rows():
    return pd.DataFrame({
        "track_id": ["t1", "t2"],
        "name": ["Bohemian Rhapsody", "Wonderwall"],
        "artist": ["Queen", "Oasis"],
        "tags": ["rock, classic_rock", None],
    })


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

def test_prompt_has_name_and_artist_numbered_from_zero():
    p = build_prompt(_rows(), with_tags=False)
    assert '0. "Bohemian Rhapsody" — Queen' in p
    assert '1. "Wonderwall" — Oasis' in p


def test_prompt_hides_tags_in_name_only_variant():
    assert "tag:" not in build_prompt(_rows(), with_tags=False)


def test_prompt_shows_tags_when_asked_and_skips_empty_tags():
    """Bài không có tag không được gửi đi dưới dạng "[tag: nan]"."""
    p = build_prompt(_rows(), with_tags=True)
    assert "[tag: rock, classic_rock]" in p
    assert "nan" not in p
    assert p.splitlines()[-1].endswith("— Oasis")


def test_schema_enum_matches_groups_plus_unknown():
    enum = SCHEMA["properties"]["ket_qua"]["items"]["properties"]["nhom"]["enum"]
    assert set(enum) == set(GROUPS) | {UNKNOWN}


def test_group_desc_covers_all_groups():
    assert set(GROUP_DESC) == set(GROUPS)


def test_two_variants_differ_only_in_tag_visibility():
    assert VARIANTS == {"llm_ten": False, "llm_ten_tag": True}


def test_cache_path_is_per_variant_and_stage():
    assert cache_csv("llm_ten", "calib") != cache_csv("llm_ten", "expand")
    assert cache_csv("llm_ten", "calib") != cache_csv("llm_ten_tag", "calib")


# --------------------------------------------------------------------------- #
# Thước đo
# --------------------------------------------------------------------------- #

def test_audio_match_ignores_unknown_rows():
    pred = pd.Series([GROUPS[0], UNKNOWN, GROUPS[1]], index=["a", "b", "c"])
    audio = pd.Series([GROUPS[0], GROUPS[0], GROUPS[0]], index=["a", "b", "c"])
    assert audio_match(pred, audio) == pytest.approx(0.5)   # chỉ a và c được tính


def test_audio_match_returns_nan_when_everything_unknown():
    pred = pd.Series([UNKNOWN, UNKNOWN], index=["a", "b"])
    assert np.isnan(audio_match(pred, pd.Series([GROUPS[0]] * 2, index=["a", "b"])))


def test_audio_group_returns_a_real_group_for_every_track():
    df = pd.DataFrame({"valence": [0.1, 0.9, 0.5], "energy": [0.1, 0.9, 0.5]})
    out = audio_group(df)
    assert len(out) == 3 and set(out) <= set(GROUPS)


def test_popularity_bins_splits_into_four():
    s = pd.Series(range(100))
    assert popularity_bins(s).nunique() == 4


# --------------------------------------------------------------------------- #
# Ghép nhãn
# --------------------------------------------------------------------------- #

def _hand():
    return pd.DataFrame({
        "track_id": ["t1", "t2", "t3", "t4"],
        "mood_label": [GROUPS[0], None, None, GROUPS[1]],
        "label_status": ["labeled", "no_emotion_tag", "conflict", "suspect"],
    })


def test_expanded_labels_fills_only_unlabeled_and_keeps_hand_labels():
    llm = pd.DataFrame({"track_id": ["t1", "t2", "t3", "t4"],
                        "llm_nhom": [GROUPS[4], GROUPS[2], GROUPS[3], GROUPS[4]]})
    out = expanded_labels(_hand(), llm).set_index("track_id")
    assert out.loc["t1", "mood_label"] == GROUPS[0]       # nhãn tay không bị đè
    assert out.loc["t2", "mood_label"] == GROUPS[2]       # no_emotion_tag -> điền
    assert out.loc["t3", "mood_label"] == GROUPS[3]       # conflict -> điền
    assert out.loc["t4", "mood_label"] == GROUPS[1]       # suspect -> giữ nguyên
    assert out.loc["t2", "label_status"] == "labeled"


def test_expanded_labels_skips_unknown_predictions():
    llm = pd.DataFrame({"track_id": ["t2", "t3"], "llm_nhom": [UNKNOWN, GROUPS[3]]})
    out = expanded_labels(_hand(), llm).set_index("track_id")
    assert pd.isna(out.loc["t2", "mood_label"])
    assert out.loc["t2", "label_status"] == "no_emotion_tag"
    assert out.loc["t3", "mood_label"] == GROUPS[3]


def test_expanded_labels_does_not_mutate_input():
    hand = _hand()
    expanded_labels(hand, pd.DataFrame({"track_id": ["t2"], "llm_nhom": [GROUPS[2]]}))
    assert pd.isna(hand.loc[1, "mood_label"])


# --------------------------------------------------------------------------- #
# Backend cắm thay được
# --------------------------------------------------------------------------- #

def test_every_backend_has_a_model_a_batch_size_and_a_classifier():
    from src.mood.labeling_llm import BACKENDS, BATCH_SIZE, CLASSIFIERS
    assert set(BACKENDS) == set(BATCH_SIZE) == set(CLASSIFIERS)


def test_cache_path_separates_backends():
    assert cache_csv("llm_ten", "calib", "mlx") != cache_csv("llm_ten", "calib", "anthropic")


def test_compact_system_prompt_states_the_line_format_and_drops_json():
    from src.mood.labeling_llm import SYSTEM_COMPACT
    assert "<số thứ tự>:<tên nhóm>" in SYSTEM_COMPACT
    assert "JSON" not in SYSTEM_COMPACT


def test_parse_lines_reads_compact_format():
    from src.mood.labeling_llm import _parse_lines
    out = _parse_lines(f"0:{GROUPS[0]}\n1: {GROUPS[1]}\n2) {GROUPS[2]}\n", 3)
    assert out == {0: GROUPS[0], 1: GROUPS[1], 2: GROUPS[2]}


def test_parse_lines_drops_invalid_groups_and_out_of_range_indices():
    from src.mood.labeling_llm import _parse_lines
    out = _parse_lines(f"0:khong_phai_nhom\n1:{GROUPS[0]}\n99:{GROUPS[1]}\n", 2)
    assert out == {1: GROUPS[0]}


def test_parse_lines_survives_chatty_output():
    from src.mood.labeling_llm import _parse_lines
    text = f"Đây là kết quả:\n\n- 0: {GROUPS[3]}\n- 1: {GROUPS[4]}\nHết."
    assert _parse_lines(text, 2) == {0: GROUPS[3], 1: GROUPS[4]}


def test_throttle_blocks_once_the_minute_budget_is_spent():
    """Bộ điều tiết phải chặn lượt vượt hạn mức, rồi cho đi khi cửa sổ 60 giây trôi qua."""
    import time
    import src.mood.labeling_llm as L
    L._SPENT.clear()
    L._throttle(L.TOKENS_PER_MIN)             # tiêu hết hạn mức
    L._SPENT[:] = [(time.time() - 59.9, L.TOKENS_PER_MIN)]
    t0 = time.time()
    L._throttle(100)                          # phải đợi phần còn lại của cửa sổ
    assert 0.0 < time.time() - t0 < 2.0
    L._SPENT.clear()


def test_throttle_lets_small_requests_through_immediately():
    import time
    import src.mood.labeling_llm as L
    L._SPENT.clear()
    t0 = time.time()
    L._throttle(10)
    assert time.time() - t0 < 0.1
    L._SPENT.clear()


def test_cache_path_separates_models_on_the_same_backend():
    """Hai mô hình khác nhau cùng backend không được dùng chung file cache."""
    a = cache_csv("llm_ten", "calib", "openai_compat", "openai/gpt-oss-120b")
    b = cache_csv("llm_ten", "calib", "openai_compat", "qwen/qwen3.8-27b")
    assert a != b
    assert "gpt-oss-120b" in a.name and "qwen3.8-27b" in b.name


def test_model_tag_strips_provider_prefix_and_odd_characters():
    from src.mood.labeling_llm import model_tag
    assert model_tag("openai/gpt-oss-120b") == "gpt-oss-120b"
    assert model_tag("Qwen/Qwen2.5-3B-Instruct") == "Qwen2.5-3B-Instruct"
    assert "/" not in model_tag("a/b:c d")
