"""Test cho src/mood/io.py: chuẩn hóa tag và tính hợp lệ của bảng bài hát."""

from __future__ import annotations

import pandas as pd
import pytest

from src.mood.config import SEED
from src.mood.io import REQUIRED_COLUMNS, load_tracks, parse_tags


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("rock, alternative, indie", ["rock", "alternative", "indie"]),
        ("  Rock ,ALTERNATIVE ", ["rock", "alternative"]),          # chữ thường + bỏ khoảng trắng
        ("rock, rock, Rock", ["rock"]),                              # bỏ tag trùng
        ("rock,, ,indie", ["rock", "indie"]),                        # bỏ tag rỗng
        ("alternative_rock", ["alternative_rock"]),                  # giữ nguyên dấu gạch dưới
        ("", []),
        (None, []),
        (float("nan"), []),
    ],
)
def test_parse_tags(raw, expected):
    assert parse_tags(raw) == expected


@pytest.fixture(scope="module")
def tracks() -> pd.DataFrame:
    return load_tracks(sample=True)


def test_required_columns(tracks):
    assert not [c for c in REQUIRED_COLUMNS if c not in tracks.columns]
    assert "tags_list" in tracks.columns


def test_track_id_unique(tracks):
    assert tracks["track_id"].notna().all()
    assert not tracks["track_id"].duplicated().any()


def test_tags_list_normalized(tracks):
    for tag_list in tracks["tags_list"]:
        assert isinstance(tag_list, list)
        assert all(t == t.strip().lower() and t for t in tag_list)
        assert len(tag_list) == len(set(tag_list))


def test_tags_string_matches_list(tracks):
    """Cột `tags` sau khi đọc phải khớp với `tags_list`."""
    for tags, tag_list in zip(tracks["tags"], tracks["tags_list"]):
        assert parse_tags(tags) == tag_list


def test_cleaning_rules(tracks):
    assert (tracks["tempo"] > 0).all()                               # tempo = 0 đã được điền
    assert tracks["tempo"].notna().all()
    if "time_signature" in tracks.columns:
        assert not (tracks["time_signature"] == 0).any()
    if "year" in tracks.columns:
        assert not (tracks["year"] < 1950).any()
    assert tracks["duration_outlier"].dtype == bool
    assert tracks["spotify_id_dup"].dtype == bool


def test_sample_is_reproducible():
    """Cùng seed phải cho cùng bản mẫu."""
    assert SEED == 42
    a = load_tracks(sample=True)["track_id"].tolist()
    b = load_tracks(sample=True)["track_id"].tolist()
    assert a == b
