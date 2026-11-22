"""Đọc bảng bài hát và chuẩn hóa tag.

Ưu tiên data/processed/tracks.parquet (TV1 xuất). Nếu chưa có thì đọc
data/raw/Music_Info.csv và làm sạch tối thiểu theo docs/eda_summary.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR, SEED, load_config

# Các cột theo hợp đồng dữ liệu (docs/data_contracts.md)
REQUIRED_COLUMNS = [
    "track_id", "name", "artist", "tags",
    "valence", "energy", "danceability", "acousticness", "instrumentalness",
    "speechiness", "liveness", "loudness", "tempo", "mode", "key", "duration_ms",
]

# Ngưỡng thời lượng bất thường (giây) — khớp 118 bài nêu trong eda_summary.md
MIN_DURATION_S = 30
MAX_DURATION_S = 900

TRACKS_PARQUET = PROCESSED_DIR / "tracks.parquet"
MUSIC_INFO_CSV = RAW_DIR / "Music_Info.csv"
# File Kaggle có thể đặt tên có dấu cách hoặc dấu gạch dưới
HISTORY_CSV_NAMES = ["User_Listening_History.csv", "User Listening History.csv"]


def parse_tags(value: object) -> list[str]:
    """Tách chuỗi tag thành danh sách đã chuẩn hóa (chữ thường, bỏ khoảng trắng).

    Giữ nguyên thứ tự, bỏ tag rỗng và tag trùng.
    """
    if value is None or not isinstance(value, str):
        return []
    tags: list[str] = []
    for raw in value.split(","):
        tag = raw.strip().lower()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _clean_music_info(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch tối thiểu bảng Music_Info.csv theo docs/eda_summary.md."""
    df = df.copy()

    # tempo = 0 -> NaN -> điền trung vị
    df.loc[df["tempo"] == 0, "tempo"] = pd.NA
    df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce")
    df["tempo"] = df["tempo"].fillna(df["tempo"].median())

    # time_signature = 0 -> NaN
    if "time_signature" in df.columns:
        df.loc[df["time_signature"] == 0, "time_signature"] = pd.NA

    # năm < 1950 -> NaN
    if "year" in df.columns:
        df.loc[df["year"] < 1950, "year"] = pd.NA

    # spotify_id trùng -> đánh dấu
    if "spotify_id" in df.columns:
        df["spotify_id_dup"] = df["spotify_id"].duplicated(keep=False)
    else:
        df["spotify_id_dup"] = False

    # thời lượng quá ngắn / quá dài -> đánh dấu (loại khỏi tập huấn luyện ở T3)
    dur_s = df["duration_ms"] / 1000.0
    df["duration_outlier"] = (dur_s < MIN_DURATION_S) | (dur_s > MAX_DURATION_S)

    return df


def load_tracks(sample: bool = False) -> pd.DataFrame:
    """Đọc bảng bài hát, thêm cột `tags_list` đã chuẩn hóa.

    sample=True: lấy ngẫu nhiên `sample_size` bài (seed 42) để chạy thử nhanh.
    """
    cfg = load_config()

    if TRACKS_PARQUET.exists():
        df = pd.read_parquet(TRACKS_PARQUET)
        source = str(TRACKS_PARQUET)
        # tracks.parquet của TV1 đã làm sạch; chỉ bổ sung cờ nếu thiếu
        for col, default in (("spotify_id_dup", False), ("duration_outlier", False)):
            if col not in df.columns:
                if col == "duration_outlier" and "duration_ms" in df.columns:
                    dur_s = df["duration_ms"] / 1000.0
                    df[col] = (dur_s < MIN_DURATION_S) | (dur_s > MAX_DURATION_S)
                else:
                    df[col] = default
    elif MUSIC_INFO_CSV.exists():
        df = pd.read_csv(MUSIC_INFO_CSV, encoding="utf-8-sig")
        df = _clean_music_info(df)
        source = str(MUSIC_INFO_CSV)
    else:
        raise FileNotFoundError(
            f"Không thấy {TRACKS_PARQUET} hoặc {MUSIC_INFO_CSV}. "
            "Đặt Music_Info.csv vào data/raw/ (xem README)."
        )

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source} thiếu cột: {missing}")

    if df["track_id"].duplicated().any():
        n_dup = int(df["track_id"].duplicated().sum())
        raise ValueError(f"{source} có {n_dup} track_id trùng")

    # Chuẩn hóa tag: chữ thường, bỏ khoảng trắng thừa
    df["tags_list"] = df["tags"].map(parse_tags)
    df["n_tags"] = df["tags_list"].map(len)
    df["tags"] = df["tags_list"].map(lambda ts: ", ".join(ts) if ts else None)

    df = df.reset_index(drop=True)

    if sample:
        n = min(int(cfg["sample_size"]), len(df))
        df = df.sample(n=n, random_state=SEED).reset_index(drop=True)

    df.attrs["source"] = source
    return df


def history_path() -> Path:
    """Đường dẫn tới file lịch sử nghe, chấp nhận cả hai cách đặt tên."""
    for name in HISTORY_CSV_NAMES:
        path = RAW_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Không thấy file lịch sử nghe trong {RAW_DIR} (thử: {HISTORY_CSV_NAMES})")


def load_history(min_user_tracks: int = 10, min_track_users: int = 10,
                 playcount_cap: int = 60) -> pd.DataFrame:
    """Đọc User_Listening_History.csv và áp bộ lọc 10/10 theo workflow của TV1.

    - user nghe < `min_user_tracks` bài và bài có < `min_track_users` người nghe: loại.
    - playcount cắt ở `playcount_cap`.
    """
    hist = pd.read_csv(history_path())
    n_users = hist["user_id"].groupby(hist["user_id"]).transform("size")
    n_tracks = hist["track_id"].groupby(hist["track_id"]).transform("size")
    hist = hist[(n_users >= min_user_tracks) & (n_tracks >= min_track_users)]
    hist = hist.copy()
    hist["playcount"] = hist["playcount"].clip(upper=playcount_cap)
    return hist.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Đọc bảng bài hát và in thông tin cơ bản.")
    parser.add_argument("--sample", action="store_true", help="chỉ đọc bản mẫu")
    args = parser.parse_args()

    df = load_tracks(sample=args.sample)
    print(f"Nguồn: {df.attrs['source']}")
    print(f"Số bài: {len(df):,} | Số cột: {df.shape[1]}")
    print(f"Thiếu tag: {int((df['n_tags'] == 0).sum()):,} bài | "
          f"Trung bình tag/bài: {df['n_tags'].mean():.2f}")
    print(f"spotify_id trùng: {int(df['spotify_id_dup'].sum()):,} | "
          f"thời lượng bất thường: {int(df['duration_outlier'].sum()):,}")
    print("\n5 dòng đầu:")
    cols = ["track_id", "name", "artist", "tags", "valence", "energy"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(df[cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
