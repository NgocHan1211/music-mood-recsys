"""Đọc và ghép dữ liệu. Người phụ trách: Thy. Xem CONTRACTS.md mục 2."""
import pandas as pd


def load_tracks(path: str = "data/processed/tracks.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def load_interactions(path: str = "data/processed/sample_interactions.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def build_tracks(music_info_path: str, spotify_path: str) -> pd.DataFrame:
    raise NotImplementedError("Thy triển khai")


def match_stats(tracks: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Thy triển khai")
