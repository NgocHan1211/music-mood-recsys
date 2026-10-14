"""Gán nhãn mood từ tag Last.fm. Người phụ trách: Như. Xem CONTRACTS.md mục 1.3, 1.4."""
import json
import pandas as pd


def load_mood_mapping(path: str = "configs/mood_mapping.json") -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["moods"]


def label_tracks(tracks: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.DataFrame:
    raise NotImplementedError("Như triển khai")
