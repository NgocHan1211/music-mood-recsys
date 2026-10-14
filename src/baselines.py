"""Baseline để so sánh với ALS: Popularity và Random. Xem docs/evaluation_protocol.md mục 6."""
import numpy as np
import pandas as pd


def popularity_recs(train: pd.DataFrame, users, n: int = 20) -> dict[str, list[str]]:
    """Gợi ý các bài được nhiều user nghe nhất, bỏ bài user đã nghe trong train."""
    ranked = train.groupby("track_id")["user_id"].nunique().sort_values(ascending=False).index.tolist()
    seen = train.groupby("user_id")["track_id"].apply(set).to_dict()
    out = {}
    for u in users:
        s = seen.get(u, set())
        out[u] = [t for t in ranked if t not in s][:n]
    return out


def random_recs(train: pd.DataFrame, users, n: int = 20, seed: int = 42) -> dict[str, list[str]]:
    """Gợi ý ngẫu nhiên, bỏ bài user đã nghe trong train."""
    rng = np.random.default_rng(seed)
    items = train["track_id"].unique()
    seen = train.groupby("user_id")["track_id"].apply(set).to_dict()
    out = {}
    for u in users:
        s = seen.get(u, set())
        cand = items[rng.permutation(len(items))]
        out[u] = [t for t in cand if t not in s][:n]
    return out
