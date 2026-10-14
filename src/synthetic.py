"""Dữ liệu giả có cấu trúc (long-tail + sở thích theo nhóm) để chạy thử pipeline khi chưa có dữ liệu thật."""
import numpy as np
import pandas as pd


def make_synthetic(n_users=1500, n_items=800, n_groups=8, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    item_group = rng.integers(0, n_groups, n_items)
    pop = rng.pareto(1.2, n_items) + 1
    rows = []
    for u in range(n_users):
        g = rng.integers(0, n_groups)
        w = pop * np.where(item_group == g, 8.0, 1.0)
        w /= w.sum()
        k = int(np.clip(rng.lognormal(3.0, 0.6), 6, 150))
        items = rng.choice(n_items, size=min(k, n_items), replace=False, p=w)
        for i in items:
            rows.append((f"u{u}", f"t{i}", int(rng.geometric(0.3))))
    return pd.DataFrame(rows, columns=["user_id", "track_id", "playcount"])
