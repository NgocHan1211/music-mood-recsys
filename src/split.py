"""Chia train/test theo giao thức. Người phụ trách: Hân. Xem docs/evaluation_protocol.md."""
import numpy as np
import pandas as pd


def split_interactions(
    interactions: pd.DataFrame,
    strategy: str = "random_holdout",
    test_ratio: float = 0.2,
    min_user_interactions: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả về (train, test). Mọi user trong test đều có trong train.

    strategy:
      - "random_holdout": mỗi user giữ ngẫu nhiên `test_ratio` tương tác (tối thiểu 1).
      - "leave_one_out": mỗi user giữ 1 tương tác (cuối nếu có timestamp, ngẫu nhiên nếu không).
      - "temporal": mỗi user giữ `test_ratio` tương tác gần nhất (cần cột timestamp).
    """
    if strategy not in {"random_holdout", "leave_one_out", "temporal"}:
        raise ValueError(f"strategy không hợp lệ: {strategy}")
    has_ts = "timestamp" in interactions.columns and interactions["timestamp"].notna().any()
    if strategy == "temporal" and not has_ts:
        raise ValueError("strategy='temporal' cần cột timestamp có dữ liệu")

    df = interactions.copy()
    counts = df.groupby("user_id")["track_id"].transform("size")
    df = df[counts >= min_user_interactions]

    rng = np.random.default_rng(seed)
    # Thứ tự ưu tiên vào test: thời gian (mới nhất trước) hoặc ngẫu nhiên
    if strategy == "temporal" or (strategy == "leave_one_out" and has_ts):
        df = df.sort_values(["user_id", "timestamp"], ascending=[True, False])
        rank = df.groupby("user_id").cumcount()
    else:
        df = df.assign(_r=rng.random(len(df))).sort_values(["user_id", "_r"])
        rank = df.groupby("user_id").cumcount()
        df = df.drop(columns="_r")

    n_user = df.groupby("user_id")["track_id"].transform("size")
    if strategy == "leave_one_out":
        n_test = 1
    else:
        n_test = np.maximum(1, np.floor(n_user * test_ratio)).astype(int)
    is_test = rank < n_test

    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
