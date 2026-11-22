"""Bộ chia train/test chính thức của nhóm (TV1 xuất, xem data_preprocessing_report.md).

Trước đây phần đánh giá tâm trạng tự chia thấy được / bị che bằng
`evaluate_mood.hidden_for_user` (gieo hạt theo user_id, che 30%). Nay nhóm đã có
bộ chia chung: lọc 10-core lặp rồi giữ ~20% tương tác mỗi user làm test. Module này
đọc đúng bộ chia đó để A2 và A3 chạy trên cùng dữ liệu với TV1 và TV2.

Khác biệt cần nhớ khi đọc kết quả:
  - phần bị che nhỏ hơn (trung vị 4 bài/user, trước là ~30% của ~24 bài),
  - kho nhạc hẹp hơn (22.557 bài sau 10-core lặp, trước 24.286),
  - có thêm cột `engagement_score` dùng được làm trọng số tin cậy thay `playcount`.

File gốc nằm ngoài mood-model (project/data/) nên chỉ đọc, không ghi.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, ROOT

SPLIT_DIR = ROOT.parent / "data"
CSV = {"train": SPLIT_DIR / "train.csv", "test": SPLIT_DIR / "test.csv"}
# Bản cache cột cần dùng: đọc CSV 930 MB mỗi lần quá chậm
CACHE = {k: PROCESSED_DIR / f"split_{k}.parquet" for k in CSV}

USECOLS = ["user_id", "track_id", "playcount", "engagement_score"]
WEIGHT_COLS = ["playcount", "engagement_score"]
# engagement_score nằm trong [0, 1]; đưa về cùng thang playcount đã chặn (60) để
# dùng lại đúng _confidence của TV2 (1 + alpha * log1p(count)) mà không sửa code TV2
ENGAGEMENT_SCALE = 60.0


def load_part(which: str, refresh: bool = False) -> pd.DataFrame:
    """Đọc một nửa bộ chia (`train` hoặc `test`), có cache parquet."""
    if which not in CSV:
        raise ValueError(f"which phải là 'train' hoặc 'test', không phải {which!r}")
    cache = CACHE[which]
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    path = CSV[which]
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy {path}. Cần folder data/ của nhóm ở cạnh mood-model "
            "(xem data_preprocessing_report.md).")
    df = pd.read_csv(path, usecols=USECOLS)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def official_split(track_filter: set[str] | None = None,
                   refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả về (train, test) của bộ chia chính thức, lọc theo kho bài nếu cần."""
    train = load_part("train", refresh)
    test = load_part("test", refresh)
    if track_filter is not None:
        train = train[train["track_id"].isin(track_filter)]
        test = test[test["track_id"].isin(track_filter)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def hidden_map(test: pd.DataFrame, users: np.ndarray | None = None) -> dict[str, set[str]]:
    """Phần bị che của từng user = đúng các bài trong tập test chính thức."""
    sub = test if users is None else test[test["user_id"].isin(set(users))]
    return {u: set(g) for u, g in sub.groupby("user_id")["track_id"]}


def user_views(train: pd.DataFrame, test: pd.DataFrame, users: np.ndarray,
               weight_col: str = "playcount") -> dict:
    """Với mỗi user: (bài thấy được, trọng số, bài bị che) theo bộ chia chính thức.

    Thấy được = tương tác train, bị che = tương tác test. Trọng số lấy từ `weight_col`
    để `api.user_mood_target` cân theo mức nghe (hoặc mức gắn bó).
    """
    keep = set(users)
    tr = train[train["user_id"].isin(keep)]
    te = test[test["user_id"].isin(keep)]
    hidden = {u: g.to_numpy() for u, g in te.groupby("user_id")["track_id"]}
    views = {}
    for user_id, g in tr.groupby("user_id"):
        if user_id not in hidden:
            continue
        views[user_id] = (g["track_id"].to_numpy(),
                          g[weight_col].to_numpy(dtype=float),
                          hidden[user_id])
    return views


def als_frame(frame: pd.DataFrame, weight_col: str = "playcount") -> pd.DataFrame:
    """Dữ liệu cho ALS của TV2: cột trọng số luôn được đặt tên `playcount`.

    weight_col = "engagement_score" -> nhân ENGAGEMENT_SCALE để cùng thang với
    playcount đã chặn, vì ALS của TV2 tính tin cậy bằng 1 + alpha * log1p(count).
    """
    if weight_col not in WEIGHT_COLS:
        raise ValueError(f"weight_col phải thuộc {WEIGHT_COLS}")
    out = frame[["user_id", "track_id"]].copy()
    if weight_col == "playcount":
        out["playcount"] = frame["playcount"].to_numpy(dtype=float)
    else:
        out["playcount"] = frame["engagement_score"].to_numpy(dtype=float) * ENGAGEMENT_SCALE
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra bộ chia chính thức của nhóm.")
    parser.add_argument("--refresh", action="store_true", help="đọc lại CSV, bỏ cache")
    args = parser.parse_args()

    from . import api
    profiles = api.load_mood_profiles()
    train, test = official_split(set(profiles.index), refresh=args.refresh)

    for name, d in (("train", train), ("test", test)):
        per_user = d.groupby("user_id").size()
        print(f"{name:<6} {len(d):>10,} dòng | {d['user_id'].nunique():>7,} user | "
              f"{d['track_id'].nunique():>6,} bài | trung vị {per_user.median():.0f} bài/user")

    pairs_tr = set(map(tuple, train[["user_id", "track_id"]].to_numpy()))
    pairs_te = set(map(tuple, test[["user_id", "track_id"]].to_numpy()))
    print(f"\nCặp (user, bài) xuất hiện ở cả train và test: {len(pairs_tr & pairs_te):,}")
    print(f"Bài trong test nhưng không có trong train: "
          f"{len(set(test['track_id']) - set(train['track_id'])):,}")


if __name__ == "__main__":
    main()
