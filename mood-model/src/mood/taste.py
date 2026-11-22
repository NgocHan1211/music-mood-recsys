"""Tầng nối sang taste-model của TV2 (mô hình gu ALS).

Module này KHÔNG cài lại ALS. Nó nạp nguyên code trong `taste-model/src/` bằng
importlib (đặt tên gói là `taste_src` để không trùng với gói `src` của mood-model),
rồi huấn luyện mô hình gu trên đúng cohort mà phần đánh giá tâm trạng đang dùng.

Hai nguồn dữ liệu:
  --split official (mặc định)  bộ chia train/test chung của nhóm (TV1, xem
                               data_preprocessing_report.md). Phần bị che = tập test
                               chính thức, nên chống rò rỉ là hiển nhiên: train và
                               test không có cặp (user, bài) nào trùng nhau.
  --split local                bộ chia cũ tự gieo hạt theo user_id (che 30%), giữ lại
                               để tái lập các số đã báo cáo ở Phần II.

Trọng số tin cậy: --weight playcount (mặc định) hoặc --weight engagement_score
(cột mới của TV1, xem src/mood/splits.py).

Theo mục 3 và mục 5 của taste-model/docs/EVALUATION_PROTOCOL.md.

Chạy:  python -m src.mood.taste [--split official|local] [--weight ...] [--sample]
"""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, ROOT, SEED

TASTE_ROOT = ROOT.parent / "taste-model"
ALS_NPZ = PROCESSED_DIR / "taste_als.npz"
COHORT_CSV = PROCESSED_DIR / "taste_cohort.csv"


def artifact_paths(split: str = "official", weight: str = "playcount"):
    """Đường dẫn mô hình gu + cohort cho từng cấu hình (để các bản không ghi đè nhau)."""
    if split == "local":
        return ALS_NPZ, COHORT_CSV
    if split != "official":
        raise ValueError(f"split phải là 'official' hoặc 'local', không phải {split!r}")
    tag = "" if weight == "playcount" else "_engage"
    return (PROCESSED_DIR / f"taste_als_official{tag}.npz",
            PROCESSED_DIR / f"taste_cohort_official{tag}.csv")

# Cấu hình ALS: giữ đúng mặc định của TV2 (chưa grid-search — việc đó thuộc TV2)
ALS_CONFIG = dict(factors=32, regularization=0.1, als_alpha=10.0, iterations=15,
                  seed=SEED, weighting="log", backend="implicit")


def _load_taste_package() -> types.ModuleType:
    """Nạp taste-model/src dưới tên gói `taste_src` (giữ import tương đối bên trong)."""
    if "taste_src" not in sys.modules:
        src = TASTE_ROOT / "src"
        if not src.is_dir():
            raise FileNotFoundError(f"Không thấy {src}. Cần có folder taste-model của TV2.")
        pkg = types.ModuleType("taste_src")
        pkg.__path__ = [str(src)]
        sys.modules["taste_src"] = pkg
    return sys.modules["taste_src"]


def taste_models():
    """Trả về module `taste_src.models` của TV2 (có class ALS, Popularity)."""
    _load_taste_package()
    return importlib.import_module("taste_src.models")


def taste_hybrid():
    """Trả về module `taste_src.hybrid` của TV2 (có combine_scores)."""
    _load_taste_package()
    return importlib.import_module("taste_src.hybrid")


def build_cohort(hist: pd.DataFrame, test_users: np.ndarray,
                 n_extra_users: int, seed: int = SEED) -> np.ndarray:
    """Cohort huấn luyện gu: toàn bộ user test + `n_extra_users` user khác."""
    rng = np.random.default_rng(seed)
    others = np.setdiff1d(np.sort(hist["user_id"].unique()), test_users)
    n = min(n_extra_users, len(others))
    extra = rng.choice(others, size=n, replace=False)
    return np.concatenate([test_users, extra])


def train_frame(hist: pd.DataFrame, cohort: np.ndarray,
                hidden: dict[str, set[str]]) -> pd.DataFrame:
    """Dữ liệu huấn luyện ALS: cohort, đã bỏ các bài bị che của user test."""
    sub = hist[hist["user_id"].isin(set(cohort))][["user_id", "track_id", "playcount"]]
    if hidden:
        mask = np.fromiter(
            ((u in hidden and t in hidden[u]) for u, t in
             zip(sub["user_id"].to_numpy(), sub["track_id"].to_numpy())),
            dtype=bool, count=len(sub))
        sub = sub[~mask]
    return sub.reset_index(drop=True)


def fit_als(train: pd.DataFrame, config: dict | None = None):
    """Huấn luyện ALS bằng chính class của TV2."""
    models = taste_models()
    cfg = dict(ALS_CONFIG if config is None else config)
    model = models.ALS(**cfg)
    model.fit(train)
    return model


def load_als(split: str = "official", weight: str = "playcount"):
    """Nạp mô hình gu đã lưu cho cấu hình (split, weight) tương ứng."""
    path, _ = artifact_paths(split, weight)
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path}. Chạy: python -m src.mood.taste "
            f"--split {split} --weight {weight}")
    return taste_models().ALS.load(path)


def taste_scores(model, user_id: str, track_ids: list[str]) -> dict[str, float]:
    """Điểm gu cho đúng tập `track_ids` (giữ cả bài mô hình chưa biết -> điểm thấp nhất).

    Dùng `model.recommend()` của TV2 để không đi chệch hợp đồng; những bài ALS không
    biết (cold-item) hoặc bị loại vì đã quan sát được gán giá trị thấp nhất trong lượt
    đó, vì `combine_scores` yêu cầu hai bảng điểm phủ đúng cùng tập ứng viên.
    """
    ranked = dict(model.recommend(user_id=user_id, top_k=len(track_ids),
                                  candidate_track_ids=track_ids))
    if len(ranked) < len(track_ids):
        floor = min(ranked.values()) if ranked else 0.0
        for t in track_ids:
            ranked.setdefault(t, floor)
    return {t: float(ranked[t]) for t in track_ids}


def main() -> None:
    from . import api
    from .config import load_config
    from .evaluate_mood import hidden_for_user, split_users
    from .io import load_history
    from .splits import als_frame, hidden_map, official_split

    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình gu (ALS của TV2) cho đánh giá ghép.")
    parser.add_argument("--split", choices=["official", "local"], default="official",
                        help="official = bộ chia chung của nhóm; local = bộ chia cũ che 30%%")
    parser.add_argument("--weight", choices=["playcount", "engagement_score"],
                        default="playcount", help="cột trọng số tin cậy cho ALS")
    parser.add_argument("--sample", action="store_true", help="cohort nhỏ để thử pipeline")
    parser.add_argument("--n-users", type=int, default=0,
                        help="split official: giới hạn số user huấn luyện (0 = dùng hết)")
    parser.add_argument("--n-test-users", type=int, default=20000,
                        help="split local: số user test")
    parser.add_argument("--n-extra-users", type=int, default=30000,
                        help="split local: số user thêm vào cohort")
    args = parser.parse_args()

    cfg = load_config()
    ev = cfg["evaluation"]
    profiles = api.load_mood_profiles()
    als_npz, cohort_csv = artifact_paths(args.split, args.weight)

    if args.split == "official":
        train, test = official_split(set(profiles.index))
        users = np.sort(train["user_id"].unique())
        n_cap = 3000 if args.sample else args.n_users
        if n_cap and n_cap < len(users):
            users = np.sort(np.random.default_rng(SEED).choice(users, size=n_cap,
                                                               replace=False))
            train = train[train["user_id"].isin(set(users))]
            test = test[test["user_id"].isin(set(users))]
        hidden = hidden_map(test, users)
        frame = als_frame(train, args.weight)
        cohort, test_users = users, users
        print(f"Bộ chia chính thức của nhóm | trọng số: {args.weight}")
        print(f"Huấn luyện trên {len(frame):,} tương tác train của {len(users):,} user")
        print(f"Phần bị che (tập test chính thức): "
              f"{sum(len(v) for v in hidden.values()):,} tương tác")
    else:
        hist = load_history(min_user_tracks=int(ev["min_user_tracks"]))
        hist = hist[hist["track_id"].isin(set(profiles.index))]
        n_test = 200 if args.sample else args.n_test_users
        n_extra = 2000 if args.sample else args.n_extra_users

        # Dùng ĐÚNG chuỗi ngẫu nhiên của evaluate_mood.py để cùng một tập user test
        rng = np.random.default_rng(SEED)
        test_users = split_users(hist, float(ev["test_user_ratio"]))
        if n_test < len(test_users):
            test_users = rng.choice(test_users, size=n_test, replace=False)

        hidden = {}
        sub = hist[hist["user_id"].isin(set(test_users))]
        for user_id, g in sub.groupby("user_id"):
            tracks = g["track_id"].to_numpy()
            if len(tracks) < int(ev["min_user_tracks"]):
                continue
            _, hid = hidden_for_user(user_id, tracks, float(ev["hidden_ratio"]))
            hidden[user_id] = set(hid)

        cohort = build_cohort(hist, test_users, n_extra)
        frame = train_frame(hist, cohort, hidden)
        print(f"Cohort gu: {len(cohort):,} user "
              f"({len(test_users):,} test + {len(cohort) - len(test_users):,} khác)")
        print(f"Tương tác huấn luyện: {len(frame):,} dòng | "
              f"{frame['track_id'].nunique():,} bài")
        print(f"Đã loại {sum(len(v) for v in hidden.values()):,} "
              "tương tác bị che của user test")

    # Rò rỉ: không cặp (user, bài) nào vừa ở dữ liệu huấn luyện vừa ở phần bị che
    leaked = sum(1 for u, t in zip(frame["user_id"].to_numpy(), frame["track_id"].to_numpy())
                 if u in hidden and t in hidden[u])
    print(f"Rò rỉ (cặp bị che có trong dữ liệu huấn luyện): {leaked:,}")
    assert leaked == 0, "phần bị che đã lọt vào dữ liệu huấn luyện ALS"

    print("Đang huấn luyện ALS...", flush=True)
    model = fit_als(frame)
    model.save(als_npz)
    print(f"ALS: {len(model.users):,} user x {len(model.items):,} bài, "
          f"{model.config['factors']} factor -> {als_npz}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # np.isin trên mảng chuỗi lớn rất chậm -> dùng set
    test_set = set(test_users)
    pd.DataFrame({"user_id": cohort,
                  "is_test": [u in test_set for u in cohort]}).to_csv(cohort_csv, index=False)
    print("Cohort:", cohort_csv)

    # Kiểm tra nhanh: mô hình có xếp bài bị che cao hơn bài chưa nghe không?
    demo = next(iter(hidden))
    ids = list(hidden[demo])[:5]
    sc = taste_scores(model, demo, ids)
    print(f"\nVí dụ user {demo}: điểm gu 5 bài bị che = "
          + ", ".join(f"{v:.3f}" for v in sc.values()))


if __name__ == "__main__":
    main()
