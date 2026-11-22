"""Đánh giá phần tâm trạng trên lịch sử nghe thật (task T6 / M6).

So sánh 3 cách đặt mục tiêu tâm trạng:
  manual   - tọa độ tự đặt tay (bản baseline)
  learned  - tâm học được từ dữ liệu (M4)
  personal - tâm cá nhân hóa theo lịch sử nghe của từng user (M6)

Giao thức: chia 20% user làm test (seed 42), mỗi user test che 30% số bài.
Với mỗi (user test, nhóm m có >= min_hidden_per_mood bài bị che thuộc m):
xếp hạng các bài ứng viên theo `mood_score` và đo Recall@k, NDCG@k trên các bài
bị che thuộc nhóm m.

Xuất: data/processed/mood_eval.csv, reports/figures/mood_eval.png
"""

from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd

from .config import FIGURES_DIR, GROUPS, PROCESSED_DIR, SEED, load_config
from .io import load_history
from . import api

TARGETS = ["manual", "learned", "personal", "seed3"]
N_SEEDS = 3   # A3: số bài "ví dụ" đưa vào, bằng số bài bài NLP4MusA 2026 dùng
EVAL_CSV = PROCESSED_DIR / "mood_eval.csv"


def split_users(hist: pd.DataFrame, test_ratio: float, seed: int = SEED) -> np.ndarray:
    """Chọn ngẫu nhiên `test_ratio` số user làm test."""
    users = np.sort(hist["user_id"].unique())
    rng = np.random.default_rng(seed)
    return rng.choice(users, size=int(len(users) * test_ratio), replace=False)


def hide_tracks(track_ids: np.ndarray, hidden_ratio: float,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Chia bài của một user thành phần thấy được và phần bị che."""
    order = rng.permutation(len(track_ids))
    n_hidden = max(1, int(round(len(track_ids) * hidden_ratio)))
    hidden = track_ids[order[:n_hidden]]
    visible = track_ids[order[n_hidden:]]
    return visible, hidden


def hidden_for_user(user_id: str, track_ids: np.ndarray,
                    hidden_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Chia thấy được / bị che cho MỘT user, không phụ thuộc thứ tự lặp.

    `hide_tracks` dùng chung một bộ sinh ngẫu nhiên nên kết quả đổi khi thứ tự duyệt
    user đổi. Hàm này gieo hạt từ chính `user_id`, nên bất kỳ module nào (đánh giá
    bốn tập, huấn luyện ALS) cũng tái lập đúng một bộ chia — điều kiện bắt buộc để
    phần bị che bị loại khỏi dữ liệu huấn luyện mô hình gu.
    """
    seed = int(hashlib.blake2b(user_id.encode(), digest_size=8).hexdigest(), 16) % (2 ** 32)
    rng = np.random.default_rng(seed + SEED)
    return hide_tracks(np.sort(track_ids), hidden_ratio, rng)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return np.nan
    hit = len(set(ranked[:k]) & relevant)
    return hit / min(len(relevant), k)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return np.nan
    gains = [1.0 if t in relevant else 0.0 for t in ranked[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else np.nan


def seed_target(user_id: str, mood: str, visible_ids: np.ndarray, weights: np.ndarray,
                cfg: dict, n_seeds: int | None = None) -> tuple[tuple[float, float], int]:
    """A3 — mục tiêu tâm trạng suy từ ĐÚNG `n_seeds` bài ví dụ ("nói ngầm").

    Khác `personal` ở hai điểm: chỉ dùng n_seeds bài thay vì toàn bộ lịch sử, và
    KHÔNG co về tâm chung — tức là không dùng tới tọa độ của nhóm tâm trạng.
    Đây là bản đối chiếu với phát hiện "implicit thắng explicit" của Hausberger,
    Jósár & Schedl (NLP4MusA 2026).

    Trả về ((valence, energy), số bài hợp lệ). Không đủ n_seeds bài -> dùng tâm chung,
    giống đường lui của `user_mood_target`.
    """
    n_seeds = N_SEEDS if n_seeds is None else int(n_seeds)
    index, proba, ve = api._profile_arrays()
    min_prob = float(cfg["personalization"]["min_prob"])
    g = GROUPS.index(mood)

    rows = np.fromiter((index.get(t, -1) for t in visible_ids), dtype=np.int64,
                       count=len(visible_ids))
    known = rows >= 0
    idx = rows[known]
    w = np.asarray(weights, dtype=float)[known]
    belongs = proba[idx, g] >= min_prob
    idx, w = idx[belongs], w[belongs]
    n_avail = int(len(idx))

    if n_avail < n_seeds:
        return api.quick_mood(mood), n_avail

    # Chọn n_seeds bài tái lập được, không ăn vào bộ sinh ngẫu nhiên chung của đánh giá
    seed = int(hashlib.blake2b(f"{user_id}|{mood}".encode(), digest_size=8).hexdigest(), 16)
    pick = np.random.default_rng(seed % (2 ** 32) + SEED).choice(n_avail, size=n_seeds,
                                                                replace=False)
    idx, w = idx[pick], w[pick]
    if w.sum() <= 0:
        w = np.ones(len(idx))
    mu = (w[:, None] * ve[idx]).sum(axis=0) / w.sum()
    return (float(mu[0]), float(mu[1])), n_avail


def target_point(kind: str, mood: str, visible_ids: np.ndarray,
                 weights: np.ndarray, cfg: dict,
                 user_id: str | None = None) -> tuple[float, float]:
    """Điểm mục tiêu trên thanh kéo theo từng cách đặt."""
    if kind == "manual":
        v, e = cfg["manual_centroids"][mood]
        return float(v), float(e)
    if kind == "learned":
        return api.quick_mood(mood)
    if kind == "personal":
        return api.user_mood_target(list(visible_ids), mood, weights=weights)
    if kind == "seed3":
        point, _ = seed_target(user_id, mood, visible_ids, weights, cfg)
        return point
    raise ValueError(kind)


def _case_rows(user_id: str, visible: np.ndarray, vis_w: np.ndarray,
               hidden: np.ndarray, listened: set[str], cfg: dict,
               universe: np.ndarray, mood_pred: pd.Series,
               rng: np.random.Generator, n_candidates: int, ks: list[int]) -> list[dict]:
    """Các dòng kết quả của MỘT user: mỗi (nhóm đủ bài bị che) x (cách đặt mục tiêu)."""
    ev = cfg["evaluation"]
    hidden_moods = mood_pred.reindex(hidden).to_numpy()
    rows = []
    for mood in GROUPS:
        rel = {t for t, m in zip(hidden, hidden_moods) if m == mood}
        if len(rel) < int(ev["min_hidden_per_mood"]):
            continue

        # Ứng viên: bài bị che + n_candidates bài user chưa nghe (lấy ngẫu nhiên)
        pool = rng.choice(universe, size=n_candidates * 2, replace=False)
        pool = [t for t in pool if t not in listened][:n_candidates]
        candidates = list(dict.fromkeys(list(hidden) + pool))

        _, n_avail = seed_target(user_id, mood, visible, vis_w, cfg)
        for kind in TARGETS:
            v, e = target_point(kind, mood, visible, vis_w, cfg, user_id=user_id)
            scores = api.mood_scores(candidates, v, e)
            order = np.argsort(-scores)
            ranked = [candidates[i] for i in order]
            row = {"user_id": user_id, "mood": mood, "target": kind,
                   "n_relevant": len(rel), "n_candidates": len(candidates),
                   "n_mood_tracks_visible": n_avail}
            for k in ks:
                row[f"recall@{k}"] = recall_at_k(ranked, rel, k)
                row[f"ndcg@{k}"] = ndcg_at_k(ranked, rel, k)
            rows.append(row)
    return rows


def evaluate(hist: pd.DataFrame | None, profiles: pd.DataFrame, cfg: dict,
             n_users: int | None = None, n_candidates: int = 1000,
             views: dict | None = None,
             universe: np.ndarray | None = None) -> pd.DataFrame:
    """Chạy đánh giá, trả về một dòng cho mỗi (user, nhóm, cách đặt mục tiêu).

    `views = None`: bộ chia cũ (che 30% theo chuỗi ngẫu nhiên chung) -> tái lập T6.
    `views` khác None: bộ chia chính thức của nhóm (xem src/mood/splits.user_views).
    """
    ev = cfg["evaluation"]
    ks = list(ev["k"])
    rng = np.random.default_rng(SEED)
    mood_pred = profiles["mood_pred"]
    universe = profiles.index.to_numpy() if universe is None else universe
    rows: list[dict] = []

    if views is None:
        test_users = split_users(hist, float(ev["test_user_ratio"]))
        if n_users is not None and n_users < len(test_users):
            test_users = rng.choice(test_users, size=n_users, replace=False)
        sub = hist[hist["user_id"].isin(set(test_users))]
        for user_id, g in sub.groupby("user_id"):
            tracks = g["track_id"].to_numpy()
            plays = g["playcount"].to_numpy(dtype=float)
            if len(tracks) < int(ev["min_user_tracks"]):
                continue
            visible, hidden = hide_tracks(tracks, float(ev["hidden_ratio"]), rng)
            if len(visible) == 0:
                continue
            vis_w = plays[np.isin(tracks, visible)]
            rows += _case_rows(user_id, visible, vis_w, hidden, set(tracks), cfg,
                               universe, mood_pred, rng, n_candidates, ks)
    else:
        for user_id in sorted(views):
            visible, vis_w, hidden = views[user_id]
            if len(visible) < int(ev["min_user_tracks"]) or len(hidden) == 0:
                continue
            listened = set(visible) | set(hidden)
            rows += _case_rows(user_id, visible, vis_w, hidden, listened, cfg,
                               universe, mood_pred, rng, n_candidates, ks)

    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Trung bình các chỉ số theo cách đặt mục tiêu."""
    ks = list(cfg["evaluation"]["k"])
    cols = [f"{m}@{k}" for k in ks for m in ("recall", "ndcg")]
    out = res.groupby("target")[cols].mean().reindex(TARGETS)
    out["n_cases"] = res.groupby("target").size().reindex(TARGETS)
    return out


def plot_results(res: pd.DataFrame, cfg: dict, tag: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = max(cfg["evaluation"]["k"])
    by_mood = res.groupby(["mood", "target"])[f"recall@{k}"].mean().unstack()[TARGETS]
    by_mood = by_mood.reindex(GROUPS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    overall = res.groupby("target")[[f"recall@{k}", f"ndcg@{k}"]].mean().reindex(TARGETS)
    x = np.arange(len(TARGETS)); w = 0.35
    axes[0].bar(x - w/2, overall[f"recall@{k}"], w, label=f"Recall@{k}", color="#4C72B0")
    axes[0].bar(x + w/2, overall[f"ndcg@{k}"], w, label=f"NDCG@{k}", color="#DD8452")
    axes[0].set_xticks(x, ["tọa độ\ntự đặt", "tâm\nhọc được", "tâm\ncá nhân", "3 bài\nví dụ"])
    axes[0].set_title(f"Ba cách đặt mục tiêu tâm trạng (@{k})")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    by_mood.plot(kind="bar", ax=axes[1], color=["#8C8C8C", "#4C72B0", "#55A868", "#C44E52"])
    axes[1].set_title(f"Recall@{k} theo từng nhóm tâm trạng")
    axes[1].set_xlabel(""); axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(["tọa độ tự đặt", "tâm học được", "tâm cá nhân", "3 bài ví dụ"])
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"mood_eval{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return str(path)


def main() -> None:
    global N_SEEDS
    parser = argparse.ArgumentParser(description="Đánh giá mục tiêu tâm trạng trên lịch sử nghe thật.")
    parser.add_argument("--sample", action="store_true", help="chỉ 200 user test")
    parser.add_argument("--n-users", type=int, default=2000, help="số user test dùng để đánh giá")
    parser.add_argument("--n-candidates", type=int, default=1000, help="số bài ứng viên mỗi lượt")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="A3: số bài ví dụ cho mục tiêu seed3 (bài báo dùng 3)")
    parser.add_argument("--split", choices=["official", "local"], default="official",
                        help="official = bộ chia chung của nhóm; local = bộ chia cũ che 30%%")
    parser.add_argument("--weight", choices=["playcount", "engagement_score"],
                        default="playcount", help="trọng số khi tính tâm cá nhân")
    parser.add_argument("--universe", choices=["official", "all"], default="official",
                        help="kho nhạc ứng viên: catalogue 10-core hay toàn bộ profiles")
    parser.add_argument("--out-tag", default="",
                        help="hậu tố riêng cho file kết quả (để không ghi đè bản đã báo cáo)")
    args = parser.parse_args()

    N_SEEDS = int(args.n_seeds)

    cfg = load_config()
    profiles = api.load_mood_profiles()
    n_users = 200 if args.sample else args.n_users

    if args.split == "official":
        from .splits import official_split, user_views
        train, test = official_split(set(profiles.index))
        users = np.sort(train["user_id"].unique())
        if n_users < len(users):
            users = np.sort(np.random.default_rng(SEED).choice(users, size=n_users,
                                                               replace=False))
        views = user_views(train, test, users, weight_col=args.weight)
        # Kho nhạc = catalogue chính thức sau lọc 10-core; "all" để đối chứng với bản cũ
        universe = (np.sort(train["track_id"].unique()) if args.universe == "official"
                    else profiles.index.to_numpy())
        print(f"Bộ chia chính thức của nhóm | trọng số tâm cá nhân: {args.weight}")
        print(f"{len(views):,} user được đánh giá | kho nhạc {len(universe):,} bài | "
              f"thấy được trung vị {int(np.median([len(v[0]) for v in views.values()]))} bài, "
              f"bị che trung vị {int(np.median([len(v[2]) for v in views.values()]))} bài")
        res = evaluate(None, profiles, cfg, n_candidates=args.n_candidates,
                       views=views, universe=universe)
    else:
        hist = load_history(min_user_tracks=int(cfg["evaluation"]["min_user_tracks"]))
        hist = hist[hist["track_id"].isin(set(profiles.index))]
        print(f"Lịch sử nghe sau lọc 10/10: {len(hist):,} dòng | "
              f"{hist['user_id'].nunique():,} user | {hist['track_id'].nunique():,} bài")
        res = evaluate(hist, profiles, cfg, n_users=n_users,
                       n_candidates=args.n_candidates)
    if res.empty:
        print("Không có trường hợp nào đủ điều kiện đánh giá.")
        return

    n_cases = res.groupby(["user_id", "mood"]).ngroups
    print(f"Số user test dùng: {res['user_id'].nunique():,} | "
          f"số cặp (user, nhóm) đủ điều kiện: {n_cases:,}")
    print(f"Ứng viên mỗi lượt: ~{int(res['n_candidates'].mean()):,} bài "
          f"(bài bị che + {args.n_candidates} bài chưa nghe)\n")

    summary = summarize(res, cfg)
    print(f"-- So sánh {len(TARGETS)} cách đặt mục tiêu (seed3 dùng {N_SEEDS} bài ví dụ) --")
    print(summary.to_string(float_format="{:.4f}".format))

    k = max(cfg["evaluation"]["k"])
    base = summary.loc["manual", f"recall@{k}"]
    for t in ["learned", "personal", "seed3"]:
        d = summary.loc[t, f"recall@{k}"]
        print(f"  {t:<9} Recall@{k} so với tọa độ tự đặt: {d - base:+.4f} "
              f"({(d/base - 1)*100:+.1f}%)")

    print("\n-- Recall@%d theo nhóm --" % k)
    print(res.groupby(["mood", "target"])[f"recall@{k}"].mean().unstack()[TARGETS]
          .reindex(GROUPS).to_string(float_format="{:.4f}".format))

    # A3: "nói ngầm" (3 bài ví dụ) so với "nói tỏ" theo số bài user có trong nhóm đó
    print(f"\n-- Recall@{k} theo số bài user đã nghe thuộc nhóm (phần thấy được) --")
    bins = [0, 2, 4, 9, 19, 10 ** 6]
    names = ["0-2 bài (seed3 phải lui về tâm chung)", "3-4 bài", "5-9 bài",
             "10-19 bài", "20+ bài"]
    res = res.copy()
    res["nhom_so_bai"] = pd.cut(res["n_mood_tracks_visible"], bins=bins, labels=names,
                                right=True, include_lowest=True)
    strat = (res.groupby(["nhom_so_bai", "target"], observed=True)[f"recall@{k}"]
             .mean().unstack()[TARGETS])
    strat["n_cặp"] = (res[res["target"] == "manual"]
                      .groupby("nhom_so_bai", observed=True).size())
    print(strat.to_string(float_format="{:.4f}".format))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_official" if args.split == "official" else ""
    if args.weight != "playcount":
        tag += "_engage"
    tag += args.out_tag
    suffix = tag + ("" if N_SEEDS == 3 else f"_seeds{N_SEEDS}")
    eval_csv = PROCESSED_DIR / f"mood_eval{tag}.csv"
    res.to_csv(eval_csv, index=False)
    summary.to_csv(PROCESSED_DIR / f"mood_eval_summary{suffix}.csv")
    print(f"\nĐã lưu: {eval_csv}")
    print("Hình:", plot_results(res, cfg, tag=tag))


if __name__ == "__main__":
    main()
