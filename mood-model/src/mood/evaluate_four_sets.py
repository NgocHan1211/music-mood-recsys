"""A2 — Đánh giá bốn tập: mô hình tâm trạng có thêm gì so với mô hình gu?

Khung lấy từ Hausberger, Jósár & Schedl (NLP4MusA 2026), "Read Between the Tracks":
với mỗi cặp (người dùng u, nhóm tâm trạng m), chia kho nhạc thành bốn tập không
giao nhau, lấy N bài mỗi tập, trộn lẫn, rồi đo NDCG@N riêng cho từng tập.

    set1  U ∩ Mood    bài u đã nghe (phần bị che) VÀ thuộc nhóm m
    set2  U \\ Mood    bài u đã nghe (phần bị che) nhưng thuộc nhóm khác
    set3  Mood \\ U    bài thuộc nhóm m mà u chưa từng nghe
    set4  không gì     bài nhóm khác mà u chưa từng nghe

Cách đọc: bộ xếp hạng chỉ biết gu sẽ nâng cả set1 và set2 (đều là bài u thích) mà
không phân biệt tâm trạng; bộ chỉ biết tâm trạng nâng set1 và set3; bộ ghép tốt phải
dồn set1 lên trên cùng. Chênh lệch NDCG@N của set1 giữa `taste` và `hybrid` chính là
phần giá trị mà mô hình tâm trạng thêm vào.

Chống rò rỉ: bộ chia thấy được/bị che gieo hạt theo user_id (evaluate_mood.hidden_for_user)
và ALS đã được huấn luyện sau khi loại hết phần bị che (src/mood/taste.py). Mục tiêu
tâm trạng cá nhân hóa chỉ dùng phần thấy được, theo mục 5 của
taste-model/docs/EVALUATION_PROTOCOL.md.

Chạy:  python -m src.mood.evaluate_four_sets [--n-per-set 5] [--sample]
"""

from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd

from . import api
from .config import FIGURES_DIR, GROUPS, PROCESSED_DIR, SEED, load_config
from .evaluate_mood import hidden_for_user, ndcg_at_k
from .io import load_history
from .taste import COHORT_CSV, load_als, taste_hybrid, taste_models, taste_scores

SETS = ["set1_gu_va_tamtrang", "set2_chi_gu", "set3_chi_tamtrang", "set4_khong_gi"]
RANKERS = ["random", "pop", "taste", "mood", "hybrid30", "hybrid50", "hybrid70"]
HYBRID_WEIGHTS = {"hybrid30": 0.3, "hybrid50": 0.5, "hybrid70": 0.7}


def _case_rng(user_id: str, mood: str) -> np.random.Generator:
    """Bộ sinh ngẫu nhiên riêng cho mỗi cặp (user, nhóm) -> tái lập được."""
    key = f"{user_id}|{mood}".encode()
    seed = int(hashlib.blake2b(key, digest_size=8).hexdigest(), 16) % (2 ** 32)
    return np.random.default_rng(seed + SEED)


def fit_popularity(model):
    """Popularity của TV2, khớp đúng dữ liệu huấn luyện ALS (lấy lại từ model.seen).

    Popularity chỉ dùng số người nghe khác nhau cho mỗi bài nên playcount = 1 không
    ảnh hưởng điểm; cách này bảo đảm hai mô hình nền dùng chung một tập huấn luyện.
    """
    rows = [(u, t, 1) for u, tracks in model.seen.items() for t in tracks]
    frame = pd.DataFrame(rows, columns=["user_id", "track_id", "playcount"])
    return taste_models().Popularity().fit(frame)


def build_candidates(hidden: np.ndarray, hidden_moods: np.ndarray, listened: set[str],
                     mood: str, mood_pred: pd.Series, universe_by_mood: dict[str, np.ndarray],
                     n_per_set: int, rng: np.random.Generator) -> dict[str, list[str]] | None:
    """Bốn tập, mỗi tập `n_per_set` bài. Trả None nếu không đủ bài."""
    s1 = hidden[hidden_moods == mood]
    s2 = hidden[(hidden_moods != mood) & (hidden_moods != None)]  # noqa: E711
    if len(s1) < n_per_set or len(s2) < n_per_set:
        return None

    out = {SETS[0]: list(rng.choice(s1, size=n_per_set, replace=False)),
           SETS[1]: list(rng.choice(s2, size=n_per_set, replace=False))}

    # set3 / set4: lấy từ kho nhạc, bỏ mọi bài user từng nghe
    for key, pool_mood in ((SETS[2], mood), (SETS[3], None)):
        pool = universe_by_mood[pool_mood] if pool_mood else universe_by_mood["_other_" + mood]
        pick, guard = [], 0
        while len(pick) < n_per_set and guard < 20:
            draw = rng.choice(pool, size=n_per_set * 4, replace=False)
            pick += [t for t in draw if t not in listened and t not in pick]
            guard += 1
        if len(pick) < n_per_set:
            return None
        out[key] = pick[:n_per_set]
    return out


def rank_all(candidates: list[str], user_id: str, mood: str, visible: np.ndarray,
             vis_w: np.ndarray, als, pop, rng: np.random.Generator) -> dict[str, list[str]]:
    """Thứ tự xếp hạng của từng bộ xếp hạng trên cùng tập ứng viên."""
    combine = taste_hybrid().combine_scores
    ranked: dict[str, list[str]] = {}

    ranked["random"] = list(rng.permutation(candidates))

    pop_scores = dict(pop.recommend(user_id=user_id, top_k=len(candidates),
                                    candidate_track_ids=candidates))
    floor = min(pop_scores.values()) if pop_scores else 0.0
    pop_scores = {t: pop_scores.get(t, floor) for t in candidates}
    ranked["pop"] = [t for t, _ in sorted(pop_scores.items(), key=lambda p: (-p[1], p[0]))]

    t_scores = taste_scores(als, user_id, candidates)
    ranked["taste"] = [t for t, _ in sorted(t_scores.items(), key=lambda p: (-p[1], p[0]))]

    v, e = api.user_mood_target(list(visible), mood, weights=vis_w)
    m_vec = api.mood_scores(candidates, v, e)
    m_scores = {t: float(s) for t, s in zip(candidates, m_vec)}
    ranked["mood"] = [t for t, _ in sorted(m_scores.items(), key=lambda p: (-p[1], p[0]))]

    for name, w in HYBRID_WEIGHTS.items():
        ranked[name] = [t for t, _ in combine(t_scores, m_scores, taste_weight=w,
                                              top_k=len(candidates))]
    return ranked


def evaluate(hist: pd.DataFrame | None, profiles: pd.DataFrame, cfg: dict, als, pop,
             n_per_set: int, n_users: int | None = None,
             views: dict | None = None) -> pd.DataFrame:
    """`views = None`: bộ chia cũ (che 30%). Khác None: bộ chia chính thức của nhóm."""
    ev = cfg["evaluation"]
    mood_pred = profiles["mood_pred"]

    # Kho nhạc của A2 = các bài ALS biết (đánh giá warm-item theo giao thức TV2)
    known = profiles.index.intersection(pd.Index(als.items))
    pred_known = mood_pred.loc[known]
    universe_by_mood: dict[str, np.ndarray] = {}
    for m in GROUPS:
        universe_by_mood[m] = pred_known.index[pred_known == m].to_numpy()
        universe_by_mood["_other_" + m] = pred_known.index[pred_known != m].to_numpy()
    known_set = set(known)

    if views is None:
        # User test phải ĐÚNG tập đã bị che khi huấn luyện ALS -> đọc từ taste_cohort.csv,
        # không rút lại bằng split_users (rút lại với size khác sẽ ra tập khác).
        cohort = pd.read_csv(COHORT_CSV, dtype={"user_id": str})
        test_users = np.sort(cohort.loc[cohort["is_test"], "user_id"].to_numpy())
        if n_users is not None and n_users < len(test_users):
            test_users = np.random.default_rng(SEED).choice(test_users, size=n_users,
                                                            replace=False)
        sub = hist[hist["user_id"].isin(set(test_users))]
        views = {}
        for user_id, g in sub.groupby("user_id"):
            tracks = g["track_id"].to_numpy()
            plays = g["playcount"].to_numpy(dtype=float)
            if len(tracks) < int(ev["min_user_tracks"]):
                continue
            visible, hidden = hidden_for_user(user_id, tracks, float(ev["hidden_ratio"]))
            order = {t: i for i, t in enumerate(tracks)}
            views[user_id] = (visible, np.array([plays[order[t]] for t in visible]), hidden)

    rows, n_cold = [], 0
    for user_id in sorted(views):
        visible, vis_w, hidden = views[user_id]
        if len(visible) < int(ev["min_user_tracks"]):
            continue

        # Bài bị che không có trong catalogue ALS -> loại (cold-item, giao thức TV2 mục 3)
        keep = np.isin(hidden, list(known_set))
        n_cold += int((~keep).sum())
        hidden = hidden[keep]
        if len(hidden) < 2 * n_per_set:
            continue

        listened = set(visible) | set(hidden)
        hidden_moods = mood_pred.reindex(hidden).to_numpy()

        for mood in GROUPS:
            case_rng = _case_rng(user_id, mood)
            sets = build_candidates(hidden, hidden_moods, listened, mood, mood_pred,
                                    universe_by_mood, n_per_set, case_rng)
            if sets is None:
                continue
            candidates = [t for key in SETS for t in sets[key]]
            ranked = rank_all(candidates, user_id, mood, visible, vis_w, als, pop, case_rng)

            for ranker, order_list in ranked.items():
                row = {"user_id": user_id, "mood": mood, "ranker": ranker,
                       "n_candidates": len(candidates)}
                for key in SETS:
                    row[key] = ndcg_at_k(order_list, set(sets[key]), n_per_set)
                rows.append(row)

    res = pd.DataFrame(rows)
    res.attrs["n_cold_dropped"] = n_cold
    return res


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    out = res.groupby("ranker")[SETS].mean().reindex(RANKERS)
    out["n_cases"] = res.groupby("ranker").size().reindex(RANKERS)
    return out


def plot_results(summary: pd.DataFrame, n_per_set: int, tag: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"random": "ngẫu nhiên", "pop": "phổ biến", "taste": "chỉ gu",
              "mood": "chỉ tâm trạng", "hybrid30": "ghép 0,3 gu",
              "hybrid50": "ghép 0,5 gu", "hybrid70": "ghép 0,7 gu"}
    colors = ["#4C72B0", "#8C8C8C", "#55A868", "#C44E52"]
    names = ["set1 gu + tâm trạng", "set2 chỉ gu", "set3 chỉ tâm trạng", "set4 không gì"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(summary)); h = 0.2
    for i, (col, c, nm) in enumerate(zip(SETS, colors, names)):
        ax.barh(y + (1.5 - i) * h, summary[col], h, label=nm, color=c)
    ax.set_yticks(y, [labels[r] for r in summary.index])
    ax.set_xlabel(f"NDCG@{n_per_set}")
    n_cases = f"{int(summary['n_cases'].iloc[0]):,}".replace(",", ".")  # dấu chấm phân cách nghìn
    ax.set_title(f"A2 — bốn tập ứng viên, N={n_per_set} bài mỗi tập "
                 f"({n_cases} cặp người dùng-nhóm)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"four_sets_n{n_per_set}{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="A2 — đánh giá bốn tập ứng viên.")
    parser.add_argument("--n-per-set", type=int, default=5, help="số bài mỗi tập (bài báo dùng 10)")
    parser.add_argument("--n-users", type=int, default=20000, help="số user test")
    parser.add_argument("--sample", action="store_true", help="chỉ 500 user test")
    parser.add_argument("--split", choices=["official", "local"], default="official",
                        help="official = bộ chia chung của nhóm; local = bộ chia cũ che 30%%")
    parser.add_argument("--weight", choices=["playcount", "engagement_score"],
                        default="playcount", help="trọng số tin cậy đã dùng khi luyện ALS")
    args = parser.parse_args()

    cfg = load_config()
    profiles = api.load_mood_profiles()
    als = load_als(args.split, args.weight)
    pop = fit_popularity(als)
    print(f"Mô hình gu: ALS {len(als.users):,} user x {len(als.items):,} bài "
          f"({als.config['factors']} factor, trọng số {args.weight})")

    n_users = 500 if args.sample else args.n_users
    if args.split == "official":
        from .splits import official_split, user_views
        train, test = official_split(set(profiles.index))
        users = np.sort(train["user_id"].unique())
        if n_users < len(users):
            users = np.sort(np.random.default_rng(SEED).choice(users, size=n_users,
                                                               replace=False))
        views = user_views(train, test, users, weight_col=args.weight)
        print(f"Bộ chia chính thức của nhóm | {len(views):,} user được đánh giá")
        res = evaluate(None, profiles, cfg, als, pop, args.n_per_set, views=views)
    else:
        hist = load_history(min_user_tracks=int(cfg["evaluation"]["min_user_tracks"]))
        hist = hist[hist["track_id"].isin(set(profiles.index))]
        res = evaluate(hist, profiles, cfg, als, pop, args.n_per_set, n_users=n_users)
    if res.empty:
        print("Không có cặp (user, nhóm) nào đủ bài cho N =", args.n_per_set)
        return

    n_cases = res.groupby(["user_id", "mood"]).ngroups
    print(f"N = {args.n_per_set} bài/tập -> {args.n_per_set * 4} ứng viên mỗi lượt | "
          f"{n_cases:,} cặp (user, nhóm) | {res['user_id'].nunique():,} user")
    print(f"Bài bị che bị loại vì ngoài catalogue ALS (cold-item): "
          f"{res.attrs['n_cold_dropped']:,}\n")

    summary = summarize(res)
    print(f"-- NDCG@{args.n_per_set} theo từng tập --")
    print(summary.to_string(float_format="{:.4f}".format))

    s1 = SETS[0]
    print(f"\nset1 (gu + tâm trạng): chỉ gu = {summary.loc['taste', s1]:.4f} | "
          f"ghép 0,5 = {summary.loc['hybrid50', s1]:.4f} "
          f"({(summary.loc['hybrid50', s1] / summary.loc['taste', s1] - 1) * 100:+.1f}%)")
    print(f"Mức phân biệt tâm trạng (set1 - set2): "
          f"chỉ gu = {summary.loc['taste', s1] - summary.loc['taste', SETS[1]]:+.4f} | "
          f"ghép 0,5 = {summary.loc['hybrid50', s1] - summary.loc['hybrid50', SETS[1]]:+.4f}")

    tag = "_official" if args.split == "official" else ""
    if args.weight != "playcount":
        tag += "_engage"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(PROCESSED_DIR / f"four_sets_eval_n{args.n_per_set}{tag}.csv", index=False)
    summary.to_csv(PROCESSED_DIR / f"four_sets_summary_n{args.n_per_set}{tag}.csv")
    print(f"\nĐã lưu: data/processed/four_sets_summary_n{args.n_per_set}{tag}.csv")
    print("Hình:", plot_results(summary, args.n_per_set, tag=tag))


if __name__ == "__main__":
    main()
