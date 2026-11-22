"""Sinh nhãn yếu từ tag Last.fm (task T2 / M2).

Nhãn mềm = tổng trọng số tag của bài, chuẩn hóa thành phân phối xác suất.
Tag cảm xúc trực tiếp có trọng số 1.0, tag thể loại mang cảm xúc nhân với
`support_weight` (xem configs/mood_config.yaml, khối labeling_tags).

Xuất: data/processed/track_mood_labels.parquet, reports/figures/mood_valence_energy_box.png
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import FIGURES_DIR, GROUPS, PROCESSED_DIR, load_config
from .io import load_tracks
from .tag_audit import find_suspect_tracks

SOFT_COLS = [f"y_{g}" for g in GROUPS]
LABELS_PARQUET = PROCESSED_DIR / "track_mood_labels.parquet"


def build_tag_weights(cfg: dict) -> dict[str, dict[str, float]]:
    """Gộp tag trực tiếp và tag hỗ trợ thành một bảng trọng số tag -> nhóm."""
    lt = cfg["labeling_tags"]
    weights: dict[str, dict[str, float]] = {}
    for tag, w in lt["direct"].items():
        weights[tag] = dict(w)
    sw = float(lt["support_weight"])
    for tag, w in lt["support"].items():
        # tag trực tiếp được ưu tiên nếu trùng
        weights.setdefault(tag, {g: v * sw for g, v in w.items()})
    return weights


def soft_labels(tags_lists: pd.Series, weights: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Nhãn mềm cho từng bài: tổng trọng số các tag, chuẩn hóa tổng = 1.

    Bài không có tag nào trong bảng trọng số -> toàn NaN.
    """
    raw = np.zeros((len(tags_lists), len(GROUPS)), dtype=float)
    gidx = {g: i for i, g in enumerate(GROUPS)}
    for row, tag_list in enumerate(tags_lists):
        for tag in tag_list:
            for g, w in weights.get(tag, {}).items():
                raw[row, gidx[g]] += w

    total = raw.sum(axis=1)
    out = np.full_like(raw, np.nan)
    has = total > 0
    out[has] = raw[has] / total[has, None]
    return pd.DataFrame(out, columns=SOFT_COLS, index=tags_lists.index)


def harden(soft: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Nhãn cứng + phát hiện mâu thuẫn.

    - conflict: không nhóm nào trội, tức hai nhóm dẫn đầu chênh nhau
      < conflict_margin, HOẶC nhóm dẫn đầu dưới min_label_confidence.
    - mood_label: nhóm có xác suất cao nhất, chỉ khi không phải conflict.
    """
    lab = cfg["labeling"]
    values = soft[SOFT_COLS].to_numpy()
    has = ~np.isnan(values).any(axis=1)

    top_idx = np.full(len(soft), -1)
    top_p = np.full(len(soft), np.nan)
    gap = np.full(len(soft), np.nan)
    if has.any():
        sub = values[has]
        order = np.argsort(-sub, axis=1)
        top_idx[has] = order[:, 0]
        top_p[has] = np.take_along_axis(sub, order[:, :1], axis=1).ravel()
        second = np.take_along_axis(sub, order[:, 1:2], axis=1).ravel()
        gap[has] = top_p[has] - second

    out = pd.DataFrame(index=soft.index)
    out["top_p"] = top_p
    # Không nhóm nào trội: hai nhóm đầu quá sát nhau, hoặc nhóm đầu chưa đủ tự tin
    weak = (gap < float(lab["conflict_margin"])) | (top_p < float(lab["min_label_confidence"]))
    out["is_conflict"] = has & weak
    confident = has & ~out["is_conflict"]
    out["mood_label"] = np.where(confident, [GROUPS[i] if i >= 0 else None for i in top_idx], None)
    out["has_label"] = has
    return out


def make_labels(df: pd.DataFrame, cfg: dict,
                weights: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """Bảng nhãn theo hợp đồng: track_id, y_*, mood_label, label_status.

    `weights=None` dùng bảng trọng số trong config (T2). Truyền bảng khác vào để so
    sánh cách gán nhãn mới mà không đụng tới nhãn chính thức (xem labeling_embed.py).
    """
    weights = build_tag_weights(cfg) if weights is None else weights
    soft = soft_labels(df["tags_list"], weights)
    hard = harden(soft, cfg)

    # Bài nghi ghép nhầm phiên bản (lý do mạnh) -> label_status = suspect
    suspects = find_suspect_tracks(df, cfg)
    strong = cfg["suspect_rules"]["strong_reasons"]
    strong_ids = set(
        suspects.loc[suspects["reason"].str.contains("|".join(strong)), "track_id"]
    )

    status = np.where(~hard["has_label"], "no_emotion_tag", "labeled")
    status = np.where(hard["is_conflict"] & hard["has_label"], "conflict", status)
    # suspect ưu tiên cao nhất trong các bài có nhãn (chất lượng dữ liệu)
    is_suspect = df["track_id"].isin(strong_ids).to_numpy()
    status = np.where(is_suspect & hard["has_label"].to_numpy(), "suspect", status)

    out = pd.DataFrame({"track_id": df["track_id"].to_numpy()})
    for col in SOFT_COLS:
        out[col] = soft[col].to_numpy()
    out["mood_label"] = hard["mood_label"].to_numpy()
    out["label_status"] = status
    return out


def kruskal_test(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Kiểm định Kruskal-Wallis: valence/energy có khác nhau giữa các nhóm không."""
    from scipy.stats import kruskal

    merged = df.merge(labels[["track_id", "mood_label", "label_status"]], on="track_id")
    merged = merged[merged["label_status"] == "labeled"]
    rows = []
    for feat in ["valence", "energy", "danceability", "acousticness", "loudness"]:
        samples = [g[feat].dropna().to_numpy() for _, g in merged.groupby("mood_label")]
        stat, p = kruskal(*samples)
        rows.append({"feature": feat, "H": stat, "p_value": p, "n_groups": len(samples)})
    return pd.DataFrame(rows)


def group_stats(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Trung vị valence/energy của từng nhóm (chỉ bài label_status = labeled)."""
    merged = df.merge(labels[["track_id", "mood_label", "label_status"]], on="track_id")
    merged = merged[merged["label_status"] == "labeled"]
    stats = merged.groupby("mood_label").agg(
        n=("track_id", "size"),
        valence_median=("valence", "median"),
        energy_median=("energy", "median"),
        acousticness_median=("acousticness", "median"),
    )
    return stats.reindex(GROUPS)


def plot_boxplot(df: pd.DataFrame, labels: pd.DataFrame) -> str:
    """Boxplot valence và energy theo nhóm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    merged = df.merge(labels[["track_id", "mood_label", "label_status"]], on="track_id")
    merged = merged[merged["label_status"] == "labeled"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, feat in zip(axes, ["valence", "energy"]):
        data = [merged.loc[merged["mood_label"] == g, feat].dropna() for g in GROUPS]
        bp = ax.boxplot(data, tick_labels=GROUPS, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], ["#DDAA33", "#4477AA", "#66BB6A", "#EE6677", "#9C27B0"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(f"{feat} theo nhóm tâm trạng (nhãn từ tag)")
        ax.set_ylabel(feat)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "mood_valence_energy_box.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh nhãn yếu từ tag.")
    parser.add_argument("--sample", action="store_true", help="chỉ chạy trên bản mẫu")
    parser.add_argument("--no-save", action="store_true", help="không ghi file parquet")
    args = parser.parse_args()

    cfg = load_config()
    df = load_tracks(sample=args.sample)
    labels = make_labels(df, cfg)

    n = len(labels)
    print(f"Số bài: {n:,}")
    print("\n-- label_status --")
    vc = labels["label_status"].value_counts()
    for k in ["labeled", "conflict", "suspect", "no_emotion_tag"]:
        print(f"  {k:<16} {int(vc.get(k, 0)):>7,} ({vc.get(k, 0)/n*100:5.2f}%)")

    print("\n-- Số bài theo nhóm (mood_label, chỉ label_status = labeled) --")
    stats = group_stats(df, labels)
    print(stats.to_string(float_format="{:.3f}".format))
    min_req = cfg["labeling"]["min_tracks_per_group"]
    thin = stats[stats["n"].fillna(0) < min_req]
    if len(thin):
        print(f"  !! Nhóm dưới min_tracks_per_group = {min_req}: {list(thin.index)}")
    else:
        print(f"  Mọi nhóm đều đạt min_tracks_per_group = {min_req}")

    print("\n-- Nhãn mềm trung bình (mọi bài có nhãn) --")
    print(labels[SOFT_COLS].mean().to_string(float_format="{:.3f}".format))

    print("\n-- Kruskal-Wallis (nhãn cứng vs đặc trưng âm thanh) --")
    kw = kruskal_test(df, labels)
    print(kw.to_string(index=False, float_format="{:.4g}".format))

    fig = plot_boxplot(df, labels)
    print(f"\nHình: {fig}")

    if not args.no_save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(LABELS_PARQUET, index=False)
        print(f"Đã lưu: {LABELS_PARQUET}")


if __name__ == "__main__":
    main()
