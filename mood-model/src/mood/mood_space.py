"""Không gian tâm trạng học từ dữ liệu (task T4 / M4).

Mỗi nhóm = một Gaussian 2 chiều trên (valence, energy), tâm và hiệp phương sai
tính có trọng số theo xác suất nhãn. Đây là "vị trí các nhóm trên thanh kéo",
thay cho tọa độ tự đặt tay của bản baseline.

Xuất: data/processed/mood_centroids.json, reports/figures/mood_space.png
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .config import FIGURES_DIR, GROUPS, PROCESSED_DIR, load_config
from .predict import PROB_COLS, TRACK_MOOD_PARQUET

CENTROIDS_JSON = PROCESSED_DIR / "mood_centroids.json"


def weighted_gaussian(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tâm và hiệp phương sai có trọng số của một đám điểm 2 chiều."""
    w = weights / weights.sum()
    mean = (w[:, None] * points).sum(axis=0)
    d = points - mean
    cov = (w[:, None, None] * (d[:, :, None] * d[:, None, :])).sum(axis=0)
    # hiệu chỉnh độ chệch của ước lượng có trọng số
    cov = cov / max(1e-12, 1 - (w ** 2).sum())
    return mean, cov


def build_centroids(track_mood: pd.DataFrame, cfg: dict, labeled_only: bool = True) -> dict:
    """Tâm + hiệp phương sai mỗi nhóm, trọng số là xác suất thuộc nhóm đó.

    Theo docs/mood_spec.md (M4), chỉ tính trên các bài CÓ NHÃN từ tag
    (`mood_source == "tag"`). Nếu tính trên mọi bài, các tâm bị kéo về
    trung bình chung của toàn bộ dữ liệu (xem reports/RESULTS.md T4).
    """
    if labeled_only:
        track_mood = track_mood[track_mood["mood_source"] == "tag"]
    points = track_mood[["valence", "energy"]].to_numpy(dtype=float)
    centroids = {}
    for i, g in enumerate(GROUPS):
        w = track_mood[PROB_COLS[i]].to_numpy(dtype=float)
        mean, cov = weighted_gaussian(points, w)
        centroids[g] = {
            "mean": [round(float(v), 6) for v in mean],
            "cov": [[round(float(c), 8) for c in row] for row in cov],
            "n": int((track_mood["mood_pred"] == g).sum()),
            "weight_sum": round(float(w.sum()), 3),
        }
    return {
        "groups": GROUPS,
        "centroids": centroids,
        "manual_baseline": {g: list(map(float, cfg["manual_centroids"][g])) for g in GROUPS},
    }


def group_probabilities(valence: float, energy: float, centroids: dict) -> dict[str, float]:
    """Điểm (valence, energy) bất kỳ -> xác suất thuộc từng nhóm (Gaussian, chuẩn hóa)."""
    x = np.array([valence, energy], dtype=float)
    dens = []
    for g in GROUPS:
        mean = np.array(centroids["centroids"][g]["mean"])
        cov = np.array(centroids["centroids"][g]["cov"])
        inv = np.linalg.inv(cov)
        d = x - mean
        dens.append(float(np.exp(-0.5 * d @ inv @ d) / np.sqrt(np.linalg.det(cov))))
    total = sum(dens)
    return {g: (v / total if total > 0 else 1 / len(GROUPS)) for g, v in zip(GROUPS, dens)}


def plot_mood_space(track_mood: pd.DataFrame, centroids: dict, sample_n: int = 4000) -> str:
    """Scatter (valence, energy) + ellipse 1σ mỗi nhóm + điểm tự đặt cũ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    colors = {"vui": "#DDAA33", "buon": "#4477AA", "thu_gian": "#66BB6A",
              "nang_dong": "#EE6677", "cang_thang": "#9C27B0"}
    sub = track_mood.sample(n=min(sample_n, len(track_mood)), random_state=42)

    fig, ax = plt.subplots(figsize=(9, 8))
    for g in GROUPS:
        s = sub[sub["mood_pred"] == g]
        ax.scatter(s["valence"], s["energy"], s=6, alpha=0.25, color=colors[g], label=None)

    for g in GROUPS:
        mean = np.array(centroids["centroids"][g]["mean"])
        cov = np.array(centroids["centroids"][g]["cov"])
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
        w, h = 2 * np.sqrt(np.maximum(vals[::-1], 0))
        ax.add_patch(Ellipse(mean, w, h, angle=angle, facecolor="none",
                             edgecolor=colors[g], lw=2.5))
        ax.scatter(*mean, marker="o", s=160, color=colors[g], edgecolor="black",
                   zorder=5, label=f"{g} (học được)")
        mb = centroids["manual_baseline"][g]
        ax.scatter(*mb, marker="X", s=160, color=colors[g], edgecolor="black", zorder=5)
        ax.annotate("", xy=mean, xytext=mb,
                    arrowprops=dict(arrowstyle="->", color=colors[g], lw=1.2, alpha=0.8))

    ax.scatter([], [], marker="X", s=120, color="gray", edgecolor="black",
               label="tọa độ tự đặt cũ (X)")
    ax.set_xlabel("valence"); ax.set_ylabel("energy")
    ax.set_title("Không gian tâm trạng học từ dữ liệu (ellipse 1σ) so với tọa độ tự đặt")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "mood_space.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Học không gian tâm trạng từ dữ liệu.")
    parser.add_argument("--sample", action="store_true", help="không dùng (giữ cho đồng bộ CLI)")
    args = parser.parse_args()

    cfg = load_config()
    track_mood = pd.read_parquet(TRACK_MOOD_PARQUET)
    centroids = build_centroids(track_mood, cfg, labeled_only=True)
    centroids_all = build_centroids(track_mood, cfg, labeled_only=False)
    n_lab = int((track_mood["mood_source"] == "tag").sum())
    print(f"Tính tâm trên {n_lab:,} bài có nhãn từ tag "
          f"(toàn bộ: {len(track_mood):,} bài — chỉ để đối chiếu)\n")

    print("-- Tâm học được so với tọa độ tự đặt --")
    print(f"{'nhóm':<12}{'học được':>18}{'tự đặt':>16}{'lệch':>9}{'n':>9}")
    for g in GROUPS:
        m = centroids["centroids"][g]["mean"]
        b = centroids["manual_baseline"][g]
        d = float(np.hypot(m[0] - b[0], m[1] - b[1]))
        n = centroids["centroids"][g]["n"]
        print(f"{g:<12}({m[0]:.3f}, {m[1]:.3f}){'':>3}({b[0]:.2f}, {b[1]:.2f}){'':>3}{d:>8.3f}{n:>9,}")

    print("\n-- Đối chiếu: nếu tính trên MỌI bài (kể cả bài mô hình suy ra) --")
    for g in GROUPS:
        m = centroids["centroids"][g]["mean"]
        a = centroids_all["centroids"][g]["mean"]
        print(f"  {g:<12} có nhãn ({m[0]:.3f}, {m[1]:.3f})  vs  mọi bài ({a[0]:.3f}, {a[1]:.3f})")

    print("\n-- Độ lệch chuẩn mỗi nhóm (căn của đường chéo hiệp phương sai) --")
    for g in GROUPS:
        cov = np.array(centroids["centroids"][g]["cov"])
        print(f"  {g:<12} sd_valence={np.sqrt(cov[0,0]):.3f}  sd_energy={np.sqrt(cov[1,1]):.3f}  "
              f"corr={cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]):+.3f}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(CENTROIDS_JSON, "w", encoding="utf-8") as f:
        json.dump(centroids, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu: {CENTROIDS_JSON}")
    print("Hình:", plot_mood_space(track_mood[track_mood["mood_source"] == "tag"], centroids))

    print("\n-- Thử thanh kéo: điểm (v, e) -> xác suất nhóm --")
    for v, e in [(0.8, 0.8), (0.2, 0.3), (0.5, 0.4), (0.3, 0.9)]:
        probs = group_probabilities(v, e, centroids)
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:2]
        print(f"  ({v}, {e}) -> " + ", ".join(f"{g} {p*100:.0f}%" for g, p in top))


if __name__ == "__main__":
    main()
