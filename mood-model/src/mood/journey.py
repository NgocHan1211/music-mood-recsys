"""Hành trình cảm xúc: playlist chuyển dần từ tâm trạng này sang tâm trạng khác (T7 / M7).

Người dùng chọn tâm trạng hiện tại A và tâm trạng mong muốn B. Đoạn A→B được chia
thành N điểm; mỗi điểm chọn bài có `mood_score` cao nhất chưa được chọn.

Xuất: reports/figures/mood_journey.png, data/processed/mood_journey_example.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import api
from .config import FIGURES_DIR, GROUPS, PROCESSED_DIR
from .io import load_tracks


def build_playlist(start: tuple[float, float], end: tuple[float, float], n_steps: int,
                   candidate_ids: list[str] | None = None) -> pd.DataFrame:
    """Chọn một bài cho mỗi điểm trên hành trình, không chọn trùng.

    Trả về bảng: step, target_valence, target_energy, track_id, mood_score.
    """
    profiles = api.load_mood_profiles()
    candidates = list(profiles.index) if candidate_ids is None else list(candidate_ids)
    points = api.mood_journey(start, end, n_steps)

    used: set[str] = set()
    rows = []
    for i, (v, e) in enumerate(points, start=1):
        scores = api.mood_scores(candidates, v, e)
        order = np.argsort(-scores)
        for pos in order:
            tid = candidates[pos]
            if tid not in used:
                used.add(tid)
                rows.append({"step": i, "target_valence": v, "target_energy": e,
                             "track_id": tid, "mood_score": float(scores[pos])})
                break
    return pd.DataFrame(rows)


def describe_playlist(playlist: pd.DataFrame) -> pd.DataFrame:
    """Thêm tên bài, nghệ sĩ, valence/energy thật và nhóm dự đoán."""
    tracks = load_tracks().set_index("track_id")
    profiles = api.load_mood_profiles()
    out = playlist.copy()
    out["name"] = tracks.loc[out["track_id"], "name"].to_numpy()
    out["artist"] = tracks.loc[out["track_id"], "artist"].to_numpy()
    out["valence"] = profiles.loc[out["track_id"], "valence"].to_numpy()
    out["energy"] = profiles.loc[out["track_id"], "energy"].to_numpy()
    out["mood_pred"] = profiles.loc[out["track_id"], "mood_pred"].to_numpy()
    return out


def plot_journey(playlist: pd.DataFrame, start_code: str, end_code: str) -> str:
    """Vẽ đường đi trên mặt phẳng valence–energy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiles = api.load_mood_profiles()
    bg = profiles.sample(n=min(4000, len(profiles)), random_state=42)
    colors = {"vui": "#DDAA33", "buon": "#4477AA", "thu_gian": "#66BB6A",
              "nang_dong": "#EE6677", "cang_thang": "#9C27B0"}

    fig, ax = plt.subplots(figsize=(9, 8))
    for g in GROUPS:
        s = bg[bg["mood_pred"] == g]
        ax.scatter(s["valence"], s["energy"], s=5, alpha=0.15, color=colors[g])

    ax.plot(playlist["target_valence"], playlist["target_energy"], "k--", lw=1.5,
            alpha=0.7, label="đường mục tiêu")
    ax.scatter(playlist["target_valence"], playlist["target_energy"], marker="x",
               s=70, color="black", zorder=4)
    ax.plot(playlist["valence"], playlist["energy"], "-o", color="#C44E52", lw=2,
            markersize=9, zorder=5, label="bài được chọn")
    for _, r in playlist.iterrows():
        ax.annotate(str(int(r["step"])), (r["valence"], r["energy"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9, zorder=6)

    for code, marker in ((start_code, "bắt đầu"), (end_code, "kết thúc")):
        v, e = api.quick_mood(code)
        ax.scatter(v, e, marker="*", s=320, color=colors[code], edgecolor="black",
                   zorder=7, label=f"{marker}: {code}")

    ax.set_xlabel("valence"); ax.set_ylabel("energy")
    ax.set_title(f"Hành trình cảm xúc: {start_code} → {end_code} ({len(playlist)} bước)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "mood_journey.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo playlist chuyển dần tâm trạng.")
    parser.add_argument("--start", default="buon", choices=GROUPS)
    parser.add_argument("--end", default="vui", choices=GROUPS)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--sample", action="store_true", help="chỉ xét 5.000 bài ứng viên")
    args = parser.parse_args()

    start, end = api.quick_mood(args.start), api.quick_mood(args.end)
    print(f"{args.start} {tuple(round(x, 3) for x in start)} -> "
          f"{args.end} {tuple(round(x, 3) for x in end)}, {args.steps} bước\n")

    candidates = None
    if args.sample:
        candidates = api.load_mood_profiles().sample(5000, random_state=42).index.tolist()

    playlist = describe_playlist(build_playlist(start, end, args.steps, candidates))
    cols = ["step", "target_valence", "target_energy", "name", "artist",
            "valence", "energy", "mood_pred", "mood_score"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 28):
        print(playlist[cols].to_string(index=False, float_format="{:.3f}".format))

    print(f"\nvalence đi từ {playlist['valence'].iloc[0]:.3f} tới "
          f"{playlist['valence'].iloc[-1]:.3f}; "
          f"energy từ {playlist['energy'].iloc[0]:.3f} tới {playlist['energy'].iloc[-1]:.3f}")

    if not args.sample:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        playlist.to_csv(PROCESSED_DIR / "mood_journey_example.csv", index=False)
        print("Hình:", plot_journey(playlist, args.start, args.end))


if __name__ == "__main__":
    main()
