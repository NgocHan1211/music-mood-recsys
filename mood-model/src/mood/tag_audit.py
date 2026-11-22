"""Kiểm kê tag Last.fm và rà bài nghi ghép nhầm phiên bản (task T1 / M1).

Xuất:
  data/processed/tag_audit.csv        tag, n_tracks, category
  data/processed/suspect_tracks.csv   track_id, name, artist, reason
  data/processed/tag_candidate_check.csv  đối chiếu candidate_tags với tag thật
  reports/figures/top30_tags.png
"""

from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd

from .config import FIGURES_DIR, PROCESSED_DIR, load_config
from .io import load_tracks

# Các category theo docs/data_contracts.md
CATEGORIES = ["genre", "decade", "country", "emotion", "other"]

# Thập niên -> năm cuối, để đo lệch giữa tag thập niên và năm phát hành
DECADE_END = {"50s": 1959, "60s": 1969, "70s": 1979, "80s": 1989, "90s": 1999, "00s": 2009}
OLDIES_END = 1969  # coi "oldies" như nhạc trước 1970


def assign_category(tag: str, cfg: dict) -> str:
    """Gán category cho một tag theo danh sách trong config.

    Thứ tự ưu tiên: decade -> country -> genre -> descriptor -> emotion -> other.
    Tag mô tả (descriptor: female_vocalists, acoustic, piano...) xếp vào `other`.
    """
    hints = cfg["non_emotion_hints"]
    if tag in hints.get("decade", []):
        return "decade"
    if tag in hints.get("country", []):
        return "country"
    if tag in hints.get("genre", []):
        return "genre"
    if tag in hints.get("descriptor", []):
        return "other"
    if tag in cfg["candidate_tags"] or tag in cfg["ambiguous"]:
        return "emotion"
    return "other"


def build_tag_audit(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Đếm số bài của từng tag và gán category. Sắp theo n_tracks giảm dần."""
    counts = Counter(tag for tag_list in df["tags_list"] for tag in tag_list)
    audit = pd.DataFrame(
        {"tag": list(counts.keys()), "n_tracks": list(counts.values())}
    )
    audit["category"] = audit["tag"].map(lambda t: assign_category(t, cfg))
    audit = audit.sort_values(["n_tracks", "tag"], ascending=[False, True])
    return audit.reset_index(drop=True)


def check_candidates(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Đối chiếu candidate_tags và ambiguous trong config với tag thật có."""
    counts = Counter(tag for tag_list in df["tags_list"] for tag in tag_list)
    rows = []
    for tag, weights in cfg["candidate_tags"].items():
        rows.append({
            "tag": tag,
            "list": "candidate",
            "groups": ", ".join(f"{g}:{w}" for g, w in weights.items()),
            "exists": tag in counts,
            "n_tracks": counts.get(tag, 0),
        })
    for tag in cfg["ambiguous"]:
        rows.append({
            "tag": tag, "list": "ambiguous", "groups": "",
            "exists": tag in counts, "n_tracks": counts.get(tag, 0),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["list", "n_tracks", "tag"], ascending=[True, False, True])


def _era_mismatch(row: pd.Series, gap: int) -> bool:
    """True nếu tag thập niên cách năm phát hành quá `gap` năm (tín hiệu yếu)."""
    if pd.isna(row.get("year")):
        return False
    ends = [DECADE_END[t] for t in row["tags_list"] if t in DECADE_END]
    if "oldies" in row["tags_list"]:
        ends.append(OLDIES_END)
    return bool(ends) and (row["year"] - max(ends) > gap)


def find_suspect_tracks(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Gắn cờ bài nghi ghép nhầm phiên bản (bản cover / live / remaster).

    Lý do mạnh (loại khỏi tập huấn luyện ở T3): cover_tag, name_version,
    spotify_id_dup, duration_outlier.
    Lý do yếu (chỉ ghi nhận): era_mismatch, title_shared.
    """
    rules = cfg["suspect_rules"]
    flags = pd.DataFrame(index=df.index)
    flags["cover_tag"] = df["tags_list"].map(lambda ts: "cover" in ts)
    flags["name_version"] = (
        df["name"].astype(str).str.lower()
        .str.contains(rules["version_name_pattern"], regex=True, na=False)
    )
    flags["spotify_id_dup"] = df["spotify_id_dup"].astype(bool)
    flags["duration_outlier"] = df["duration_outlier"].astype(bool)
    flags["era_mismatch"] = df.apply(_era_mismatch, axis=1, gap=rules["era_gap_years"])

    # Cùng tên bài nhưng khác nghệ sĩ -> có thể là bản cover của nhau
    norm_name = df["name"].astype(str).str.lower().str.strip()
    n_artists = df.assign(_n=norm_name).groupby("_n")["artist"].transform("nunique")
    flags["title_shared"] = n_artists > 1

    reason = flags.apply(lambda r: ";".join(c for c in flags.columns if r[c]), axis=1)
    out = df.loc[reason != "", ["track_id", "name", "artist"]].copy()
    out["reason"] = reason[reason != ""]
    out = out.sort_values("track_id").reset_index(drop=True)
    out.attrs["flags"] = flags
    return out


def popular_tracks(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """50 bài "phổ biến" nhất. Không có User_Listening_History.csv nên dùng số tag
    làm proxy theo docs/mood_spec.md (M1)."""
    return df.sort_values(["n_tags", "track_id"], ascending=[False, True]).head(n)


def plot_top_tags(audit: pd.DataFrame, top_n: int = 30) -> str:
    """Vẽ top tag theo tần suất, tô màu theo category."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = audit.head(top_n).iloc[::-1]
    colors = {"genre": "#4C72B0", "decade": "#DD8452", "country": "#55A868",
              "emotion": "#C44E52", "other": "#8C8C8C"}

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["tag"], top["n_tracks"],
            color=[colors[c] for c in top["category"]])
    ax.set_xlabel("Số bài")
    ax.set_title(f"Top {top_n} tag Last.fm theo tần suất (tô màu theo loại tag)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in CATEGORIES]
    ax.legend(handles, CATEGORIES, title="Loại tag", loc="lower right")
    for y, v in zip(range(len(top)), top["n_tracks"]):
        ax.text(v + 60, y, f"{v:,}", va="center", fontsize=8)
    ax.margins(x=0.12)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"top{top_n}_tags.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm kê tag và rà bài nghi bản cover.")
    parser.add_argument("--sample", action="store_true", help="chỉ chạy trên bản mẫu")
    args = parser.parse_args()

    cfg = load_config()
    df = load_tracks(sample=args.sample)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Kiểm kê tag
    audit = build_tag_audit(df, cfg)
    audit_path = PROCESSED_DIR / "tag_audit.csv"
    audit.to_csv(audit_path, index=False)
    print(f"Số bài: {len(df):,} | số tag khác nhau: {len(audit)}")
    print("\n-- Số tag và số lượt gán theo loại --")
    by_cat = audit.groupby("category").agg(
        n_tags=("tag", "size"), n_tag_assignments=("n_tracks", "sum")).reindex(CATEGORIES)
    print(by_cat.to_string())

    # 2. Tag cảm xúc thật có
    emo = audit[audit["category"] == "emotion"]
    n_with_emo = int(df["tags_list"].map(lambda ts: bool(set(ts) & set(emo["tag"]))).sum())
    print(f"\n-- Tag cảm xúc THẬT CÓ trong dữ liệu ({len(emo)} tag) --")
    print(emo.to_string(index=False))
    print(f"Số bài có ít nhất 1 tag cảm xúc: {n_with_emo:,} ({n_with_emo/len(df)*100:.2f}%)")

    check = check_candidates(df, cfg)
    check_path = PROCESSED_DIR / "tag_candidate_check.csv"
    check.to_csv(check_path, index=False)
    missing = check.loc[~check["exists"], "tag"].tolist()
    print(f"\n-- Tag ứng viên trong config KHÔNG tồn tại ({len(missing)}) --")
    print(", ".join(sorted(missing)))

    # 3. Rà bài nghi ghép nhầm phiên bản
    suspects = find_suspect_tracks(df, cfg)
    suspects.to_csv(PROCESSED_DIR / "suspect_tracks.csv", index=False)
    flags = suspects.attrs["flags"]
    print("\n-- Số bài theo từng dấu hiệu nghi vấn --")
    for col in flags.columns:
        print(f"  {col:<18} {int(flags[col].sum()):>6,}")
    strong = cfg["suspect_rules"]["strong_reasons"]
    n_strong = int(flags[strong].any(axis=1).sum())
    print(f"  {'=> lý do mạnh':<18} {n_strong:>6,} ({n_strong/len(df)*100:.2f}%) -> loại khỏi huấn luyện T3")
    print(f"  {'=> tổng gắn cờ':<18} {len(suspects):>6,} ({len(suspects)/len(df)*100:.2f}%)")

    top = popular_tracks(df, 50)
    top_suspect = suspects[suspects["track_id"].isin(top["track_id"])]
    top_strong = top_suspect[top_suspect["reason"].str.contains("|".join(strong))]
    print(f"\n-- Trong 50 bài 'phổ biến' nhất (proxy: số tag): "
          f"{len(top_suspect)} bài bị gắn cờ, {len(top_strong)} bài có lý do mạnh --")
    if len(top_strong):
        print(top_strong.to_string(index=False, max_colwidth=36))

    # 4. Hình
    fig_path = plot_top_tags(audit, 30)
    print(f"\nĐã lưu: {audit_path}\n          {check_path}\n"
          f"          {PROCESSED_DIR / 'suspect_tracks.csv'}\n          {fig_path}")


if __name__ == "__main__":
    main()
