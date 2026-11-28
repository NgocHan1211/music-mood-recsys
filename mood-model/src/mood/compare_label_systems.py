"""Đối chiếu bốn hệ gán nhãn tâm trạng trên cùng một tập bài.

KHÔNG phải nghiên cứu đồng thuận và KHÔNG sinh nhãn vàng. `score_agreement.py` mới là
cái đó, và nó cần nhãn người. Module này trả lời câu hỏi yếu hơn nhưng vẫn dùng được:
**bốn hệ nhãn khớp nhau ở đâu và lệch nhau ở đâu**, tách theo việc tag có dùng được hay không.

Bốn hệ, kèm nguồn thông tin của từng hệ:
    mood_label (T2)   bảng trọng số tag Last.fm          -> dùng TAG
    mood_pred  (T4)   bộ phân loại trên đặc trưng âm thanh -> dùng ÂM THANH
    llm_ten    (L1)   LLM đọc tên bài + nghệ sĩ           -> dùng TÊN
    llm_ten_tag(L1)   LLM đọc tên bài + nghệ sĩ + tag     -> dùng TÊN + TAG

Ba hệ đầu không chia sẻ đầu vào nào. Khớp nhau giữa chúng là bằng chứng hội tụ; lệch
nhau chỉ khoanh được vùng có vấn đề, **không nói được hệ nào sai**. Muốn biết hệ nào
sai thì phải có nhãn người — xem `docs/huong_dan_gan_nhan_mood.md`.

Chạy:
    python -m src.mood.compare_label_systems
    python -m src.mood.compare_label_systems --tracks-file <csv có cột track_id>
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GROUPS, PROCESSED_DIR

MOOD_ROOT = Path(__file__).resolve().parents[2]
UNKNOWN = "khong_ro"

# tên cột trong bảng ghép -> (nhãn hiển thị, nguồn thông tin)
# CẨN TRỌNG: `mood_pred` trong track_mood.parquet KHÔNG phải dự đoán mô hình ở mọi bài.
# Cột `mood_source` cho biết: "tag" ở 26.752 bài có nhãn tag (mood_pred = chính nhãn tag),
# "model" ở phần còn lại. Nếu so `mood_pred` với `mood_label` mà không lọc thì được 99,88%
# — một con số vô nghĩa vì hai cột là một. Nên ở đây T4 chỉ được so trên bài
# `mood_source == "model"`, tức chỗ mô hình thật sự phải đoán.
SYSTEMS = {
    "mood_label":       ("T2 bảng tag", "tag"),
    "mood_pred_model":  ("T4 mô hình thật đoán", "âm thanh"),
    "llm_ten":          ("L1 LLM tên bài", "tên"),
    "llm_ten_tag":      ("L1 LLM tên + tag", "tên + tag"),
}
LLM_FILES = {
    "llm_ten":     "llm_labels_openai_compat_gpt-oss-120b_llm_ten_gold.csv",
    "llm_ten_tag": "llm_labels_openai_compat_gpt-oss-120b_llm_ten_tag_gold.csv",
}


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa trên các bài cả hai hệ đều có nhãn thật (bỏ rỗng và khong_ro)."""
    ok = a.notna() & b.notna() & a.ne("") & b.ne("") & a.ne(UNKNOWN) & b.ne(UNKNOWN)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return float("nan")
    cats = sorted(set(a) | set(b))
    obs = float((a.to_numpy() == b.to_numpy()).mean())
    exp = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float((obs - exp) / (1 - exp)) if exp < 1 else float("nan")


def agreement(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    ok = a.notna() & b.notna() & a.ne("") & b.ne("") & a.ne(UNKNOWN) & b.ne(UNKNOWN)
    if not ok.any():
        return (float("nan"), 0)
    return (float((a[ok] == b[ok]).mean()), int(ok.sum()))


def load_table(tracks_file: Path) -> pd.DataFrame:
    """Ghép bốn hệ nhãn lên cùng danh sách bài."""
    key = pd.read_csv(tracks_file, dtype=str)
    if "track_id" not in key:
        raise ValueError(f"{tracks_file.name} không có cột track_id")
    keep = [c for c in ["item_id", "track_id", "label_status", "mood_label",
                        "mood_pred", "mood_source", "name", "artist"] if c in key]
    out = key[keep].copy()
    # T4 chỉ tính ở chỗ mô hình thật sự dự đoán; chỗ mood_source == "tag" thì
    # mood_pred là bản sao của mood_label, so sẽ ra 100% một cách giả tạo.
    if "mood_pred" in out and "mood_source" in out:
        out["mood_pred_model"] = out["mood_pred"].where(out["mood_source"].eq("model"))
        n_tag = int(out["mood_source"].eq("tag").sum())
        if n_tag:
            print(f"  [i] {n_tag} bài có mood_source='tag' -> mood_pred là bản sao của "
                  f"mood_label, đã loại khỏi dòng T4")
    elif "mood_pred" in out:
        out["mood_pred_model"] = out["mood_pred"]
        print("  [!] không có cột mood_source — dòng T4 có thể lẫn nhãn tag")
    for col, fname in LLM_FILES.items():
        p = PROCESSED_DIR / fname
        if not p.exists():
            print(f"  [!] thiếu {fname} — bỏ qua {col}")
            continue
        d = pd.read_csv(p, dtype=str).drop_duplicates("track_id")
        out = out.merge(d[["track_id", "llm_nhom"]].rename(columns={"llm_nhom": col}),
                        on="track_id", how="left")
    return out


def pairwise(tab: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(cols, 2):
        ag, n = agreement(tab[a], tab[b])
        rows.append({"hệ A": SYSTEMS[a][0], "hệ B": SYSTEMS[b][0],
                     "đầu vào": f"{SYSTEMS[a][1]} ↔ {SYSTEMS[b][1]}",
                     "n": n, "khớp_%": 100 * ag if n else np.nan,
                     "cohen_kappa": cohen_kappa(tab[a], tab[b])})
    return pd.DataFrame(rows).sort_values("khớp_%", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đối chiếu bốn hệ gán nhãn tâm trạng (không sinh nhãn vàng).")
    parser.add_argument("--tracks-file", type=Path,
                        default=MOOD_ROOT / "annotation" / "KEY_KHONG_MO_TRUOC_KHI_GAN.csv")
    parser.add_argument("--out", type=Path,
                        default=PROCESSED_DIR / "label_systems_compare.csv")
    args = parser.parse_args()

    tab = load_table(args.tracks_file)
    cols = [c for c in SYSTEMS if c in tab]
    print(f"Tập bài: {len(tab)} bài từ {args.tracks_file.name}")
    print(f"Hệ nhãn có mặt: {', '.join(SYSTEMS[c][0] for c in cols)}\n")

    print("-- Độ phủ: hệ nào dám gán nhãn cho bao nhiêu bài --")
    for c in cols:
        s = tab[c]
        real = s.notna() & s.ne("") & s.ne(UNKNOWN)
        unk = s.eq(UNKNOWN).sum()
        print(f"  {SYSTEMS[c][0]:<18} ({SYSTEMS[c][1]:<9}) {real.sum():3d}/{len(tab)} bài "
              f"({100*real.mean():5.1f}%) | khong_ro {unk:3d} | trống {int(s.isna().sum() + s.eq('').sum()):3d}")

    print("\n-- Khớp nhau từng cặp, trên toàn tập --")
    allp = pairwise(tab, cols)
    print(allp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    frames = [allp.assign(tang="tất cả")]
    if "label_status" in tab:
        for status, g in tab.groupby("label_status"):
            print(f"\n-- Tầng `{status}` ({len(g)} bài) --")
            p = pairwise(g, cols)
            print(p.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            frames.append(p.assign(tang=status))

    # Không thể tam giác hoá ba hệ trên cùng một bài: T2 chỉ tồn tại ở bài có tag cảm
    # xúc, còn T4-mô-hình chỉ tồn tại ở bài KHÔNG có tag. Hai cột loại trừ nhau theo
    # thiết kế, nên "ba nguồn rời nhau cùng xác nhận một nhãn" là chuyện bất khả thi.
    print("\n-- Cặp nào thật sự so được với nhau --")
    if {"mood_label", "mood_pred_model"} <= set(tab):
        overlap = (tab["mood_label"].notna() & tab["mood_label"].ne("")
                   & tab["mood_pred_model"].notna()).sum()
        print(f"  bài có CẢ T2 và T4-mô-hình: {overlap} -> hai hệ loại trừ nhau theo thiết kế")
    print("  Chỉ còn hai phép so, trên hai tập bài rời nhau:")
    for status, who, lab in [("có tag cảm xúc", "mood_label", "T2 ↔ L1"),
                             ("tag bó tay", "mood_pred_model", "T4 ↔ L1")]:
        if who not in tab or "llm_ten" not in tab:
            continue
        a, b = tab[who], tab["llm_ten"]
        ok = (a.notna() & a.ne("") & a.ne(UNKNOWN)
              & b.notna() & b.ne("") & b.ne(UNKNOWN))
        if not ok.any():
            continue
        print(f"    {lab:<9} ({status:<14}) n={int(ok.sum()):3d} | "
              f"khớp {100*float((a[ok]==b[ok]).mean()):5.1f}% | "
              f"kappa {cohen_kappa(a, b):.3f}")
    print("\n  Vì vậy KHÔNG có cách nào xác nhận nhãn bằng chính các hệ tự động này.")
    print("  Chỗ cần nhãn nhất (tag bó tay) là chỗ hai hệ còn lại gần như không đồng ý gì.")
    print("  Chỉ nhãn người phá được vòng này — xem docs/huong_dan_gan_nhan_mood.md.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(args.out, index=False)
    print(f"\nĐã ghi: {args.out.relative_to(MOOD_ROOT)}")


if __name__ == "__main__":
    main()
