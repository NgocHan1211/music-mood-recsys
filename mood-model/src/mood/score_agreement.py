"""Chấm bộ nhãn do người gán: mức đồng thuận, nhãn chốt, và so với các cách gán nhãn máy.

Chạy sau khi tất cả `annotation/nhan_<ten>.csv` đã điền xong.
Xem `docs/huong_dan_gan_nhan_mood.md`.

Sinh ra:
  annotation/mood_gold.csv            -> nhãn chốt (đa số phiếu) + cờ tranh chấp
  annotation/agreement_report.csv     -> bảng số để dán vào báo cáo

Chạy:
    python -m src.mood.score_agreement
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GROUPS, PROCESSED_DIR

MOOD_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = MOOD_ROOT / "annotation"
LABELS = GROUPS + ["khong_ro"]
KEY_CSV = ANNOTATION_DIR / "KEY_KHONG_MO_TRUOC_KHI_GAN.csv"


def fleiss_kappa(counts: np.ndarray) -> float:
    """Fleiss' kappa. `counts[i, j]` = số người gán bài i vào nhóm j.

    Cần mọi bài có cùng số người gán (bài thiếu nhãn phải loại trước khi gọi).
    """
    n_items, _ = counts.shape
    n_raters = int(counts[0].sum())
    if n_raters < 2:
        raise ValueError("cần ít nhất 2 người gán")
    if not (counts.sum(axis=1) == n_raters).all():
        raise ValueError("số người gán không đều giữa các bài")
    p_item = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_item.mean()
    p_cat = counts.sum(axis=0) / (n_items * n_raters)
    p_exp = (p_cat ** 2).sum()
    return float((p_bar - p_exp) / (1 - p_exp)) if p_exp < 1 else float("nan")


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa cho hai người gán, trên các bài cả hai đều có nhãn."""
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    if not len(a):
        return float("nan")
    cats = sorted(set(a) | set(b))
    obs = float((a.to_numpy() == b.to_numpy()).mean())
    exp = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float((obs - exp) / (1 - exp)) if exp < 1 else float("nan")


def read_annotations(who: list[str] | None) -> pd.DataFrame:
    """Đọc các file nhan_*.csv thành bảng rộng: item_id × người gán."""
    files = sorted(ANNOTATION_DIR.glob("nhan_*.csv"))
    if who:
        files = [f for f in files if f.stem.replace("nhan_", "") in who]
    if len(files) < 2:
        raise FileNotFoundError(
            f"Chỉ thấy {len(files)} file nhan_*.csv trong {ANNOTATION_DIR}. "
            "Cần ít nhất 2 người gán để đo đồng thuận.")
    wide = None
    for f in files:
        name = f.stem.replace("nhan_", "")
        d = pd.read_csv(f, dtype=str)
        if "item_id" not in d or "mood_chinh" not in d:
            raise ValueError(f"{f.name} thiếu cột item_id hoặc mood_chinh")
        d["mood_chinh"] = d.mood_chinh.fillna("").str.strip().str.lower()
        bad = set(d.mood_chinh.unique()) - set(LABELS) - {""}
        if bad:
            raise ValueError(
                f"{f.name} có mã nhóm sai: {sorted(bad)}. Hợp lệ: {LABELS}")
        col = d[["item_id", "mood_chinh"]].rename(columns={"mood_chinh": name})
        col[name] = col[name].replace("", np.nan)
        wide = col if wide is None else wide.merge(col, on="item_id", how="outer")
        print(f"  {f.name}: {col[name].notna().sum()}/{len(col)} bài đã gán")
    return wide.set_index("item_id").sort_index()


def majority_vote(row: pd.Series) -> tuple[str, int, bool]:
    """Nhãn chốt = nhóm được từ 2 phiếu trở lên. Không ai quá bán -> tranh chấp."""
    votes = row.dropna()
    if votes.empty:
        return ("", 0, False)
    tally = votes.value_counts()
    top, n_top = tally.index[0], int(tally.iloc[0])
    tied = (tally == n_top).sum() > 1
    if n_top >= 2 and not tied:
        return (str(top), n_top, False)
    return ("", n_top, True)


def compare(gold: pd.DataFrame, col: str, label: str) -> dict:
    """Tỷ lệ khớp giữa nhãn chốt và một cách gán nhãn máy."""
    ok = gold[gold.mood_chot.ne("") & gold[col].notna() & gold[col].ne("")]
    if not len(ok):
        return {"cach_gan_nhan": label, "n": 0, "khop_pct": np.nan}
    out = {"cach_gan_nhan": label, "n": len(ok),
           "khop_pct": 100 * float((ok.mood_chot == ok[col]).mean())}
    for status, g in ok.groupby("label_status"):
        out[f"khop_{status}"] = 100 * float((g.mood_chot == g[col]).mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chấm bộ nhãn người gán: Fleiss' kappa, nhãn chốt, so với nhãn máy.")
    parser.add_argument("--labellers", default="",
                        help="chỉ dùng các người này (phân tách bằng dấu phẩy); rỗng = tất cả")
    args = parser.parse_args()
    who = [s.strip() for s in args.labellers.split(",") if s.strip()]

    print("Đọc file gán nhãn:")
    wide = read_annotations(who)
    raters = list(wide.columns)

    # --- đồng thuận ---------------------------------------------------------
    full = wide.dropna()
    print(f"\n{len(full)}/{len(wide)} bài có đủ nhãn của cả {len(raters)} người "
          "(chỉ các bài này vào Fleiss' kappa)")
    if len(full) < 10:
        raise SystemExit("Quá ít bài có đủ nhãn — chưa chấm được.")
    counts = np.zeros((len(full), len(LABELS)), dtype=int)
    for j, lab in enumerate(LABELS):
        counts[:, j] = (full == lab).sum(axis=1).to_numpy()
    kappa = fleiss_kappa(counts)
    exact = float((full.nunique(axis=1) == 1).mean())
    print(f"\n-- Mức đồng thuận --")
    print(f"  Fleiss' kappa:              {kappa:.4f}")
    print(f"  Cả {len(raters)} người trùng nhau: {100 * exact:.1f}% số bài")
    print("  Cohen's kappa từng cặp:")
    pairs = []
    for a, b in itertools.combinations(raters, 2):
        k = cohen_kappa(wide[a], wide[b])
        pairs.append({"cap": f"{a}–{b}", "cohen_kappa": k})
        print(f"    {a:<6} – {b:<6} {k:.4f}")

    # --- nhãn chốt ----------------------------------------------------------
    voted = wide.apply(majority_vote, axis=1, result_type="expand")
    voted.columns = ["mood_chot", "so_phieu", "tranh_chap"]
    gold = wide.join(voted)
    n_chot = int(gold.mood_chot.ne("").sum())
    print(f"\n-- Nhãn chốt --")
    print(f"  Chốt được (≥2 phiếu): {n_chot}/{len(gold)}")
    print(f"  Tranh chấp:           {int(gold.tranh_chap.sum())}")
    print("  Phân bố nhãn chốt:")
    for lab, n in gold[gold.mood_chot.ne("")].mood_chot.value_counts().items():
        print(f"    {lab:<12} {n}")

    # --- so với nhãn máy ----------------------------------------------------
    if not KEY_CSV.exists():
        raise SystemExit(f"\nKhông thấy {KEY_CSV.name} — không so được với nhãn máy.")
    key = pd.read_csv(KEY_CSV, dtype={"item_id": str, "track_id": str})
    gold = gold.reset_index().merge(key, on="item_id", how="left")

    # `mood_pred` là bản sao của `mood_label` ở bài có mood_source == "tag" (26.752 bài),
    # nên so cả cột sẽ nhân đôi dòng T2. Chỉ tính T4 ở chỗ mô hình thật sự đoán.
    if "mood_source" in gold:
        gold["mood_pred_model"] = gold["mood_pred"].where(gold["mood_source"].eq("model"))
    else:
        gold["mood_pred_model"] = gold["mood_pred"]
        print("  [!] thiếu cột mood_source — dòng T4 có thể lẫn nhãn tag")
    rows = [compare(gold, "mood_label", "nhãn tag T2"),
            compare(gold, "mood_pred_model", "T4 mô hình thật đoán")]
    for path, label in [
        # Hai file "gold" chạy trên đúng 180 bài này -> n đầy đủ. Ba file dưới là các
        # lần chạy L1 cũ trên bài khác, chỉ trùng vài bài nên n rất nhỏ.
        ("llm_labels_openai_compat_gpt-oss-120b_llm_ten_gold.csv",
         "LLM trên đúng 180 bài (tên + nghệ sĩ)"),
        ("llm_labels_openai_compat_gpt-oss-120b_llm_ten_tag_gold.csv",
         "LLM trên đúng 180 bài (tên + nghệ sĩ + tag)"),
        ("llm_labels_openai_compat_gpt-oss-120b_llm_ten_calib.csv", "LLM (tên + nghệ sĩ)"),
        ("llm_labels_openai_compat_gpt-oss-120b_llm_ten_tag_calib.csv", "LLM (tên + nghệ sĩ + tag)"),
        ("llm_labels_openai_compat_gpt-oss-120b_llm_ten_expand.csv", "LLM (mở rộng, tên + nghệ sĩ)"),
    ]:
        p = PROCESSED_DIR / path
        if not p.exists():
            continue
        llm = pd.read_csv(p, dtype=str)[["track_id", "llm_nhom"]]
        merged = gold.merge(llm, on="track_id", how="left")
        rows.append(compare(merged, "llm_nhom", label))

    report = pd.DataFrame(rows)
    print(f"\n-- Khớp với nhãn chốt của người (%) --")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("\n  Ngẫu nhiên = 20% (5 nhóm). Mẫu cân bằng theo thiết kế, "
          "không phải phân bố thật của catalogue.")

    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["item_id"] + raters + ["mood_chot", "so_phieu", "tranh_chap",
                                   "label_status", "mood_label", "mood_pred",
                                   "mood_source", "track_id", "name", "artist"]
    gold[[c for c in cols if c in gold.columns]].to_csv(
        ANNOTATION_DIR / "mood_gold.csv", index=False)
    meta = pd.DataFrame([{"chi_so": "fleiss_kappa", "gia_tri": kappa},
                         {"chi_so": "trung_hoan_toan_pct", "gia_tri": 100 * exact},
                         {"chi_so": "n_bai_du_nhan", "gia_tri": len(full)},
                         {"chi_so": "n_chot_duoc", "gia_tri": n_chot},
                         {"chi_so": "n_tranh_chap", "gia_tri": int(gold.tranh_chap.sum())}]
                        + [{"chi_so": f"cohen_{p['cap']}", "gia_tri": p["cohen_kappa"]}
                           for p in pairs])
    pd.concat([meta, pd.DataFrame([{}]), report], ignore_index=True).to_csv(
        ANNOTATION_DIR / "agreement_report.csv", index=False)
    print(f"\nĐã ghi: annotation/mood_gold.csv · annotation/agreement_report.csv")


if __name__ == "__main__":
    main()
