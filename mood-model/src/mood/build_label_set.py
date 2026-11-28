"""Dựng bộ nhãn tâm trạng do người gán — mẫu phân tầng, che hết nhãn máy.

Vì sao cần: toàn bộ nhãn tâm trạng hiện tại là **nhãn yếu** suy từ tag Last.fm theo
bảng trọng số nhóm tự đặt. Mọi so sánh T2 / embedding (A1) / LLM (L1) đều chấm bằng
chính hệ nhãn đó nên có tính vòng lặp. Một bộ nhãn do người **nghe** rồi gán, giữ
ngoài mọi bước chỉnh bảng tag, là cách duy nhất phá vòng lặp đó.
Xem `reports/RESULTS.md` mục 26 và `docs/huong_dan_gan_nhan_mood.md`.

Thiết kế mẫu: phân tầng theo `label_status` × `mood_pred` (3 × 5 = 15 ô), lấy đều mỗi
ô, chỉ trong catalogue chính thức 22.557 bài để bộ nhãn dùng được cho cả bộ phân loại
lẫn phần đánh giá gợi ý. Thứ tự hàng bị trộn để không đoán được tầng.

Sinh ra:
  annotation/mood_gold_blind.csv          -> file người gán nhãn dùng (KHÔNG có nhãn máy)
  annotation/nhan_<ten>.csv               -> bản trống cho từng người
  annotation/KEY_KHONG_MO_TRUOC_KHI_GAN.csv -> đáp án máy, chỉ mở sau khi gán xong

Chạy:
    python -m src.mood.build_label_set --n-per-cell 12 --labellers nhu,han,thy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GROUPS, PROCESSED_DIR, SEED
from .io import RAW_DIR

MOOD_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = MOOD_ROOT / "annotation"
STATUSES = ["labeled", "conflict", "no_emotion_tag"]

# Cột người gán nhãn phải điền. Để trống sẵn trong file.
ANSWER_COLS = ["mood_chinh", "mood_phu", "do_tu_tin", "nghe_duoc", "ghi_chu"]


def load_pool(train_csv: Path) -> pd.DataFrame:
    """Gộp hồ sơ tâm trạng + trạng thái nhãn + link nghe, giới hạn catalogue chính thức."""
    prof = pd.read_parquet(PROCESSED_DIR / "track_mood.parquet")
    lab = pd.read_parquet(PROCESSED_DIR / "track_mood_labels.parquet")
    mi = pd.read_csv(RAW_DIR / "Music_Info.csv")
    mi.columns = [c.lstrip("﻿") for c in mi.columns]

    cat = set(pd.read_csv(train_csv, usecols=["track_id"], dtype=str).track_id.unique())
    pool = (prof.merge(lab[["track_id", "mood_label", "label_status"]], on="track_id")
                .merge(mi[["track_id", "name", "artist", "tags",
                           "spotify_id", "spotify_preview_url"]], on="track_id"))
    pool = pool[pool.track_id.isin(cat)]
    pool = pool[pool.label_status.isin(STATUSES)]
    pool = pool[pool.spotify_preview_url.notna() & pool.name.notna()]
    return pool.reset_index(drop=True)


def stratified_sample(pool: pd.DataFrame, n_per_cell: int,
                      seed: int = SEED) -> pd.DataFrame:
    """Lấy `n_per_cell` bài mỗi ô (label_status × mood_pred)."""
    picked = []
    for status in STATUSES:
        for group in GROUPS:
            cell = pool[(pool.label_status == status) & (pool.mood_pred == group)]
            if len(cell) < n_per_cell:
                raise ValueError(
                    f"Ô ({status}, {group}) chỉ có {len(cell)} bài, cần {n_per_cell}")
            picked.append(cell.sample(n_per_cell, random_state=seed))
    out = pd.concat(picked, ignore_index=True)
    # Trộn thứ tự để người gán nhãn không đoán được tầng
    out = out.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    out.insert(0, "item_id", [f"M{i:03d}" for i in range(1, len(out) + 1)])
    return out


def write_files(sample: pd.DataFrame, labellers: list[str]) -> list[Path]:
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    # 1. File mù: chỉ thứ tự, link nghe, tên bài. KHÔNG có tag, nhãn, dự đoán.
    blind = pd.DataFrame({
        "item_id": sample.item_id,
        "link_nghe_30s": sample.spotify_preview_url,
        "link_spotify": "https://open.spotify.com/track/" + sample.spotify_id.astype(str),
        "ten_bai": sample.name,
        "nghe_si": sample.artist,
    })
    for col in ANSWER_COLS:
        blind[col] = ""
    path = ANNOTATION_DIR / "mood_gold_blind.csv"
    blind.to_csv(path, index=False)
    written.append(path)

    # 2. Bản trống cho từng người
    for who in labellers:
        p = ANNOTATION_DIR / f"nhan_{who}.csv"
        if p.exists():
            print(f"  (bỏ qua {p.name} — đã có, không ghi đè)")
            continue
        blind.to_csv(p, index=False)
        written.append(p)

    # 3. Đáp án máy — chỉ mở sau khi cả ba gán xong
    key = sample[["item_id", "track_id", "label_status", "mood_label", "mood_pred",
                  "mood_confidence", "mood_source", "valence", "energy", "tags",
                  "name", "artist"]].copy()
    path = ANNOTATION_DIR / "KEY_KHONG_MO_TRUOC_KHI_GAN.csv"
    key.to_csv(path, index=False)
    written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dựng bộ nhãn tâm trạng do người gán (mẫu phân tầng, che nhãn máy).")
    parser.add_argument("--train", type=Path, default=MOOD_ROOT.parent / "data" / "train.csv",
                        help="dùng để lấy catalogue chính thức")
    parser.add_argument("--n-per-cell", type=int, default=12,
                        help="số bài mỗi ô (3 trạng thái × 5 nhóm). 12 -> 180 bài")
    parser.add_argument("--labellers", default="nhu,han,thy",
                        help="tên người gán nhãn, phân tách bằng dấu phẩy")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    labellers = [s.strip() for s in args.labellers.split(",") if s.strip()]
    if len(labellers) < 2:
        parser.error("cần ít nhất 2 người gán nhãn để đo mức đồng thuận")

    pool = load_pool(args.train)
    print(f"Kho chọn mẫu: {len(pool):,} bài (catalogue chính thức, có link nghe)")
    sample = stratified_sample(pool, args.n_per_cell, args.seed)
    n = len(sample)
    print(f"Mẫu: {n} bài = {args.n_per_cell} bài × 3 trạng thái × 5 nhóm")
    print(f"Thời lượng nghe: ~{n * 30 / 60:.0f} phút audio/người "
          f"(ước cả cân nhắc: ~{n * 50 / 3600:.1f} giờ)")
    print("\nPhân bố mẫu (kiểm tra cân bằng):")
    print(pd.crosstab(sample.label_status, sample.mood_pred, margins=True).to_string())

    print("\nĐã ghi:")
    for p in write_files(sample, labellers):
        print(f"  {p.relative_to(MOOD_ROOT)}")
    print(f"\nHướng dẫn gán nhãn: docs/huong_dan_gan_nhan_mood.md")
    print("Sau khi cả nhóm gán xong: python -m src.mood.score_agreement")


if __name__ == "__main__":
    main()
