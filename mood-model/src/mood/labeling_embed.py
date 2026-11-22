"""A1 — Gán nhãn tag bằng embedding văn bản thay cho bảng trọng số chỉnh tay.

Cách làm lấy từ Hausberger, Jósár & Schedl (NLP4MusA 2026): họ gán 32 ý định nghe
cho bài hát mà không có nhãn nào, bằng cách nhúng TÊN playlist và TÊN ý định rồi so
cosine, sau đó **chuẩn hóa z-score theo từng ý định trên toàn bộ bài hát** trước khi
chọn giá trị cao nhất.

Áp sang đây: nhúng TÊN TAG và TÊN NHÓM TÂM TRẠNG.

    1. cos(tag, nhóm) cho 100 tag × 5 nhóm
    2. z-score theo CỘT (từng nhóm) trên toàn bộ tag  -> xóa độ lệch thang đo giữa
       các nhóm, chính là chỗ bảng trọng số tay phải chỉnh bù cho nhóm `buon`
    3. trọng số = relu(z) chuẩn hóa tổng 1; chỉ giữ tag có max z >= z_min

Module này KHÔNG ghi đè nhãn chính thức của T2. Nó sinh bảng trọng số mới, gán nhãn
song song, rồi so sánh: độ phủ từng nhóm, mức trùng khớp với bảng tay, và macro-F1
của cùng bộ phân loại T3 huấn luyện trên nhãn mới.

Chạy:  python -m src.mood.labeling_embed [--z-min 1.0] [--sample]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import api
from .config import GROUPS, PROCESSED_DIR, SEED, load_config
from .io import load_tracks
from .labeling import SOFT_COLS, build_tag_weights, make_labels

# Bài báo dùng google/embeddinggemma-300m nhưng repo đó bị khóa trên HuggingFace
# (GatedRepoError 401 khi tải). Trong 3 mô hình mở đã thử, bge-base-en-v1.5 khớp vị trí
# valence/energy thật của tag tốt nhất (50% so với 32% của MiniLM) -> chọn làm mặc định.
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
WEIGHTS_CSV = PROCESSED_DIR / "tag_weights_embed.csv"
COMPARE_CSV = PROCESSED_DIR / "labeling_compare.csv"
SIMILARITY_CSV = PROCESSED_DIR / "tag_group_similarity.csv"
AUDIO_POS_CSV = PROCESSED_DIR / "tag_audio_position.csv"

# Mô tả tiếng Anh cho 5 nhóm (tag trong dữ liệu là tiếng Anh nên phải nhúng cùng ngôn ngữ).
# Đây là lựa chọn thiết kế duy nhất còn lại của phương pháp — ghi vào docs/decisions.md.
GROUP_TEXT = {
    "vui":        "happy cheerful joyful upbeat feel good music",
    "buon":       "sad melancholic sorrowful heartbroken depressing music",
    "thu_gian":   "relaxing calm peaceful mellow soothing music",
    "nang_dong":  "energetic lively danceable high energy party music",
    "cang_thang": "tense aggressive angry violent intense music",
}


def tag_vocabulary(df: pd.DataFrame) -> list[str]:
    """Toàn bộ tag thật có trong dữ liệu, xếp theo số bài giảm dần."""
    counts: dict[str, int] = {}
    for tag_list in df["tags_list"]:
        for t in tag_list:
            counts[t] = counts.get(t, 0) + 1
    return [t for t, _ in sorted(counts.items(), key=lambda p: (-p[1], p[0]))]


def embed(texts: list[str]) -> np.ndarray:
    """Vector đơn vị cho từng chuỗi (SentenceTransformer, đã chuẩn hóa L2)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    return np.asarray(model.encode(texts, normalize_embeddings=True,
                                   show_progress_bar=False), dtype=float)


def similarity_table(tags: list[str], refresh: bool = False) -> pd.DataFrame:
    """cos(tag, nhóm) rồi z-score theo từng nhóm trên toàn bộ tag.

    Kết quả được cache ra `tag_group_similarity.csv`. Lần chạy sau đọc cache và KHÔNG
    nạp sentence-transformers: torch và LightGBM mang hai bản OpenMP khác nhau, nạp cả
    hai trong một tiến trình làm segfault trên macOS (đã gặp: exit 139).
    """
    if not refresh and SIMILARITY_CSV.exists():
        cached = pd.read_csv(SIMILARITY_CSV, index_col=0)
        if set(tags).issubset(cached.index):
            print(f"(dùng cache {SIMILARITY_CSV.name}, không nạp lại mô hình nhúng)")
            return cached.loc[tags]

    # Tag trong dữ liệu dùng gạch dưới (vd. indie_rock, death_metal) -> trả về khoảng trắng
    tag_vecs = embed([t.replace("_", " ") for t in tags])
    group_vecs = embed([GROUP_TEXT[g] for g in GROUPS])

    cos = tag_vecs @ group_vecs.T
    z = (cos - cos.mean(axis=0)) / cos.std(axis=0, ddof=0)
    out = pd.DataFrame(z, index=tags, columns=[f"z_{g}" for g in GROUPS])
    for i, g in enumerate(GROUPS):
        out[f"cos_{g}"] = cos[:, i]
    out.index.name = "tag"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.round(6).to_csv(SIMILARITY_CSV)
    return out


def weights_from_z(table: pd.DataFrame, z_min: float,
                   mode: str = "argmax") -> dict[str, dict[str, float]]:
    """Bảng trọng số tag -> nhóm, chỉ giữ tag có max z >= z_min.

    mode="argmax": mỗi tag về đúng một nhóm (z cao nhất), trọng số 1.0 — đúng cách
        bài NLP4MusA gán ý định ("the listening intent with the highest score is assigned").
    mode="soft":   relu(z) chuẩn hóa tổng 1, một tag có thể trỏ nhiều nhóm như bảng tay.
    """
    if mode not in ("argmax", "soft"):
        raise ValueError(mode)
    zcols = [f"z_{g}" for g in GROUPS]
    weights: dict[str, dict[str, float]] = {}
    for tag, row in table[zcols].iterrows():
        z = row.to_numpy()
        if z.max() < z_min:
            continue
        if mode == "argmax":
            weights[tag] = {GROUPS[int(np.argmax(z))]: 1.0}
            continue
        pos = np.clip(z, 0.0, None)
        if pos.sum() <= 0:
            continue
        w = pos / pos.sum()
        weights[tag] = {g: float(v) for g, v in zip(GROUPS, w) if v > 0.01}
    return weights


def hand_plus_embed(hand: dict[str, dict[str, float]],
                    emb: dict[str, dict[str, float]],
                    support_weight: float) -> dict[str, dict[str, float]]:
    """Giữ nguyên bảng tay, chỉ dùng embedding cho các tag bảng tay CHƯA phủ.

    Mở rộng độ phủ mà không chỉnh tay thêm dòng nào; tag mới nhận trọng số
    `support_weight` như tag thể loại hỗ trợ của T2 vì độ tin cậy thấp hơn tag cảm xúc.
    """
    out = {t: dict(w) for t, w in hand.items()}
    for tag, w in emb.items():
        if tag not in out:
            out[tag] = {g: v * support_weight for g, v in w.items()}
    return out


def audio_match(weights: dict[str, dict[str, float]], pos: pd.DataFrame) -> float:
    """Tỷ lệ tag mà nhóm được gán trùng nhóm gần nhất theo vị trí (valence, energy) thật.

    Thước khách quan, không phụ thuộc bảng tay: mỗi tag có một vị trí trung bình trên
    mặt phẳng valence x energy; nhóm "đúng" là tâm học được gần nhất theo Mahalanobis.
    """
    hits, n = 0, 0
    for tag, w in weights.items():
        if tag not in pos.index:
            continue
        n += 1
        hits += int(max(w, key=w.get) == pos.loc[tag, "nhom_theo_amthanh"])
    return float(hits / n) if n else np.nan


def tag_audio_position(df: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    """Vị trí (valence, energy) trung bình của từng tag + nhóm gần nhất."""
    centroids = api.load_centroids()["centroids"]
    inv = {g: np.linalg.inv(np.array(centroids[g]["cov"])) for g in GROUPS}
    mu = {g: np.array(centroids[g]["mean"]) for g in GROUPS}

    rows = []
    for t in tags:
        mask = df["tags_list"].apply(lambda L, t=t: t in L)
        v = float(df.loc[mask, "valence"].mean())
        e = float(df.loc[mask, "energy"].mean())
        d = {g: float((np.array([v, e]) - mu[g]) @ inv[g] @ (np.array([v, e]) - mu[g]))
             for g in GROUPS}
        rows.append({"tag": t, "n_tracks": int(mask.sum()), "valence": v, "energy": e,
                     "nhom_theo_amthanh": min(d, key=d.get)})
    return pd.DataFrame(rows).set_index("tag")


def label_coverage(labels: pd.DataFrame) -> pd.Series:
    """Số bài mỗi nhóm + số bài theo trạng thái nhãn."""
    out = labels["mood_label"].value_counts().reindex(GROUPS).fillna(0).astype(int)
    out["TỔNG có nhãn"] = int((labels["label_status"] == "labeled").sum())
    for st in ("conflict", "no_emotion_tag", "suspect"):
        out[st] = int((labels["label_status"] == st).sum())
    return out


def agreement(hand: pd.DataFrame, embed_labels: pd.DataFrame) -> dict:
    """Mức trùng khớp nhãn cứng giữa hai cách, trên các bài cả hai đều gán được."""
    merged = hand[["track_id", "mood_label"]].merge(
        embed_labels[["track_id", "mood_label"]], on="track_id", suffixes=("_tay", "_embed"))
    both = merged.dropna()
    same = (both["mood_label_tay"] == both["mood_label_embed"]).sum()
    return {"n_both_labeled": len(both),
            "n_agree": int(same),
            "agree_rate": float(same / len(both)) if len(both) else np.nan}


def classifier_macro_f1(df: pd.DataFrame, labels: pd.DataFrame, cfg: dict) -> dict:
    """Huấn luyện lại đúng bộ phân loại tốt nhất của T3 trên nhãn mới."""
    from .features import training_set
    from .train_classifier import _make_gbm, cross_val_proba, score_table

    X, y, groups, _ = training_set(df, labels, cfg)
    proba = cross_val_proba(_make_gbm(), X, y, groups, int(cfg["classifier"]["n_splits"]))
    row = score_table(y, proba)
    row["n_train"] = len(X)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="A1 — gán nhãn tag bằng embedding + z-score.")
    parser.add_argument("--z-min", type=float, default=1.0,
                        help="ngưỡng z để giữ một tag (1.0 = trên trung bình 1 độ lệch chuẩn)")
    parser.add_argument("--sample", action="store_true", help="chạy trên bản mẫu")
    parser.add_argument("--no-classifier", action="store_true", help="bỏ bước huấn luyện lại T3")
    parser.add_argument("--refresh-embed", action="store_true",
                        help="tính lại embedding (nạp sentence-transformers); mặc định đọc cache")
    parser.add_argument("--catalogue", choices=["all", "official"], default="all",
                        help="official = chỉ các bài trong catalogue 10-core của nhóm")
    args = parser.parse_args()

    cfg = load_config()
    df = load_tracks(sample=args.sample)
    tag = ""
    if args.catalogue == "official":
        from .splits import load_part
        keep = set(load_part("train")["track_id"].unique())
        n_before = len(df)
        df = df[df["track_id"].isin(keep)].reset_index(drop=True)
        tag = "_official"
        print(f"Catalogue chính thức của nhóm: {len(df):,} / {n_before:,} bài "
              "(bỏ bài dưới 10 người nghe)")
    tags = tag_vocabulary(df)
    sw = float(cfg["labeling_tags"]["support_weight"])
    print(f"Dữ liệu: {len(df):,} bài | {len(tags)} tag khác nhau")
    print(f"Mô hình nhúng: {EMBED_MODEL}\n")

    table = similarity_table(tags, refresh=args.refresh_embed)
    # Vị trí âm thanh của tag phụ thuộc kho bài -> mỗi catalogue một file riêng
    pos_csv = AUDIO_POS_CSV.with_name(AUDIO_POS_CSV.stem + tag + AUDIO_POS_CSV.suffix)
    if not args.refresh_embed and pos_csv.exists():
        pos = pd.read_csv(pos_csv, index_col=0)
    else:
        pos = tag_audio_position(df, tags)
        pos.round(6).to_csv(pos_csv)

    print("-- 4 tag giống nhất mỗi nhóm theo embedding (z-score) --")
    for g in GROUPS:
        top = table[f"z_{g}"].sort_values(ascending=False).head(4)
        print(f"  {g:<11}", ", ".join(f"{t} ({v:+.1f})" for t, v in top.items()))

    hand_weights = build_tag_weights(cfg)
    hand_labels = make_labels(df, cfg)

    emb_argmax = weights_from_z(table, args.z_min, mode="argmax")
    emb_soft = weights_from_z(table, args.z_min, mode="soft")
    variants = {
        "bảng tay (T2)": hand_weights,
        f"embed argmax z>={args.z_min}": emb_argmax,
        f"embed soft z>={args.z_min}": emb_soft,
        "bảng tay + embed cho tag chưa phủ": hand_plus_embed(hand_weights, emb_argmax, sw),
    }

    rows, covs = [], {}
    for name, w in variants.items():
        labels = hand_labels if name.startswith("bảng tay (") else make_labels(df, cfg, weights=w)
        cov = label_coverage(labels)
        covs[name] = cov
        row = {"method": name, "n_tags_used": len(w),
               "n_labeled": int(cov["TỔNG có nhãn"]),
               "khop_vi_tri_am_thanh": audio_match(w, pos)}
        row.update(agreement(hand_labels, labels))
        for g in GROUPS:
            row[f"n_{g}"] = int(cov[g])
        if not args.no_classifier:
            try:
                sc = classifier_macro_f1(df, labels, cfg)
                row.update({"macro_f1": sc["macro_f1"], "accuracy": sc["accuracy"],
                            "n_train": sc["n_train"]})
                for g in GROUPS:
                    row[f"f1_{g}"] = sc[f"f1_{g}"]
            except Exception as exc:   # nhãn thiếu hẳn một nhóm -> không huấn luyện được
                row["classifier_error"] = f"{type(exc).__name__}: {exc}"
                print(f"  [!] không huấn luyện được bộ phân loại: {row['classifier_error']}")
        rows.append(row)
        print(f"\n{name}: {len(w)} tag | {int(cov['TỔNG có nhãn']):,} bài có nhãn | "
              f"khớp vị trí âm thanh {row['khop_vi_tri_am_thanh']:.1%}"
              + (f" | macro-F1 {row['macro_f1']:.4f}" if "macro_f1" in row else ""))

    print(f"\n{'='*78}\n-- Độ phủ nhãn theo nhóm --")
    print(pd.DataFrame(covs).to_string())

    out = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    compare_csv = COMPARE_CSV.with_name(COMPARE_CSV.stem + tag + COMPARE_CSV.suffix)
    out.to_csv(compare_csv, index=False)
    cols = [c for c in ["method", "n_tags_used", "n_labeled", "khop_vi_tri_am_thanh",
                        "agree_rate", "macro_f1", "accuracy"] if c in out]
    print(f"\n-- Tổng hợp A1 --")
    print(out[cols].to_string(index=False, float_format="{:.4f}".format))

    pd.DataFrame([{"tag": t, **{f"w_{g}": v.get(g, 0.0) for g in GROUPS}}
                  for t, v in sorted(emb_argmax.items())]).to_csv(WEIGHTS_CSV, index=False)
    print(f"\nĐã lưu: {compare_csv}")
    print(f"        {WEIGHTS_CSV}")
    print(f"        {SIMILARITY_CSV}")
    print(f"        {pos_csv}")
    print("\nLưu ý: nhãn chính thức (track_mood_labels.parquet) KHÔNG bị ghi đè.")


if __name__ == "__main__":
    main()
