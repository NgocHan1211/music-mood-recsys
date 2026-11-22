"""Hồ sơ tâm trạng cho MỌI bài hát (task T4 / M3).

Bài có nhãn từ tag: trộn nhãn tag với dự đoán mô hình, `mood_source = tag`.
Bài còn lại: hoàn toàn từ mô hình, `mood_source = model`.

Xuất: data/processed/track_mood.parquet (file TV2 dùng)
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from .config import GROUPS, MODELS_DIR, PROCESSED_DIR, load_config
from .features import build_features
from .io import load_tracks
from .labeling import LABELS_PARQUET, SOFT_COLS

PROB_COLS = [f"p_{g}" for g in GROUPS]
TRACK_MOOD_PARQUET = PROCESSED_DIR / "track_mood.parquet"
MODEL_PATH = MODELS_DIR / "mood_classifier.joblib"


def load_model(path=None) -> dict:
    """Đọc mô hình đã lưu ở T3 (dict: model, features, groups, ...)."""
    bundle = joblib.load(path or MODEL_PATH)
    if bundle["groups"] != GROUPS:
        raise ValueError("Mô hình lưu nhóm khác GROUPS")
    return bundle


def predict_proba(df: pd.DataFrame, bundle: dict, cfg: dict | None = None) -> np.ndarray:
    """Xác suất 5 nhóm cho từng bài, cột theo thứ tự GROUPS."""
    X = build_features(df, cfg)[bundle["features"]]
    model = bundle["model"]
    proba = model.predict_proba(X)
    order = [list(model.classes_).index(g) for g in GROUPS]
    return proba[:, order]


def blend(tag_soft: np.ndarray, model_proba: np.ndarray, alpha: float) -> np.ndarray:
    """Trộn nhãn tag và dự đoán mô hình cho các bài có nhãn tag.

    alpha = trọng số của nhãn tag (nhãn người nghe), 1 - alpha cho mô hình.
    """
    out = alpha * tag_soft + (1 - alpha) * model_proba
    return out / out.sum(axis=1, keepdims=True)


def build_track_mood(df: pd.DataFrame, labels: pd.DataFrame, bundle: dict,
                     cfg: dict | None = None) -> pd.DataFrame:
    """Hồ sơ tâm trạng cho mọi bài, đúng hợp đồng data_contracts.md."""
    cfg = cfg or load_config()
    alpha = float(cfg["predict"]["tag_weight"])

    merged = df.merge(labels[["track_id"] + SOFT_COLS + ["label_status"]],
                      on="track_id", how="left")
    model_proba = predict_proba(merged, bundle, cfg)

    tag_soft = merged[SOFT_COLS].to_numpy(dtype=float)
    # Chỉ tin nhãn tag khi nó đủ sạch: labeled (không conflict, không suspect)
    use_tag = (merged["label_status"] == "labeled").to_numpy() & ~np.isnan(tag_soft).any(axis=1)

    proba = model_proba.copy()
    if use_tag.any():
        proba[use_tag] = blend(tag_soft[use_tag], model_proba[use_tag], alpha)

    out = pd.DataFrame({"track_id": merged["track_id"].to_numpy()})
    for i, g in enumerate(GROUPS):
        out[f"p_{g}"] = proba[:, i]
    out["mood_pred"] = np.array(GROUPS)[proba.argmax(axis=1)]
    out["mood_confidence"] = proba.max(axis=1)
    out["mood_source"] = np.where(use_tag, "tag", "model")
    out["valence"] = merged["valence"].to_numpy()
    out["energy"] = merged["energy"].to_numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Dự đoán hồ sơ tâm trạng cho mọi bài.")
    parser.add_argument("--sample", action="store_true", help="chỉ chạy trên bản mẫu")
    args = parser.parse_args()

    cfg = load_config()
    df = load_tracks(sample=args.sample)
    labels = pd.read_parquet(LABELS_PARQUET)
    bundle = load_model()
    print(f"Mô hình: {bundle['model_name']} (hiệu chỉnh: {bundle.get('calibration')}, "
          f"huấn luyện trên {bundle['n_train']:,} bài)")

    out = build_track_mood(df, labels, bundle, cfg)

    # Kiểm tra hợp đồng
    assert len(out) == len(df), "mỗi bài phải có đúng 1 dòng"
    assert not out["track_id"].duplicated().any()
    assert np.allclose(out[PROB_COLS].sum(axis=1), 1.0), "tổng xác suất phải = 1"

    print(f"\nSố bài: {len(out):,}")
    print("\n-- mood_source --")
    print(out["mood_source"].value_counts().to_string())
    print("\n-- mood_pred --")
    vc = out["mood_pred"].value_counts().reindex(GROUPS)
    for g in GROUPS:
        print(f"  {g:<12} {int(vc[g]):>7,} ({vc[g]/len(out)*100:5.2f}%)")
    print(f"\nĐộ tự tin trung bình: {out['mood_confidence'].mean():.3f} "
          f"(tag: {out.loc[out.mood_source=='tag','mood_confidence'].mean():.3f}, "
          f"model: {out.loc[out.mood_source=='model','mood_confidence'].mean():.3f})")

    if not args.sample:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(TRACK_MOOD_PARQUET, index=False)
        print(f"\nĐã lưu: {TRACK_MOOD_PARQUET}")


if __name__ == "__main__":
    main()
