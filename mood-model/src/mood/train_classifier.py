"""Huấn luyện và so sánh bộ phân loại tâm trạng từ đặc trưng âm thanh (task T3 / M3).

4 mô hình: baseline_ve (khoảng cách tới tọa độ tự đặt), logreg_ve, logreg_full, gbm_full.
Đánh giá bằng GroupKFold 5 fold theo `artist` để cùng một nghệ sĩ không nằm ở cả
train và test.

Xuất: models/mood_classifier.joblib, reports/figures/*.png,
      data/processed/model_comparison.csv, data/processed/oof_proba.parquet
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FIGURES_DIR, GROUPS, MODELS_DIR, PROCESSED_DIR, SEED, load_config
from .features import build_features, feature_columns, training_set
from .io import load_tracks
from .labeling import LABELS_PARQUET

VE_FEATURES = ["valence", "energy"]


class ManualCentroidClassifier(BaseEstimator, ClassifierMixin):
    """Bản baseline cần cải tiến: gán nhóm theo khoảng cách tới tọa độ tự đặt tay.

    Xác suất = softmax của -d²/(2σ²) với d là khoảng cách Euclid trên (valence, energy),
    để so sánh log-loss được với các mô hình khác.
    """

    def __init__(self, centroids: dict[str, list[float]], sigma: float = 0.15):
        self.centroids = centroids
        self.sigma = sigma

    def fit(self, X, y=None):
        # Không học gì từ dữ liệu — đó chính là điểm yếu đang muốn chứng minh
        self.classes_ = np.array(sorted(self.centroids))
        self._c = np.array([self.centroids[g] for g in self.classes_])
        return self

    def predict_proba(self, X):
        Xv = np.asarray(X, dtype=float)[:, :2]
        d2 = ((Xv[:, None, :] - self._c[None, :, :]) ** 2).sum(axis=2)
        logits = -d2 / (2 * self.sigma ** 2)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def make_models(cfg: dict) -> dict[str, tuple[object, list[str]]]:
    """Bốn mô hình cần so sánh, kèm danh sách cột đặc trưng mỗi mô hình dùng."""
    full = feature_columns(cfg)
    models: dict[str, tuple[object, list[str]]] = {
        "baseline_ve": (
            ManualCentroidClassifier(cfg["manual_centroids"], cfg["scoring"]["sigma"]),
            VE_FEATURES,
        ),
        "logreg_ve": (
            Pipeline([("sc", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                 random_state=SEED))]),
            VE_FEATURES,
        ),
        "logreg_full": (
            Pipeline([("sc", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                 random_state=SEED))]),
            full,
        ),
    }
    models["gbm_full"] = (_make_gbm(), full)
    return models


def _make_gbm():
    """LightGBM nếu dùng được, nếu không thì HistGradientBoostingClassifier."""
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1,
        )
    except Exception:  # pragma: no cover - chỉ chạy khi thiếu libomp
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, class_weight="balanced", random_state=SEED
        )


def cross_val_proba(model, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                    n_splits: int) -> np.ndarray:
    """Xác suất out-of-fold với GroupKFold theo artist."""
    gkf = GroupKFold(n_splits=n_splits)
    proba = np.zeros((len(X), len(GROUPS)))
    classes = np.array(GROUPS)
    for train_idx, test_idx in gkf.split(X, y, groups):
        m = _clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        p = m.predict_proba(X.iloc[test_idx])
        # xếp lại cột theo thứ tự GROUPS
        order = [list(m.classes_).index(g) for g in classes]
        proba[test_idx] = p[:, order]
    return proba


def _clone(model):
    from sklearn.base import clone
    return clone(model)


def score_table(y: pd.Series, proba: np.ndarray) -> dict:
    """macro-F1, accuracy, log-loss và F1 từng nhóm."""
    pred = np.array(GROUPS)[proba.argmax(axis=1)]
    # log_loss của sklearn sắp nhãn theo thứ tự chữ cái -> phải xếp lại cột xác suất
    # cho khớp, nếu không log-loss bị tính sai hoàn toàn.
    order = np.argsort(GROUPS)
    row = {
        "macro_f1": f1_score(y, pred, average="macro"),
        "accuracy": accuracy_score(y, pred),
        "log_loss": log_loss(y, proba[:, order], labels=sorted(GROUPS)),
    }
    per = f1_score(y, pred, average=None, labels=GROUPS)
    for g, v in zip(GROUPS, per):
        row[f"f1_{g}"] = v
    return row


def expected_calibration_error(y: pd.Series, proba: np.ndarray, n_bins: int = 10) -> float:
    """ECE trên xác suất của nhóm được dự đoán."""
    conf = proba.max(axis=1)
    pred = np.array(GROUPS)[proba.argmax(axis=1)]
    correct = (pred == y.to_numpy()).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def plot_confusion(y: pd.Series, proba: np.ndarray, name: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = np.array(GROUPS)[proba.argmax(axis=1)]
    cm = confusion_matrix(y, pred, labels=GROUPS, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(GROUPS)), GROUPS, rotation=20)
    ax.set_yticks(range(len(GROUPS)), GROUPS)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Nhãn từ tag")
    ax.set_title(f"Ma trận nhầm lẫn — {name} (chuẩn hóa theo hàng)")
    for i in range(len(GROUPS)):
        for j in range(len(GROUPS)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = FIGURES_DIR / f"confusion_{name}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return str(path)


def make_calibrated(model, method: str):
    """Bọc mô hình bằng CalibratedClassifierCV, hoặc trả về chính nó nếu method='none'."""
    if method == "none":
        return _clone(model)
    return CalibratedClassifierCV(_clone(model), method=method, cv=3)


def plot_reliability(y: pd.Series, before: np.ndarray, after: np.ndarray,
                     label_a: str = "trước hiệu chỉnh",
                     label_b: str = "sau hiệu chỉnh") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="hoàn hảo")
    for proba, label in [(before, label_a), (after, label_b)]:
        conf = proba.max(axis=1)
        pred = np.array(GROUPS)[proba.argmax(axis=1)]
        correct = (pred == y.to_numpy()).astype(float)
        bins = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() > 20:
                xs.append(conf[m].mean()); ys.append(correct[m].mean())
        ax.plot(xs, ys, "o-", label=f"{label} (ECE={expected_calibration_error(y, proba):.3f})")
    ax.set_xlabel("Xác suất mô hình đưa ra"); ax.set_ylabel("Tỷ lệ đúng thực tế")
    ax.set_title("Reliability diagram (out-of-fold)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "reliability.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return str(path)


def plot_importance(imp: pd.DataFrame) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = imp.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color="#4C72B0")
    ax.set_xlabel("Mức giảm macro-F1 khi xáo trộn đặc trưng")
    ax.set_title("Permutation importance — mô hình tốt nhất")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "permutation_importance.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện bộ phân loại tâm trạng.")
    parser.add_argument("--sample", action="store_true", help="chỉ chạy trên bản mẫu")
    args = parser.parse_args()

    cfg = load_config()
    df = load_tracks(sample=args.sample)
    labels = pd.read_parquet(LABELS_PARQUET)
    X, y, groups, track_ids = training_set(df, labels, cfg)
    n_splits = int(cfg["classifier"]["n_splits"])

    print(f"Tập huấn luyện: {len(X):,} bài có nhãn | {groups.nunique():,} nghệ sĩ | "
          f"{len(feature_columns(cfg))} đặc trưng")
    print("Phân bố nhãn:", y.value_counts().to_dict())
    print(f"Chia fold: GroupKFold {n_splits} fold theo artist\n")

    # 1. So sánh 4 mô hình
    rows, oof = [], {}
    for name, (model, cols) in make_models(cfg).items():
        proba = cross_val_proba(model, X[cols], y, groups, n_splits)
        oof[name] = proba
        row = {"model": name, "n_features": len(cols)}
        row.update(score_table(y, proba))
        rows.append(row)
        print(f"  {name:<12} macro-F1={row['macro_f1']:.3f}  acc={row['accuracy']:.3f}  "
              f"log-loss={row['log_loss']:.3f}")

    comp = pd.DataFrame(rows).set_index("model")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    comp.to_csv(PROCESSED_DIR / "model_comparison.csv")
    print("\n-- Bảng so sánh --")
    print(comp.to_string(float_format="{:.3f}".format))

    best = comp["macro_f1"].idxmax()
    print(f"\nMô hình tốt nhất: {best} (macro-F1 = {comp.loc[best, 'macro_f1']:.3f})")
    if comp.loc[best, "macro_f1"] < 0.3:
        print("!! CẢNH BÁO: macro-F1 < 0.3 — cần dừng và báo người dùng")

    print(f"\n-- Báo cáo chi tiết: {best} --")
    pred = np.array(GROUPS)[oof[best].argmax(axis=1)]
    print(classification_report(y, pred, labels=GROUPS, digits=3, zero_division=0))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Hình:", plot_confusion(y, oof[best], best))

    # 2. Hiệu chỉnh xác suất: so sánh các phương án trong config
    base_model, base_cols = make_models(cfg)[best]
    chosen = cfg["classifier"]["calibration"]
    cal_rows, cal_proba = [], {}
    for method in cfg["classifier"]["calibration_candidates"]:
        proba_m = (oof[best] if method == "none" else
                   cross_val_proba(make_calibrated(base_model, method),
                                   X[base_cols], y, groups, n_splits))
        cal_proba[method] = proba_m
        row = {"calibration": method, "ECE": expected_calibration_error(y, proba_m)}
        row.update(score_table(y, proba_m))
        cal_rows.append(row)
        print(f"  {method:<9} macro-F1={row['macro_f1']:.3f} acc={row['accuracy']:.3f} "
              f"log-loss={row['log_loss']:.3f} ECE={row['ECE']:.4f} "
              f"f1_buon={row['f1_buon']:.3f}")

    cal = pd.DataFrame(cal_rows).set_index("calibration")
    cal.to_csv(PROCESSED_DIR / "calibration_comparison.csv")
    print(f"\n-- Hiệu chỉnh xác suất (đã chọn: {chosen}) --")
    print(cal[["macro_f1", "accuracy", "log_loss", "ECE"] +
              [f"f1_{g}" for g in GROUPS]].to_string(float_format="{:.3f}".format))

    other = next((m for m in cal_proba if m != chosen), None)
    if other:
        print("Hình:", plot_reliability(y, cal_proba[chosen], cal_proba[other],
                                        label_a=f"đã chọn: {chosen}", label_b=other))

    proba_cal = cal_proba[chosen]
    oof_df = pd.DataFrame({"track_id": track_ids, "mood_label": y.to_numpy()})
    for i, g in enumerate(GROUPS):
        oof_df[f"p_{g}"] = proba_cal[:, i]
    oof_df.to_parquet(PROCESSED_DIR / "oof_proba.parquet", index=False)

    # 3. Permutation importance (trên fold cuối, đo bằng macro-F1)
    gkf = GroupKFold(n_splits=n_splits)
    tr, te = next(iter(gkf.split(X, y, groups)))
    m = _clone(base_model).fit(X.iloc[tr][base_cols], y.iloc[tr])
    r = permutation_importance(m, X.iloc[te][base_cols], y.iloc[te], n_repeats=5,
                               random_state=SEED, scoring="f1_macro", n_jobs=-1)
    imp = pd.DataFrame({"feature": base_cols,
                        "importance_mean": r.importances_mean,
                        "importance_std": r.importances_std}
                       ).sort_values("importance_mean", ascending=False)
    print("\n-- Permutation importance (giảm macro-F1) --")
    print(imp.to_string(index=False, float_format="{:.4f}".format))
    imp.to_csv(PROCESSED_DIR / "permutation_importance.csv", index=False)
    print("Hình:", plot_importance(imp))

    # 4. Mô hình cuối: huấn luyện trên toàn bộ bài có nhãn
    final = make_calibrated(base_model, chosen)
    final.fit(X[base_cols], y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    path = MODELS_DIR / "mood_classifier.joblib"
    joblib.dump({"model": final, "features": base_cols, "groups": GROUPS,
                 "model_name": best, "calibration": chosen, "n_train": len(X)}, path)
    print(f"\nĐã lưu mô hình cuối: {path}")


if __name__ == "__main__":
    main()
