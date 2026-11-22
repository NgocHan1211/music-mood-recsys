"""G1 — đánh giá trên toàn catalogue: Popularity / ALS / ALS+mood.

Dựng dòng thứ ba của bảng G1 (`ALS + mood`) sao cho **so được trực tiếp** với dòng
`ALS` của TV2. Cách bảo đảm điều đó: không viết lại pipeline, mà nạp và gọi đúng
các hàm của TV2 (`load_processed`, `inner_train_validation`, `make_truth`,
`measure`, `evaluate`) nên cohort, catalogue, tập ứng viên, cách dựng ground truth
và công thức metric đều y nguyên.

Bộ code của TV2 bị tách làm hai chỗ (xem docs/decisions.md 2026-11-22):
  - `music-mood-recsys/src/models.py` + `run_real_taste.py`  (bản mới, Hân push 22/11)
  - `taste-model/src/data.py` + `evaluate.py`                (bản `run_real_taste` cần)
Module này ghép lại dưới tên gói `taste_src` rồi dùng, không sửa dòng nào của TV2.

Giao thức (giữ đúng của TV2, không dùng test để chọn gì):
  1. Ẩn 15% mỗi user từ train -> validation.
  2. Quét `--alphas` trên validation, chọn alpha theo NDCG@10.
  3. Train lại trên toàn bộ train sample, đánh giá **đúng một lần** trên test.

Mục tiêu tâm trạng của mỗi user suy từ **lịch sử thấy được** (nhóm nghe trội nhất,
rồi `user_mood_target` co về tâm chung) — không bao giờ dùng phần bị ẩn.

Chạy:
    python -m src.mood.evaluate_catalogue \
        --train ../data/train.csv --test ../data/test.csv \
        --out reports/results/g1_taste_mood_10k
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import api
from .config import GROUPS
from .predict import PROB_COLS

HERE = Path(__file__).resolve()
MOOD_ROOT = HERE.parents[2]          # .../mood-model
PROJECT_ROOT = MOOD_ROOT.parent      # thư mục chứa mood-model

# Nơi có thể chứa code mô hình gu của TV2, xét theo thứ tự ưu tiên. Hai mục đầu
# phủ cả hai cách bố trí: mood-model nằm cạnh repo nhóm, và mood-model nằm *trong*
# repo nhóm (sau khi push). `taste-model/` chỉ còn là dự phòng cho máy chưa clone
# repo nhóm — từ 2026-11-22 repo nhóm đã tự đủ (commit 26516b5).
TASTE_CANDIDATES = (
    PROJECT_ROOT / "music-mood-recsys" / "src",   # cạnh nhau
    MOOD_ROOT.parent / "src",                     # mood-model nằm trong repo nhóm
    PROJECT_ROOT / "taste-model" / "src",         # dự phòng
)
# Không chỉ cần đúng tên file: repo nhóm từng chứa bản scaffold cũ trong đó
# `evaluate()` trả về một dict key chữ thường, dùng với `run_real_taste.py` sẽ ra
# số sai một cách âm thầm. Nên mỗi file phải chứa đúng hàm đặc trưng của bản dùng được.
REQUIRED = {
    "data.py": "def clean_interactions",
    "evaluate.py": "def ranking_metrics",
    "models.py": "class ALS",
    "run_real_taste.py": "def inner_train_validation",
}


# --------------------------------------------------------------------------- #
# Nạp code của TV2
# --------------------------------------------------------------------------- #
def taste_sources() -> dict[str, Path]:
    """Chọn nguồn **cho từng file** code mô hình gu của TV2.

    Chọn theo từng file thay vì theo cả thư mục, vì bộ code của TV2 từng bị tách:
    `run_real_taste.py` chỉ có trong repo nhóm, còn bản `data.py`/`evaluate.py` dùng
    được thì trước 22/11 chỉ có trong `taste-model/`. Cách này tự chạy đúng ở cả hai
    tình huống: repo nhóm đã cập nhật (lấy hết từ đó) hay chưa (ghép hai nguồn).
    """
    chosen: dict[str, Path] = {}
    report: list[str] = []
    for name, marker in REQUIRED.items():
        for path in TASTE_CANDIDATES:
            target = path / name
            if not target.is_file():
                continue
            if marker in target.read_text(encoding="utf-8", errors="ignore"):
                chosen[name] = target
                break
            report.append(f"{target} — bản cũ, thiếu `{marker}`")
        if name not in chosen:
            report.append(f"{name} — không tìm được bản dùng được")
    if len(chosen) != len(REQUIRED):
        raise FileNotFoundError(
            "Không đủ code mô hình gu dùng được của TV2.\n  "
            + "\n  ".join(report)
            + "\n  Đã tìm trong:\n    " + "\n    ".join(str(p) for p in TASTE_CANDIDATES)
            + "\n  Cập nhật repo nhóm: cd music-mood-recsys && "
              "git pull --ff-only origin main   (cần commit 26516b5, 22/11/2026)")
    return chosen


def load_taste_package() -> types.ModuleType:
    """Tạo gói `taste_src` trỏ vào code của TV2.

    Tên `taste_src` để không trùng gói `src` của mood-model. Trước 22/11 phải ghép
    hai nguồn vì repo nhóm thiếu `data.py`/`evaluate.py`; nay repo đã đủ nên nạp
    tất cả từ một chỗ.
    """
    if "taste_src" in sys.modules:
        return sys.modules["taste_src"]
    sources = taste_sources()
    pkg = types.ModuleType("taste_src")
    pkg.__path__ = [str(sources["models.py"].parent)]
    pkg.__package__ = "taste_src"
    pkg.__taste_sources__ = sources
    sys.modules["taste_src"] = pkg
    return pkg


def _load_from(name: str, path: Path) -> types.ModuleType:
    """Nạp một file .py của TV2 thành `taste_src.<name>`, giữ import tương đối."""
    full = f"taste_src.{name}"
    if full in sys.modules:
        return sys.modules[full]
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy {path}")
    spec = importlib.util.spec_from_file_location(full, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "taste_src"
    sys.modules[full] = module
    spec.loader.exec_module(module)
    setattr(sys.modules["taste_src"], name, module)
    return module


def taste_modules():
    """Trả về (data, evaluate, models, run) của TV2.

    Nạp `data` và `evaluate` trước vì `models.py` import tương đối từ `.data`.
    """
    pkg = load_taste_package()
    src = pkg.__taste_sources__
    data = _load_from("data", src["data.py"])
    evaluate = _load_from("evaluate", src["evaluate.py"])
    models = _load_from("models", src["models.py"])
    run = _load_from("run_real_taste", src["run_real_taste.py"])
    # Chốt lần cuối: `run_real_taste.py` cần `evaluate()` trả về (metrics, detail).
    # Bản scaffold cũ trả về một dict key chữ thường -> sai số liệu một cách âm thầm.
    if not hasattr(evaluate, "ranking_metrics"):
        raise RuntimeError(
            f"{evaluate.__file__} là bản scaffold cũ, không dùng được với "
            "run_real_taste.py (cần evaluate() trả về tuple).")
    # Kiểm tra nạp đúng bản: models.py mới của TV2 có tham số `weighting`.
    if "weighting" not in models.ALS.__init__.__code__.co_varnames:
        raise RuntimeError(
            f"models.py không phải bản mới của TV2 (thiếu `weighting`): {models.__file__}")
    return data, evaluate, models, run


# --------------------------------------------------------------------------- #
# Mục tiêu tâm trạng theo user, suy từ lịch sử thấy được
# --------------------------------------------------------------------------- #
def dominant_group(track_ids, weights, proba_of) -> str:
    """Nhóm tâm trạng trội nhất trong lịch sử nghe: argmax Σ playcount · P(m | bài)."""
    total = np.zeros(len(GROUPS), dtype=float)
    for tid, w in zip(track_ids, weights):
        row = proba_of.get(tid)
        if row is not None:
            total += float(w) * row
    if not total.any():
        return "thu_gian"          # nhóm phủ rộng nhất; chỉ dùng khi không có bằng chứng
    return GROUPS[int(total.argmax())]


def mood_targets_for(visible: pd.DataFrame, users, proba_of) -> dict[str, tuple[float, float]]:
    """user_id -> (valence, energy) mục tiêu, chỉ từ `visible` (phần thấy được)."""
    out: dict[str, tuple[float, float]] = {}
    wanted = set(users)
    grouped = visible[visible.user_id.isin(wanted)].groupby("user_id")
    for user, block in grouped:
        tids = block.track_id.tolist()
        counts = block.playcount.to_numpy(dtype=float)
        code = dominant_group(tids, counts, proba_of)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[user] = api.user_mood_target(tids, code, weights=counts)
    for user in wanted - set(out):
        out[user] = api.quick_mood("thu_gian")
    return out


# --------------------------------------------------------------------------- #
# Mô hình ghép
# --------------------------------------------------------------------------- #
def _normalize(values: np.ndarray) -> np.ndarray:
    """Min-max, giống `taste_src.hybrid.normalize` (đã kiểm: cùng công thức)."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x
    if not np.isfinite(x).all():
        raise ValueError("Điểm không hữu hạn")
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


class TasteMoodHybrid:
    """ALS của TV2 ghép với điểm tâm trạng của TV3.

    Giữ đúng interface mà `run_real_taste.measure` cần (`.items`, `.recommend`), và
    lặp lại **y nguyên** cách ALS chọn ứng viên (catalogue đã fit, trừ lịch sử đã
    thấy) nên tập ứng viên của hai dòng bảng là một.
    """

    def __init__(self, als, mood_target_of: dict[str, tuple[float, float]],
                 alpha: float):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha phải trong [0, 1]")
        self.als = als
        self.items = als.items
        self.item_index = als.item_index
        self.seen = als.seen
        self.mood_target_of = mood_target_of
        self.alpha = float(alpha)
        self._cache: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}

    def candidate_scores(self, user_id: str):
        """(ids, điểm gu thô, điểm tâm trạng thô) — không phụ thuộc alpha nên cache được."""
        hit = self._cache.get(user_id)
        if hit is not None:
            return hit
        if user_id not in self.als.user_index:
            raise ValueError(f"User lạ với ALS: {user_id}")
        vector = self.als.user_factors[self.als.user_index[user_id]]
        seen = self.als.seen.get(user_id, set())
        ids = sorted(set(self.items).intersection(self.item_index) - seen)
        taste = self.als.item_factors[[self.item_index[t] for t in ids]] @ vector
        valence, energy = self.mood_target_of[user_id]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mood = api.mood_scores(ids, valence, energy)
        value = (ids, np.asarray(taste, dtype=float), np.asarray(mood, dtype=float))
        self._cache[user_id] = value
        return value

    def recommend(self, user_id=None, top_k=50, candidate_track_ids=None,
                  seed_counts=None):
        if top_k < 1:
            raise ValueError("top_k phải dương")
        if seed_counts is not None or candidate_track_ids is not None:
            raise NotImplementedError("Chỉ dùng cho user có trong ALS, toàn catalogue")
        ids, taste, mood = self.candidate_scores(user_id)
        score = self.alpha * _normalize(taste) + (1.0 - self.alpha) * _normalize(mood)
        keep = min(top_k, len(ids))
        if keep < len(ids):
            picked = np.argpartition(-score, keep - 1)[:keep]
            pairs = [(ids[i], float(score[i])) for i in picked]
        else:
            pairs = list(zip(ids, map(float, score)))
        return sorted(pairs, key=lambda p: (-p[1], p[0]))

    def ranking(self, user_id: str, top_k: int = 50) -> list[str]:
        return [tid for tid, _ in self.recommend(user_id, top_k=top_k)]


# --------------------------------------------------------------------------- #
# Quét alpha trên validation
# --------------------------------------------------------------------------- #
def sweep_alpha(als, heldout, visible, alphas, eval_users, seed, proba_of,
                evaluate_mod, run_mod):
    """Chọn alpha theo NDCG@10 trên validation. Không chạm tới test."""
    truth, _, users = run_mod.make_truth(heldout, set(als.items), eval_users, seed)
    targets = mood_targets_for(visible, users, proba_of)
    hybrid = TasteMoodHybrid(als, targets, alpha=1.0)
    for user in users:                     # nạp cache một lần cho mọi alpha
        hybrid.candidate_scores(user)
    rows = []
    for alpha in alphas:
        hybrid.alpha = float(alpha)
        rankings = {user: hybrid.ranking(user, top_k=50) for user in users}
        result, _ = evaluate_mod.evaluate(rankings, truth)
        result.update(model=f"als_mood_a{alpha:g}", stage="validation",
                      alpha=float(alpha))
        rows.append(result)
    best = max(rows, key=lambda r: (r["NDCG@10"] if r["NDCG@10"] is not None else -1))
    return rows, float(best["alpha"]), targets


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="G1: Popularity / ALS / ALS+mood trên catalogue, giao thức của TV2")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-train-users", type=int, default=10000,
                        help="0=toàn bộ; mặc định 10000 để khớp lần chạy của TV2")
    parser.add_argument("--eval-users", type=int, default=2000)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--weighting", default="binary",
                        choices=["binary", "log", "engagement"],
                        help="cấu hình TV2 đã chọn trên validation: binary")
    parser.add_argument("--alphas", default="0,0.25,0.4,0.5,0.6,0.75,0.9,1",
                        help="trọng số gu để quét trên validation")
    parser.add_argument("--report-alphas", default="0.5",
                        help="alpha báo cáo thêm trên test: điểm vận hành ĐỊNH TRƯỚC "
                             "(lấy từ A2), KHÔNG dùng để chọn gì. Rỗng = bỏ qua")
    parser.add_argument("--backend", choices=["implicit", "numpy"], default="implicit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-baselines", action="store_true",
                        help="bỏ Popularity/ALS thuần nếu đã có số của TV2")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Thư mục đầu ra không rỗng: {args.out}")
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    if not alphas or any(not 0 <= a <= 1 for a in alphas):
        parser.error("--alphas phải là các số trong [0, 1]")
    report_alphas = [float(a) for a in args.report_alphas.split(",") if a.strip()]
    if any(not 0 <= a <= 1 for a in report_alphas):
        parser.error("--report-alphas phải là các số trong [0, 1]")
    args.out.mkdir(parents=True, exist_ok=True)

    data_mod, evaluate_mod, models_mod, run_mod = taste_modules()
    print(f"[1/6] Nạp code TV2: models={models_mod.__file__}")
    print(f"      evaluate={evaluate_mod.__file__}")

    # --- dữ liệu: dùng đúng hàm đọc/lấy mẫu của TV2 -------------------------
    print(f"[2/6] Lấy mẫu {args.max_train_users} user rồi đọc train/test "
          "(stream, vài phút)...")
    selected = run_mod.selected_users_from_train(args.train, args.max_train_users,
                                                 args.seed)
    train = run_mod.load_processed(args.train, selected)
    test = run_mod.load_processed(args.test, selected)
    inner_train, validation = run_mod.inner_train_validation(train, args.seed)
    print(f"      train {len(train):,} dòng · {train.user_id.nunique():,} user · "
          f"{train.track_id.nunique():,} bài")

    # --- hồ sơ tâm trạng ----------------------------------------------------
    profiles = api.load_mood_profiles()
    proba_of = {tid: row for tid, row in
                zip(profiles.index, profiles[PROB_COLS].to_numpy(dtype=float))}
    covered = train.track_id.isin(profiles.index).mean()
    print(f"[3/6] Hồ sơ tâm trạng phủ {covered:.1%} bài trong train sample")
    if covered < 0.99:
        print("      CẢNH BÁO: phủ < 99%, bài thiếu sẽ có điểm tâm trạng 0")

    rows: list[dict] = []
    cfg = dict(factors=args.factors, regularization=.1, als_alpha=10.,
               weighting=args.weighting, iterations=args.iterations, seed=args.seed)

    # --- validation: ALS + quét alpha ---------------------------------------
    print(f"[4/6] Fit ALS({args.weighting}) trên inner-train rồi quét alpha "
          f"{alphas} trên validation...")
    als_val = models_mod.ALS(**cfg, backend=args.backend).fit(inner_train)
    if not args.skip_baselines:
        rows.append(run_mod.measure(models_mod.Popularity().fit(inner_train),
                                    validation, "validation", "popularity",
                                    args.out, False, args.eval_users, args.seed,
                                    {}, 0))
        rows.append(run_mod.measure(als_val, validation, "validation",
                                    f"als_{args.weighting}", args.out, False,
                                    args.eval_users, args.seed, {}, 0))
    sweep, best_alpha, _ = sweep_alpha(als_val, validation, inner_train, alphas,
                                       args.eval_users, args.seed, proba_of,
                                       evaluate_mod, run_mod)
    rows.extend(sweep)
    print(f"      alpha được chọn theo NDCG@10 validation: {best_alpha:g}")
    for row in sweep:
        print(f"        alpha={row['alpha']:<5g} NDCG@10={row['NDCG@10']:.6f} "
              f"Recall@10={row['Recall@10']:.6f}")

    # --- test: chạy đúng một lần -------------------------------------------
    print("[5/6] Train lại trên toàn bộ train sample, đánh giá một lần trên test...")
    als_final = models_mod.ALS(**cfg, backend=args.backend).fit(train)
    if not args.skip_baselines:
        rows.append(run_mod.measure(models_mod.Popularity().fit(train), test, "test",
                                    "popularity", args.out, False, args.eval_users,
                                    args.seed, {}, 0))
        rows.append(run_mod.measure(als_final, test, "test", "als_selected",
                                    args.out, False, args.eval_users, args.seed,
                                    {}, 0))
    truth, _, test_users = run_mod.make_truth(test, set(als_final.items),
                                              args.eval_users, args.seed)
    targets = mood_targets_for(train, test_users, proba_of)
    hybrid = TasteMoodHybrid(als_final, targets, alpha=best_alpha)
    result = run_mod.measure(hybrid, test, "test", "als_mood_selected", args.out,
                             False, args.eval_users, args.seed, {}, 0)
    result.update(alpha=best_alpha, selection_use="selected_on_validation")
    rows.append(result)

    # Điểm vận hành định trước (lấy từ A2), báo cáo thêm chứ KHÔNG dùng để chọn.
    for alpha in report_alphas:
        if abs(alpha - best_alpha) < 1e-12:
            continue
        hybrid.alpha = float(alpha)
        extra = run_mod.measure(hybrid, test, "test", f"als_mood_a{alpha:g}_reported",
                                args.out, False, args.eval_users, args.seed, {}, 0)
        extra.update(alpha=float(alpha), selection_use="reported_only")
        rows.append(extra)
        print(f"      [báo cáo thêm] alpha={alpha:g} NDCG@10={extra['NDCG@10']:.6f} "
              f"Recall@10={extra['Recall@10']:.6f}")

    # --- ghi kết quả --------------------------------------------------------
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "metrics.csv", index=False)
    (args.out / "selected_alpha.json").write_text(
        json.dumps({"alpha": best_alpha, "rule": "max NDCG@10 trên validation",
                    "alphas_considered": alphas, "taste_config": cfg},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "train_source": str(args.train.resolve()),
        "test_source": str(args.test.resolve()),
        "train_sha256": run_mod.sha256_file(args.train),
        "test_sha256": run_mod.sha256_file(args.test),
        "seed": args.seed, "backend": args.backend,
        "taste_config": cfg, "selected_alpha": best_alpha,
        "alphas_considered": alphas,
        "report_only_alphas": report_alphas,
        "max_train_users": args.max_train_users, "max_eval_users": args.eval_users,
        "official_test_used_for_selection": False,
        "validation_policy": "15% per-user validation split drawn only from official train.csv",
        "alpha_policy": "selected on validation NDCG@10; test evaluated once",
        "mood_target_policy": "dominant mood group from visible history, then user_mood_target shrinkage",
        "candidate_policy": "full fitted catalogue minus observed training history",
        "mood_profile_coverage_train": float(covered),
        "train_rows": len(train), "test_rows": len(test),
        "train_users": int(train.user_id.nunique()),
        "train_tracks": int(train.track_id.nunique()),
        "taste_models_file": models_mod.__file__,
        "taste_evaluate_file": evaluate_mod.__file__,
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[6/6] Xong.")
    cols = [c for c in ("model", "stage", "alpha", "selection_use", "Recall@10", "NDCG@10",
                        "MAP@10", "Recall@50", "NDCG@50", "MAP@50",
                        "n_users_evaluated") if c in frame.columns]
    print(frame[cols].to_string(index=False))
    print(f"Kết quả: {args.out.resolve()}")


if __name__ == "__main__":
    main()
