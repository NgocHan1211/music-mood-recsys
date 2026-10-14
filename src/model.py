"""ALS (implicit feedback). Xem docs/als_summary.md và CONTRACTS.md mục 2."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def build_matrix(
    interactions: pd.DataFrame, transform: str = "log1p", alpha: float = 40.0
) -> tuple[csr_matrix, dict, dict]:
    """Trả về (ma trận user x item đã nhân alpha, user_id->idx, track_id->idx)."""
    users = interactions["user_id"].astype("category")
    items = interactions["track_id"].astype("category")
    pc = interactions["playcount"].to_numpy(dtype="float32")
    if transform == "log1p":
        pc = np.log1p(pc)
    elif transform != "raw":
        raise ValueError(f"transform không hợp lệ: {transform}")
    mat = csr_matrix(
        (pc * alpha, (users.cat.codes, items.cat.codes)),
        shape=(len(users.cat.categories), len(items.cat.categories)),
    )
    u_map = {u: i for i, u in enumerate(users.cat.categories)}
    t_map = {t: i for i, t in enumerate(items.cat.categories)}
    return mat, u_map, t_map


def train_als(matrix, factors=64, regularization=0.01, alpha=1.0, iterations=20, seed=42):
    """`matrix` đã nhân alpha trong build_matrix, nên alpha ở đây để 1.0 (tránh nhân hai lần)."""
    import implicit

    model = implicit.als.AlternatingLeastSquares(
        factors=factors, regularization=regularization, alpha=alpha,
        iterations=iterations, random_state=seed,
    )
    model.fit(matrix)  # implicit >= 0.5: ma trận user x item
    return model


def recommend(model, matrix, user_index: int, n: int = 10) -> list[tuple[int, float]]:
    ids, scores = model.recommend(
        user_index, matrix[user_index], N=n, filter_already_liked_items=True
    )
    return list(zip(ids.tolist(), scores.tolist()))


def recommend_all(model, matrix, u_map: dict, t_map: dict, users, n: int = 20) -> dict[str, list[str]]:
    """Gợi ý top-n cho nhiều user, trả về user_id -> [track_id]. Bỏ user không có trong train."""
    inv_t = {i: t for t, i in t_map.items()}
    users = [u for u in users if u in u_map]
    idx = np.array([u_map[u] for u in users])
    ids, _ = model.recommend(idx, matrix[idx], N=n, filter_already_liked_items=True)
    return {u: [inv_t[j] for j in row] for u, row in zip(users, ids)}
