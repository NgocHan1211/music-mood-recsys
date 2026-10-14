"""Các metric xếp hạng. Người phụ trách: Hân. Xem docs/evaluation_protocol.md mục 6."""
import numpy as np
import pandas as pd


def evaluate(
    recommendations: dict[str, list[str]],
    test: pd.DataFrame,
    ks: tuple[int, ...] = (10, 20),
    n_items: int | None = None,
) -> dict[str, float]:
    """recommendations: user_id -> danh sách track_id đã xếp hạng (đã loại bài trong train).
    Chỉ tính cho user có trong cả `recommendations` và `test`.
    """
    truth = test.groupby("user_id")["track_id"].apply(set).to_dict()
    users = [u for u in truth if u in recommendations]
    out: dict[str, float] = {"n_users": float(len(users))}
    for k in ks:
        rec_k, pre_k, ndcg_k, ap_k, hit_k = [], [], [], [], []
        recommended = set()
        for u in users:
            recs = recommendations[u][:k]
            rel = truth[u]
            recommended.update(recs)
            hits = np.array([1.0 if t in rel else 0.0 for t in recs])
            n_hit = hits.sum()
            rec_k.append(n_hit / len(rel))
            pre_k.append(n_hit / k)
            hit_k.append(1.0 if n_hit > 0 else 0.0)
            disc = 1.0 / np.log2(np.arange(2, len(recs) + 2))
            dcg = float((hits * disc).sum())
            idcg = float(disc[: min(len(rel), len(recs))].sum()) if len(recs) else 0.0
            ndcg_k.append(dcg / idcg if idcg > 0 else 0.0)
            prec_at = np.cumsum(hits) / np.arange(1, len(recs) + 1)
            ap_k.append(float((prec_at * hits).sum() / min(len(rel), k)) if len(recs) else 0.0)
        m = lambda x: float(np.mean(x)) if x else 0.0
        out[f"recall@{k}"] = m(rec_k)
        out[f"precision@{k}"] = m(pre_k)
        out[f"ndcg@{k}"] = m(ndcg_k)
        out[f"map@{k}"] = m(ap_k)
        out[f"hitrate@{k}"] = m(hit_k)
        if n_items:
            out[f"coverage@{k}"] = len(recommended) / n_items
    return out
