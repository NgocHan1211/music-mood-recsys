import math
import pandas as pd

def ranking_metrics(ranking, relevant, k):
    if k < 1:
        raise ValueError('k must be positive')
    if len(ranking) != len(set(ranking)):
        raise ValueError('Duplicate recommended track')
    relevant = set(relevant)
    if not relevant:
        raise ValueError('Empty ground truth')
    hits = [int(t in relevant) for t in ranking[:k]]
    recall = sum(hits) / len(relevant)
    dcg = sum(h/math.log2(i+2) for i, h in enumerate(hits))
    ideal = sum(1/math.log2(i+2) for i in range(min(k, len(relevant))))
    found, ap = 0, 0.
    for i, hit in enumerate(hits, 1):
        found += hit
        ap += hit * found / i
    return recall, dcg/ideal, ap/min(k, len(relevant))

def evaluate(recommendations, ground_truth, ks=(10,50)):
    if not ks or any(k < 1 for k in ks):
        raise ValueError('Provide positive cutoffs')
    rows, skipped = [], 0
    for user, truth in sorted(ground_truth.items()):
        if not truth:
            skipped += 1
            continue
        row = {'user_id': user}
        for k in ks:
            values = ranking_metrics(recommendations.get(user, []), truth, k)
            row.update({f'{name}@{k}': v for name, v in zip(['Recall','NDCG','MAP'],values)})
        rows.append(row)
    detail = pd.DataFrame(rows)
    result = {f'{name}@{k}': (float(detail[f'{name}@{k}'].mean()) if rows else None)
              for k in ks for name in ['Recall','NDCG','MAP']}
    result.update(n_users_evaluated=len(rows), n_users_skipped_empty=skipped)
    return result, detail
