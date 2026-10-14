import numpy as np
import pandas as pd
from src.split import split_interactions
from src.evaluate import evaluate


def make_df(n_users=50, n_items=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        k = rng.integers(3, 30)
        for t in rng.choice(n_items, k, replace=False):
            rows.append((f"u{u}", f"t{t}", int(rng.integers(1, 20))))
    return pd.DataFrame(rows, columns=["user_id", "track_id", "playcount"])


def test_random_holdout_no_overlap_and_users_in_train():
    df = make_df()
    tr, te = split_interactions(df, "random_holdout", 0.2, 5, 42)
    key = lambda d: set(zip(d.user_id, d.track_id))
    assert not key(tr) & key(te)
    assert set(te.user_id) <= set(tr.user_id)
    assert tr.groupby("user_id").size().min() >= 1


def test_min_interactions_filter():
    df = make_df()
    tr, te = split_interactions(df, "random_holdout", 0.2, 10, 42)
    full = pd.concat([tr, te]).groupby("user_id").size()
    assert full.min() >= 10


def test_leave_one_out_one_per_user():
    tr, te = split_interactions(make_df(), "leave_one_out", min_user_interactions=5)
    assert te.groupby("user_id").size().eq(1).all()


def test_reproducible():
    df = make_df()
    a = split_interactions(df, seed=1)[1]
    b = split_interactions(df, seed=1)[1]
    assert a.equals(b)


def test_evaluate_perfect_and_zero():
    test = pd.DataFrame({"user_id": ["a", "a"], "track_id": ["x", "y"]})
    perfect = evaluate({"a": ["x", "y", "z"]}, test, ks=(2,))
    assert perfect["recall@2"] == 1.0 and abs(perfect["ndcg@2"] - 1.0) < 1e-9
    zero = evaluate({"a": ["p", "q"]}, test, ks=(2,))
    assert zero["recall@2"] == 0.0 and zero["ndcg@2"] == 0.0
