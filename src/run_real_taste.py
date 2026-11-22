"""Train and evaluate taste recommenders on Thy's fixed processed train/test files.

The test file is never used to select a weighting or hyperparameter.  If
engagement_score exists, it is consumed only from train; test engagement_level is
used only for stratified reporting after ranking.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .data import clean_interactions, read_table, split_interactions
from .evaluate import evaluate
from .models import ALS, Popularity


CORE_COLUMNS = ['user_id', 'track_id', 'playcount']


def selected_users_from_train(path: Path, max_users: int, seed: int) -> set[str] | None:
    """Choose users without materialising a multi-million-row CSV in RAM."""
    if max_users == 0:
        return None
    if path.suffix.lower() == '.parquet':
        users = pd.read_parquet(path, columns=['user_id']).user_id.astype(str).str.strip().unique()
    else:
        users_seen: set[str] = set()
        for chunk in pd.read_csv(path, usecols=['user_id'], dtype={'user_id': str}, chunksize=100_000):
            users_seen.update(chunk.user_id.dropna().astype(str).str.strip())
        users = np.array(sorted(users_seen))
    if not len(users):
        raise ValueError(f'{path} has no users')
    rng = np.random.default_rng(seed)
    return set(rng.choice(users, min(max_users, len(users)), replace=False))


def load_processed(path: Path, allowed_users: set[str] | None = None) -> pd.DataFrame:
    """Load only needed columns, optionally streaming just a fixed user sample.

    Thy's train file has millions of rows.  Streaming prevents a low-memory laptop
    from allocating the whole file before ``--max-train-users`` can take effect.
    """
    path = Path(path)
    if path.suffix.lower() == '.parquet':
        header = pd.read_parquet(path, columns=None).columns
    else:
        header = pd.read_csv(path, nrows=0).columns
    if not set(CORE_COLUMNS).issubset(header):
        raise ValueError(f'{path} needs columns {CORE_COLUMNS}')
    keep = CORE_COLUMNS + [col for col in ('engagement_score', 'engagement_level') if col in header]
    if path.suffix.lower() == '.parquet':
        x = pd.read_parquet(path, columns=keep)
        if allowed_users is not None:
            x = x[x.user_id.astype(str).str.strip().isin(allowed_users)]
    elif allowed_users is None:
        # Full-data mode is deliberately explicit: it needs enough RAM for every row.
        x = pd.read_csv(path, usecols=keep, dtype={'user_id': str, 'track_id': str})
    else:
        pieces = []
        for chunk in pd.read_csv(path, usecols=keep, dtype={'user_id': str, 'track_id': str}, chunksize=100_000):
            chunk['user_id'] = chunk.user_id.astype(str).str.strip()
            filtered = chunk[chunk.user_id.isin(allowed_users)]
            if not filtered.empty:
                pieces.append(filtered)
        if not pieces:
            raise ValueError(f'{path} contains no rows for the selected train users')
        x = pd.concat(pieces, ignore_index=True)
    x = x.copy()
    x['user_id'] = x.user_id.astype(str).str.strip()
    x['track_id'] = x.track_id.astype(str).str.strip()
    x['playcount'] = pd.to_numeric(x.playcount, errors='raise')
    # Processed splits should already be unique. Refusing duplicates avoids an
    # undocumented rule for combining their engagement labels.
    if x.duplicated(['user_id', 'track_id']).any():
        raise ValueError(f'{path} has duplicate user_id/track_id rows')
    clean_interactions(x[CORE_COLUMNS])
    if 'engagement_score' in x:
        x['engagement_score'] = pd.to_numeric(x.engagement_score, errors='raise')
        if not x.engagement_score.between(0, 1).all():
            raise ValueError(f'{path}: engagement_score must be in [0, 1]')
    return x.sort_values(['user_id', 'track_id']).reset_index(drop=True)


def sample_users(train: pd.DataFrame, test: pd.DataFrame, max_users: int, seed: int):
    if max_users == 0:
        return train, test
    users = np.array(sorted(train.user_id.unique()))
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(users, min(max_users, len(users)), replace=False))
    return (train[train.user_id.isin(chosen)].reset_index(drop=True),
            test[test.user_id.isin(chosen)].reset_index(drop=True))


def sha256_file(path: Path) -> str:
    """Hash large source files without reading the complete CSV into memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def inner_train_validation(train: pd.DataFrame, seed: int):
    """Split only the provided training file; the official holdout remains sealed."""
    base_train, validation, _ = split_interactions(train[CORE_COLUMNS], seed=seed)
    keys_train = base_train[['user_id', 'track_id']]
    keys_validation = validation[['user_id', 'track_id']]
    return (train.merge(keys_train, on=['user_id', 'track_id'], how='inner', validate='one_to_one'),
            train.merge(keys_validation, on=['user_id', 'track_id'], how='inner', validate='one_to_one'))


def build_artist_lookup(tracks_path: Path | None, artist_col: str) -> dict[str, str]:
    if tracks_path is None:
        return {}
    tracks = read_table(tracks_path)
    if not {'track_id', artist_col}.issubset(tracks.columns):
        raise ValueError(f'{tracks_path} needs track_id and {artist_col}')
    tracks = tracks[['track_id', artist_col]].dropna().copy()
    tracks.track_id = tracks.track_id.astype(str).str.strip()
    if tracks.track_id.duplicated().any():
        raise ValueError('tracks file has duplicate track_id')
    return dict(zip(tracks.track_id, tracks[artist_col].astype(str).str.strip()))


def artist_rerank(ranked, artist_of: dict[str, str], max_per_artist: int, top_k: int):
    if max_per_artist == 0 or not artist_of:
        return ranked[:top_k]
    chosen, counts = [], {}
    for track_id, score in ranked:
        artist = artist_of.get(track_id)
        if artist and counts.get(artist, 0) >= max_per_artist:
            continue
        chosen.append((track_id, score))
        if artist:
            counts[artist] = counts.get(artist, 0) + 1
        if len(chosen) == top_k:
            break
    return chosen


def make_truth(heldout: pd.DataFrame, catalogue: set[str], max_users: int, seed: int):
    eligible = heldout[heldout.track_id.isin(catalogue)].copy()
    grouped = eligible.groupby('user_id').track_id.agg(set).to_dict()
    users = sorted(grouped)
    if max_users and len(users) > max_users:
        rng = np.random.default_rng(seed)
        users = sorted(rng.choice(np.array(users), max_users, replace=False))
    truth = {user: grouped[user] for user in users}
    return truth, len(heldout) - len(eligible), users


def add_segment_metrics(metrics: dict, rankings: dict, heldout: pd.DataFrame, users: list[str]):
    """Report test engagement strata without feeding them into a model."""
    if 'engagement_level' not in heldout:
        return
    visible = heldout[heldout.user_id.isin(users)]
    for level in sorted(visible.engagement_level.dropna().astype(str).unique()):
        truth = (visible[visible.engagement_level.astype(str) == level]
                 .groupby('user_id').track_id.agg(set).to_dict())
        if truth:
            result, _ = evaluate(rankings, truth)
            for key in ('Recall@10', 'NDCG@10', 'MAP@10', 'n_users_evaluated'):
                metrics[f'{level}_{key}'] = result[key]


def measure(model, heldout, stage, label, out, synthetic, eval_users, seed,
            artist_of, max_per_artist):
    truth, unknown, users = make_truth(heldout, set(model.items), eval_users, seed)
    rankings, records = {}, []
    pool_size = 500 if max_per_artist else 50
    for user in users:
        ranked = model.recommend(user, top_k=pool_size)
        ranked = artist_rerank(ranked, artist_of, max_per_artist, top_k=50)
        rankings[user] = [track_id for track_id, _ in ranked]
        records.extend((user, track_id, rank, score) for rank, (track_id, score) in enumerate(ranked, 1))
    metrics, detail = evaluate(rankings, truth)
    add_segment_metrics(metrics, rankings, heldout, users)
    metrics.update(model=label, stage=stage, synthetic=synthetic,
                   excluded_unseen_item_positives=unknown,
                   heldout_positives=int(sum(len(value) for value in truth.values())),
                   eligible_users_before_sampling=len(make_truth(heldout, set(model.items), 0, seed)[0]),
                   max_eval_users=eval_users or 'all',
                   artist_cap=max_per_artist or 'none')
    detail.to_csv(out / f'{stage}_{label}_per_user.csv', index=False)
    pd.DataFrame(records, columns=['user_id', 'track_id', 'rank', 'score']).to_csv(
        out / f'{stage}_{label}_recommendations.csv', index=False)
    return metrics


def configurations(weightings: list[str], factors: int, iterations: int, seed: int):
    return [dict(factors=factors, regularization=.1, als_alpha=10.,
                 weighting=weighting, iterations=iterations, seed=seed)
            for weighting in weightings]


def main():
    parser = argparse.ArgumentParser(description='Real-data taste experiment using fixed processed train/test')
    parser.add_argument('--train', required=True, type=Path, help='Thy data/processed/train.csv')
    parser.add_argument('--test', required=True, type=Path, help='Thy data/processed/test.csv; never used for selection')
    parser.add_argument('--tracks', type=Path, help='Optional tracks/Music Info file for artist diversity reranking')
    parser.add_argument('--artist-col', default='artist')
    parser.add_argument('--max-per-artist', type=int, default=0, help='0 disables reranking; e.g. 2')
    parser.add_argument('--backend', choices=['implicit', 'numpy'], default='implicit',
                        help='Use numpy only for tiny smoke runs; implicit is required for real data.')
    parser.add_argument('--weightings', default='binary,log,engagement')
    parser.add_argument('--factors', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--max-train-users', type=int, default=0, help='0=all; use 10000 for a smoke run')
    parser.add_argument('--eval-users', type=int, default=2000, help='0=all eligible users')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    if args.factors < 1 or args.iterations < 1 or args.max_per_artist < 0 or \
            args.max_train_users < 0 or args.eval_users < 0:
        parser.error('factors/iterations must be positive; user limits and artist cap must be non-negative')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f'Output folder is not empty: {args.out}')
    args.out.mkdir(parents=True, exist_ok=True)
    # Sample user IDs first, then stream both large CSVs.  This must happen before
    # loading train.csv; otherwise a 5.6M-row train file may exhaust laptop RAM.
    selected_users = selected_users_from_train(args.train, args.max_train_users, args.seed)
    train = load_processed(args.train, selected_users)
    test = load_processed(args.test, selected_users)
    if 'engagement' in args.weightings and 'engagement_score' not in train:
        parser.error('engagement weighting requested but train has no engagement_score')
    weightings = [value.strip() for value in args.weightings.split(',') if value.strip()]
    if not weightings or any(value not in {'binary', 'log', 'engagement'} for value in weightings):
        parser.error('--weightings must be a comma-separated subset of binary,log,engagement')
    artist_of = build_artist_lookup(args.tracks, args.artist_col)
    inner_train, validation = inner_train_validation(train, args.seed)
    rows, best_score, selected = [], -float('inf'), None
    val_popularity = Popularity().fit(inner_train)
    rows.append(measure(val_popularity, validation, 'validation', 'popularity', args.out, False,
                        args.eval_users, args.seed, artist_of, args.max_per_artist))
    for index, config in enumerate(configurations(weightings, args.factors, args.iterations, args.seed)):
        model = ALS(**config, backend=args.backend).fit(inner_train)
        label = f"als_{config['weighting']}_{index}"
        result = measure(model, validation, 'validation', label, args.out, False,
                         args.eval_users, args.seed, artist_of, args.max_per_artist)
        rows.append(result)
        if result['NDCG@10'] is not None and result['NDCG@10'] > best_score:
            best_score, selected = result['NDCG@10'], config
    if selected is None:
        raise ValueError('No eligible validation users after filtering')
    (args.out / 'selected_config.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    if not args.validation_only:
        final = ALS(**selected, backend=args.backend).fit(train)
        rows.append(measure(Popularity().fit(train), test, 'test', 'popularity', args.out, False,
                            args.eval_users, args.seed, artist_of, args.max_per_artist))
        rows.append(measure(final, test, 'test', 'als_selected', args.out, False,
                            args.eval_users, args.seed, artist_of, args.max_per_artist))
        final.save(args.out / 'als_model.npz')
    pd.DataFrame(rows).to_csv(args.out / 'metrics.csv', index=False)
    manifest = {
        'train_source': str(args.train.resolve()),
        'test_source': str(args.test.resolve()),
        'train_sha256': sha256_file(args.train),
        'test_sha256': sha256_file(args.test),
        'seed': args.seed, 'backend': args.backend, 'weightings_considered': weightings,
        'selected_config': selected, 'max_train_users': args.max_train_users,
        'max_eval_users': args.eval_users, 'max_per_artist': args.max_per_artist,
        'official_test_used_for_selection': False,
        'validation_policy': '15% per-user validation split drawn only from official train.csv',
        'test_engagement_policy': 'reporting strata only; never model input',
        'candidate_policy': 'full fitted catalogue minus observed training history',
        'train_rows': len(train), 'test_rows': len(test),
        'train_users': int(train.user_id.nunique()), 'train_tracks': int(train.track_id.nunique()),
        'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
    }
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'Outputs: {args.out.resolve()}')


if __name__ == '__main__':
    main()
