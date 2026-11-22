from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def read_table(path):
    path = Path(path)
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={'user_id': str, 'track_id': str})

def clean_interactions(frame):
    cols = ['user_id', 'track_id', 'playcount']
    if not set(cols).issubset(frame.columns):
        raise ValueError(f'Required columns: {cols}')
    x = frame[cols].copy()
    if x.isna().any().any():
        raise ValueError('Missing user_id, track_id or playcount')
    for col in cols[:2]:
        x[col] = x[col].astype(str).str.strip()
        if x[col].eq('').any():
            raise ValueError(f'Empty {col}')
    x['playcount'] = pd.to_numeric(x.playcount, errors='raise')
    if not np.isfinite(x.playcount).all() or (x.playcount <= 0).any():
        raise ValueError('playcount must be finite and > 0')
    x = x.groupby(cols[:2], as_index=False, sort=True).playcount.sum()
    if not np.isfinite(x.playcount).all():
        raise ValueError('Aggregated playcount overflow')
    return x.sort_values(cols[:2]).reset_index(drop=True)

def load_interactions(sample=True, path=None):
    name = 'sample_interactions.csv' if sample else 'interactions.parquet'
    return clean_interactions(read_table(path or ROOT / 'data' / name))

def load_tracks(path=None):
    x = read_table(path or ROOT / 'data' / 'tracks.csv')
    if 'track_id' not in x or x.track_id.isna().any():
        raise ValueError('tracks needs non-null track_id')
    x['track_id'] = x.track_id.astype(str)
    if x.track_id.duplicated().any():
        raise ValueError('Duplicate track_id')
    return x

def k_core(frame, min_user=10, min_item=5):
    if min_user < 1 or min_item < 1:
        raise ValueError('Thresholds must be positive')
    x = clean_interactions(frame)
    while len(x):
        before = len(x)
        x = x[x.groupby('user_id').track_id.transform('size') >= min_user]
        x = x[x.groupby('track_id').user_id.transform('size') >= min_item]
        if len(x) == before:
            break
    if x.empty:
        raise ValueError('No interactions remain after filtering')
    return x.reset_index(drop=True)

def split_interactions(frame, seed=42, val_ratio=.15, test_ratio=.15):
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError('Ratios must be > 0 and sum to < 1')
    x = clean_interactions(frame)
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for _, group in x.groupby('user_id', sort=True):
        n = len(group)
        nv, nt = max(1, int(n * val_ratio)), max(1, int(n * test_ratio))
        if n - nv - nt < 1:
            raise ValueError('Each user needs enough distinct tracks for three splits')
        order = rng.permutation(n)
        for bucket, indices in zip(parts, [order[nv+nt:], order[:nv], order[nv:nv+nt]]):
            bucket.append(group.iloc[indices])
    if not parts[0]:
        raise ValueError('Empty input')
    return tuple(pd.concat(p).sort_values(['user_id', 'track_id']).reset_index(drop=True)
                 for p in parts)

def make_sample(seed=42):
    rng = np.random.default_rng(seed)
    tracks = pd.DataFrame({
        'track_id': [f't{i:04}' for i in range(240)],
        'name': [f'Synthetic track {i}' for i in range(240)],
        'artist': [f'Synthetic artist {i//3}' for i in range(240)],
        'genre': [f'genre_{i//40}' for i in range(240)],
        'valence': rng.random(240), 'energy': rng.random(240),
    })
    rows = []
    for u in range(120):
        weights = np.ones(240)
        weights[(np.arange(240)//40) == u % 6] = 12
        chosen = rng.choice(240, 35, replace=False, p=weights/weights.sum())
        rows.extend((f'u{u:04}', f't{i:04}', int(rng.integers(1, 25))) for i in chosen)
    return tracks, clean_interactions(pd.DataFrame(rows, columns=['user_id','track_id','playcount']))
