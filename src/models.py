"""Hu-style weighted ALS, with NumPy reference and optional implicit backend."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import clean_interactions

class Popularity:
    def fit(self, frame):
        frame = clean_interactions(frame)
        self.items = sorted(frame.track_id.unique())
        self.scores = frame.groupby('track_id').user_id.nunique().to_dict()
        self.seen = frame.groupby('user_id').track_id.agg(set).to_dict()
        return self

    def recommend(self, user_id=None, top_k=50, candidate_track_ids=None, seed_counts=None):
        if top_k < 1:
            raise ValueError('top_k must be positive')
        seen = set(seed_counts) if seed_counts is not None else self.seen.get(user_id, set())
        candidates = set(self.items if candidate_track_ids is None else candidate_track_ids)
        ids = sorted(candidates.intersection(self.items)-seen,
                     key=lambda t: (-self.scores[t], t))[:top_k]
        return [(t, float(self.scores[t])) for t in ids]

class ALS:
    def __init__(self, factors=16, regularization=.1, als_alpha=10., iterations=8,
                 seed=42, weighting='log', backend='numpy'):
        if factors < 1 or iterations < 1 or regularization <= 0 or als_alpha < 0:
            raise ValueError('Invalid ALS parameters')
        if weighting not in ('log','binary','engagement') or backend not in ('numpy','implicit'):
            raise ValueError('Invalid weighting/backend')
        self.config = dict(factors=factors, regularization=regularization,
                           als_alpha=als_alpha, iterations=iterations, seed=seed,
                           weighting=weighting, backend=backend)

    def _confidence(self, counts, engagement=None):
        counts = np.asarray(counts, dtype=float)
        if not np.isfinite(counts).all() or (counts <= 0).any():
            raise ValueError('Seed counts must be positive finite numbers')
        if self.config['weighting'] == 'binary':
            weights = np.ones_like(counts)
        elif self.config['weighting'] == 'log':
            weights = np.log1p(counts)
        else:
            if engagement is None:
                # A fold-in seed normally has no engagement label. Treat it as neutral,
                # rather than silently fabricating a favorite/passive label.
                engagement = np.full_like(counts, .5)
            engagement = np.asarray(engagement, dtype=float)
            if engagement.shape != counts.shape or not np.isfinite(engagement).all() or \
                    (engagement < 0).any() or (engagement > 1).any():
                raise ValueError('engagement_score must be finite and within [0, 1]')
            # Retain frequency while downweighting weak interactions (0.5x) and
            # upweighting strong ones (1.5x). This is selected on validation, not test.
            weights = np.log1p(counts) * (.5 + engagement)
        return 1 + self.config['als_alpha'] * weights

    def _solve(self, factors, indices, confidence, gram=None):
        f = self.config['factors']
        if not len(indices):
            return np.zeros(f)
        selected = factors[indices]
        gram = factors.T @ factors if gram is None else gram
        a = gram + (selected.T * (confidence-1)) @ selected
        a = a + self.config['regularization']*np.eye(f)
        return np.linalg.solve(a, selected.T @ confidence)

    def fit(self, frame):
        if self.config['weighting'] == 'engagement':
            required = ['user_id', 'track_id', 'playcount', 'engagement_score']
            if not set(required).issubset(frame.columns):
                raise ValueError('engagement weighting requires engagement_score in the training data')
            # The processed files are one row per user-track. Refuse ambiguous duplicate
            # engagement labels rather than combining them with an undocumented rule.
            raw = frame[required].copy()
            if raw.duplicated(['user_id', 'track_id']).any():
                raise ValueError('engagement weighting requires unique user_id/track_id rows')
            base = clean_interactions(raw[['user_id', 'track_id', 'playcount']])
            labels = raw[['user_id', 'track_id', 'engagement_score']].copy()
            labels['user_id'] = labels.user_id.astype(str).str.strip()
            labels['track_id'] = labels.track_id.astype(str).str.strip()
            x = base.merge(labels, on=['user_id', 'track_id'], how='left', validate='one_to_one')
            x['engagement_score'] = np.asarray(pd.to_numeric(x.engagement_score, errors='raise'), dtype=float)
            if not np.isfinite(x.engagement_score).all() or not x.engagement_score.between(0, 1).all():
                raise ValueError('engagement_score must be finite and within [0, 1]')
        else:
            x = clean_interactions(frame)
        if x.empty:
            raise ValueError('Empty training data')
        self.users = sorted(x.user_id.unique())
        self.items = sorted(x.track_id.unique())
        um, im = {u:i for i,u in enumerate(self.users)}, {t:i for i,t in enumerate(self.items)}
        self.item_index = im
        self.seen = x.groupby('user_id').track_id.agg(set).to_dict()
        ui = x.user_id.map(um).to_numpy()
        ti = x.track_id.map(im).to_numpy()
        confidence = self._confidence(x.playcount.to_numpy(),
                                      x.engagement_score.to_numpy() if 'engagement_score' in x else None)
        if self.config['backend']=='implicit':
            from scipy.sparse import csr_matrix
            from implicit.cpu.als import AlternatingLeastSquares
            matrix = csr_matrix((confidence.astype('float32'),(ui,ti)),
                                shape=(len(self.users),len(self.items)))
            model = AlternatingLeastSquares(
                factors=self.config['factors'], regularization=self.config['regularization'],
                alpha=1., iterations=self.config['iterations'], random_state=self.config['seed'],
                num_threads=1)
            model.fit(matrix, show_progress=False)
            self.user_factors = model.user_factors.astype(float)
            self.item_factors = model.item_factors.astype(float)
        else:
            if len(x) > 100000:
                raise ValueError('NumPy reference is for small runs; use --backend implicit')
            rng = np.random.default_rng(self.config['seed'])
            self.user_factors = rng.normal(0,.05,(len(self.users),self.config['factors']))
            self.item_factors = rng.normal(0,.05,(len(self.items),self.config['factors']))
            user_rows = [[] for _ in self.users]
            item_rows = [[] for _ in self.items]
            for u,t,c in zip(ui,ti,confidence):
                user_rows[u].append((t,c))
                item_rows[t].append((u,c))
            for _ in range(self.config['iterations']):
                for target, other, groups in [(self.user_factors,self.item_factors,user_rows),
                                               (self.item_factors,self.user_factors,item_rows)]:
                    gram = other.T @ other
                    for i, group in enumerate(groups):
                        indices, cs = zip(*group)
                        target[i] = self._solve(other, list(indices), np.array(cs), gram)
        self.user_index = um
        return self

    def fold_in(self, seed_counts):
        known = sorted(set(seed_counts).intersection(self.item_index))
        if not known:
            raise ValueError('No seed tracks in trained catalogue; use Popularity fallback')
        return self._solve(self.item_factors, [self.item_index[t] for t in known],
                           self._confidence([seed_counts[t] for t in known]))

    def recommend(self, user_id=None, top_k=50, candidate_track_ids=None, seed_counts=None):
        if top_k < 1:
            raise ValueError('top_k must be positive')
        if seed_counts is not None:
            vector, seen = self.fold_in(seed_counts), set(seed_counts)
        elif user_id in self.user_index:
            vector, seen = self.user_factors[self.user_index[user_id]], self.seen[user_id]
        else:
            raise ValueError('Unknown user: supply seed_counts or use Popularity')
        candidates = set(self.items if candidate_track_ids is None else candidate_track_ids)
        ids = sorted(candidates.intersection(self.item_index)-seen)
        scores = self.item_factors[[self.item_index[t] for t in ids]] @ vector
        keep = min(top_k, len(ids))
        if keep < len(ids):
            selected = np.argpartition(-scores, keep - 1)[:keep]
            pairs = [(ids[i], float(scores[i])) for i in selected]
        else:
            pairs = list(zip(ids, map(float, scores)))
        return sorted(pairs, key=lambda p:(-p[1],p[0]))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(config=self.config, users=self.users, items=self.items,
                    seen={u:sorted(v) for u,v in self.seen.items()})
        with path.open('wb') as f:
            np.savez_compressed(f, user_factors=self.user_factors, item_factors=self.item_factors,
                                metadata=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            model = cls(**meta['config'])
            model.user_factors = z['user_factors'].copy()
            model.item_factors = z['item_factors'].copy()
        model.users, model.items = meta['users'], meta['items']
        model.seen = {u:set(v) for u,v in meta['seen'].items()}
        model.user_index = {u:i for i,u in enumerate(model.users)}
        model.item_index = {t:i for i,t in enumerate(model.items)}
        return model
