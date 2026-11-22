import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import make_sample, clean_interactions, split_interactions, k_core
from src.evaluate import ranking_metrics, evaluate
from src.models import ALS, Popularity
from src.hybrid import combine_scores
from src.run import truth_for

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.raw = make_sample()
        cls.train,cls.val,cls.test = split_interactions(cls.raw)
        cls.model = ALS(factors=4,iterations=2).fit(cls.train)

    def test_metrics_hand_calculation(self):
        r,n,a = ranking_metrics(['a','x','b'],{'a','b','c'},3)
        self.assertAlmostEqual(r,2/3)
        self.assertAlmostEqual(n,1.5/(1+1/math.log2(3)+.5))
        self.assertAlmostEqual(a,(1+2/3)/3)

    def test_ap_denominator(self):
        self.assertEqual(ranking_metrics(['a'],{'a','b','c'},1),(1/3,1.,1.))

    def test_empty_and_missing_recommendations(self):
        m,_ = evaluate({}, {'u':{'a'},'v':set()})
        self.assertEqual(m['Recall@10'],0)
        self.assertEqual(m['n_users_skipped_empty'],1)

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            ranking_metrics(['a','a'],{'a'},10)

    def test_split_reproducibility_and_no_overlap(self):
        again = split_interactions(self.raw.sample(frac=1,random_state=9))
        sets = []
        for x,y in zip([self.train,self.val,self.test],again):
            pd.testing.assert_frame_equal(x,y)
            sets.append(set(zip(x.user_id,x.track_id)))
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(len(set.union(*sets)),len(self.raw))

    def test_aggregate_and_invalid_counts(self):
        x = pd.DataFrame([['u','t',2],['u','t',3]],columns=['user_id','track_id','playcount'])
        self.assertEqual(clean_interactions(x).playcount.iloc[0],5)
        x.loc[0,'playcount']=-1
        with self.assertRaises(ValueError): clean_interactions(x)

    def test_iterative_core(self):
        x = k_core(self.raw,10,5)
        self.assertGreaterEqual(x.groupby('user_id').size().min(),10)
        self.assertGreaterEqual(x.groupby('track_id').size().min(),5)

    def test_popularity_train_only_seen_filter(self):
        pop = Popularity().fit(self.train)
        u = self.train.user_id.iloc[0]
        self.assertFalse(set(t for t,_ in pop.recommend(u)) & pop.seen[u])
        self.assertEqual(pop.scores, self.train.groupby('track_id').user_id.nunique().to_dict())

    def test_als_recommendation_and_candidate_filter(self):
        u = self.model.users[0]
        available = sorted(set(self.model.items)-self.model.seen[u])[:6]
        recs = self.model.recommend(u,candidate_track_ids=available)
        self.assertEqual(set(t for t,_ in recs),set(available))
        self.assertTrue(all(np.isfinite(s) for _,s in recs))

    def test_fold_in_normal_equation(self):
        seeds = {self.model.items[0]:3,self.model.items[1]:2}
        v = self.model.fold_in(seeds)
        y = self.model.item_factors
        ids = [self.model.item_index[t] for t in sorted(seeds)]
        c = self.model._confidence([seeds[t] for t in sorted(seeds)])
        a = y.T@y + (y[ids].T*(c-1))@y[ids] + .1*np.eye(4)
        np.testing.assert_allclose(a@v,y[ids].T@c,atol=1e-9)
        self.assertFalse(set(seeds)&{t for t,_ in self.model.recommend(seed_counts=seeds)})

    def test_engagement_weighting_requires_valid_train_labels(self):
        frame = self.train.copy()
        frame['engagement_score'] = .5
        frame.loc[frame.index[0], 'engagement_score'] = 1.
        model = ALS(factors=4, iterations=2, weighting='engagement').fit(frame)
        self.assertTrue(np.isfinite(model.user_factors).all())
        broken = frame.drop(columns='engagement_score')
        with self.assertRaises(ValueError):
            ALS(factors=4, iterations=2, weighting='engagement').fit(broken)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model.npz'
            self.model.save(path)
            loaded = ALS.load(path)
            self.assertEqual(self.model.recommend(self.model.users[0]),loaded.recommend(self.model.users[0]))

    def test_unknown_user_and_seed(self):
        with self.assertRaises(ValueError): self.model.recommend('unknown')
        with self.assertRaises(ValueError): self.model.fold_in({'unknown':1})

    def test_cold_item_accounting(self):
        x = pd.DataFrame([['u','a',1],['v','new',1]],columns=['user_id','track_id','playcount'])
        truth, excluded = truth_for(x,{'a'})
        self.assertEqual(excluded,1)
        self.assertEqual(truth,{'u':{'a'},'v':set()})
        result,_ = evaluate({'u':['a']},truth)
        self.assertEqual(result['n_users_evaluated'],1)
        self.assertEqual(result['n_users_skipped_empty'],1)

    def test_no_eligible_users_is_null(self):
        metrics,_ = evaluate({}, {'u':set()})
        self.assertIsNone(metrics['Recall@10'])

    def test_hybrid_alignment_and_endpoints(self):
        a,b = {'a':10,'b':0}, {'b':.7,'a':.1}
        self.assertEqual(combine_scores(a,b,1)[0][0],'a')
        self.assertEqual(combine_scores(a,b,0)[0][0],'b')
        with self.assertRaises(ValueError): combine_scores(a,{'a':1})
        self.assertEqual(combine_scores({'a':2},{'a':.4}),[('a',0.)])

if __name__ == '__main__':
    unittest.main()
