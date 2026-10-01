"""Behavioral checks using synthetic data only."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cluster_local_logistic import ClusterLocalLogisticClassifier


class ClusterLocalLogisticTests(unittest.TestCase):
    def test_pure_clusters_return_stored_labels(self) -> None:
        left = np.linspace(-6, -4, 30)
        right = np.linspace(4, 6, 30)
        frame = pd.DataFrame({"x": np.r_[left, right]})
        labels = np.r_[np.zeros(30, dtype=int), np.ones(30, dtype=int)]
        model = ClusterLocalLogisticClassifier(
            max_clusters=2,
            c_values=(1.0,),
            imbalance_cutoffs=(0.3,),
            validation_splits=1,
            random_state=7,
        ).fit(frame, labels)

        self.assertEqual(model.selected_k_, 2)
        self.assertEqual(set(model.cluster_report()["fit_mode"]), {"pure"})
        predicted = model.predict(pd.DataFrame({"x": [-5.0, 5.0]}))
        np.testing.assert_array_equal(predicted, [0, 1])
        probabilities = model.predict_proba(pd.DataFrame({"x": [-5.0, 5.0]}))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_mixed_clusters_weighting_and_unknown_category(self) -> None:
        rng = np.random.default_rng(9)
        x = np.r_[rng.normal(-5, 0.2, 80), rng.normal(5, 0.2, 80)]
        labels = np.r_[np.tile([0, 0, 0, 1], 20), np.tile([0, 0, 0, 1], 20)]
        frame = pd.DataFrame({"x": x, "category": ["known"] * len(x)})
        frame.loc[0, "x"] = np.nan
        model = ClusterLocalLogisticClassifier(
            max_clusters=2,
            c_values=(1.0,),
            imbalance_cutoffs=(0.5,),
            validation_splits=1,
            random_state=11,
        ).fit(frame, labels)

        self.assertIn("weighted", set(model.cluster_report()["fit_mode"]))
        new = pd.DataFrame({"x": [-5.0, 5.0], "category": ["unseen", "known"]})
        probability = model.predict_proba(new)
        self.assertEqual(probability.shape, (2, 2))
        self.assertTrue(np.isfinite(probability).all())
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)

    def test_same_seed_is_reproducible(self) -> None:
        rng = np.random.default_rng(25)
        frame = pd.DataFrame({"x": rng.normal(size=100), "z": rng.normal(size=100)})
        labels = (frame["x"] + frame["z"] > 0).astype(int).to_numpy()
        options = dict(
            max_clusters=3,
            c_values=(0.1, 1.0),
            imbalance_cutoffs=(0.2, 0.4),
            validation_splits=1,
            random_state=29,
        )
        first = ClusterLocalLogisticClassifier(**options).fit(frame, labels)
        second = ClusterLocalLogisticClassifier(**options).fit(frame, labels)
        self.assertEqual(first.selected_k_, second.selected_k_)
        self.assertEqual(first.selected_c_, second.selected_c_)
        self.assertEqual(first.selected_cutoff_, second.selected_cutoff_)
        np.testing.assert_allclose(first.predict_proba(frame), second.predict_proba(frame))


if __name__ == "__main__":
    unittest.main()
