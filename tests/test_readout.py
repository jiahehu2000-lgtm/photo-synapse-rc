"""Numerical checks for exported readout scores and training-only spans."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import softmax

from photo_synapse_rc.readout import fit_candidates, metric, ridge_scores, training_span


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2026100307)
        self.features = self.rng.normal(size=(60, 75))
        self.features[:, -1] = 4.
        self.labels = np.tile(np.arange(1, 11), 6)
        self.train = np.arange(40)

    def test_training_span_preserves_gram_and_ignores_evaluation_scaling(self):
        phi, basis, mu, sd, diagnostic = training_span(self.features, self.train)
        standardized = (self.features-mu)/sd
        np.testing.assert_allclose(phi[self.train] @ phi[self.train].T,
                                   standardized[self.train] @ standardized[self.train].T,
                                   atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(basis.T @ basis, np.eye(diagnostic["rank"]), atol=1e-10)
        changed = self.features.copy()
        changed[40:] += 1e6
        _, _, new_mu, new_sd, _ = training_span(changed, self.train)
        np.testing.assert_array_equal(mu, new_mu)
        np.testing.assert_array_equal(sd, new_sd)

    def test_softmax_linear_export_and_npz_reload(self):
        phi, basis, mu, sd, _ = training_span(self.features, self.train)
        fits = fit_candidates(phi[self.train], self.labels[self.train], "LINEAR_SOFTMAX", [1e-4, 1e-2])
        self.assertEqual([row["C"] for row in fits], [1e-4, 1e-2])
        fit = fits[-1]
        weights = basis @ fit["weights"]
        scores = phi @ fit["weights"]+fit["bias"]
        direct = ((self.features-mu)/sd) @ weights+fit["bias"]
        np.testing.assert_allclose(direct, scores, atol=1e-11, rtol=1e-11)
        probabilities = softmax(direct, axis=1)
        np.testing.assert_allclose(probabilities.sum(1), 1., atol=1e-15)
        np.testing.assert_array_equal(probabilities.argmax(1), direct.argmax(1))
        self.assertLess(fit["softmax_gradient_max"], 2e-5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"readout.npz"
            np.savez_compressed(path, weights=weights, bias=fit["bias"], mu=mu, sd=sd, classes=np.arange(1, 11))
            with np.load(path, allow_pickle=False) as saved:
                reloaded = ((self.features-saved["mu"])/saved["sd"]) @ saved["weights"]+saved["bias"]
            np.testing.assert_array_equal(direct, reloaded)

    def test_svm_scores_match_returned_predictions(self):
        phi, _, _, _, _ = training_span(self.features, self.train)
        fit = fit_candidates(phi[self.train], self.labels[self.train], "LINEAR_SVM", [1e-3])[0]
        np.testing.assert_array_equal(fit["scores"].argmax(1)+1, fit["predicted"])
        self.assertEqual(fit["weights"].shape, (phi.shape[1], 10))

    def test_partial_class_training_is_rejected(self):
        for method in ["LINEAR_SOFTMAX", "LINEAR_SVM"]:
            with self.assertRaisesRegex(ValueError, "all ten action classes"):
                fit_candidates(self.features[:20], np.ones(20, dtype=int), method, [1e-3])

    def test_ridge_matches_independent_dual_solve(self):
        evaluation = np.arange(40, 60)
        scores = ridge_scores(self.features, self.labels, self.train, evaluation, [.1])[0]
        mu = self.features[self.train].mean(0)
        sd = self.features[self.train].std(0, ddof=1)
        sd[sd < 1e-12] = 1.
        z = (self.features[self.train]-mu)/sd
        target = np.eye(10)[self.labels[self.train]-1]
        bias = target.mean(0)
        dual = np.linalg.solve(z @ z.T+len(self.train)*.1*np.eye(len(self.train)), target-bias)
        direct = ((self.features[evaluation]-mu)/sd) @ z.T @ dual+bias
        np.testing.assert_allclose(scores, direct, atol=1e-11, rtol=1e-11)

    def test_metric_retains_ten_classes_when_some_are_unobserved(self):
        result = metric(np.array([1, 1, 2]), np.array([1, 2, 2]))
        self.assertEqual(np.asarray(result["confusion"]).shape, (10, 10))
        self.assertEqual(result["correct"], 2)
        self.assertAlmostEqual(result["accuracy_pct"], 100*2/3)
        self.assertAlmostEqual(result["macro_f1_pct"], 100*(2/3+2/3)/10)


if __name__ == "__main__":
    unittest.main()
