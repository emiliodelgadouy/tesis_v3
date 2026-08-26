import unittest

import numpy as np
import pandas as pd

from src.evaluation_report import (
    bootstrap_metrics_by_patient,
    compute_metrics,
    select_validation_thresholds,
)


class EvaluationReportTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p2", "p3", "p4", "p4"],
                "y_true": [1, 1, 0, 0, 1, 0],
                "y_prob": [0.9, 0.8, 0.1, 0.2, 0.7, 0.3],
            }
        )

    def test_thresholds_and_metrics_are_finite(self):
        thresholds = select_validation_thresholds(
            self.frame["y_true"], self.frame["y_prob"]
        )
        self.assertEqual(
            set(thresholds),
            {
                "balanced_accuracy",
                "specificity_90",
                "specificity_95",
                "sensitivity_90",
            },
        )
        metrics = compute_metrics(self.frame, thresholds)
        for key in ("roc_auc", "pr_auc", "pr_lift", "brier", "accuracy"):
            self.assertTrue(np.isfinite(metrics[key]), key)
        self.assertEqual(metrics["patients"], 4)

    def test_bootstrap_resamples_patient_clusters(self):
        thresholds = select_validation_thresholds(
            self.frame["y_true"], self.frame["y_prob"]
        )
        bootstrap = bootstrap_metrics_by_patient(
            self.frame, thresholds, samples=20, seed=42
        )
        self.assertGreater(len(bootstrap), 0)
        self.assertLessEqual(len(bootstrap), 20)
        self.assertIn("roc_auc", bootstrap)


if __name__ == "__main__":
    unittest.main()
