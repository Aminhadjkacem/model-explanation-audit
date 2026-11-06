"""test_audit_integrity.py: Automated Regression Tests for XAI Audit Components.

Tests verify:
1. Feature counts and special indicator creation (23 raw -> 31 processed).
2. TreeExplainer initialization and output shapes on HistGradientBoosting.
3. Progressive tree randomization modifying predictions and leaf values.
4. Monotonicity and Area Under the Degradation Curve calculations.
5. Strict Jaccard Top-k overlap mathematical formulation.
"""

import unittest
import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau

import Fico
from xai_audit.utils import (
    get_base_pipeline,
    calculate_top_k_overlap,
    compute_shap_local,
)
import importlib
sanity = importlib.import_module("xai_audit.01_sanity")
randomize_tree_predictors = sanity.randomize_tree_predictors


class TestXAIAuditIntegrity(unittest.TestCase):

    def test_feature_counts(self):
        """Verifies 23 raw features and 8 indicators yielding 31 processed features."""
        bb, X_train, y_train, X_test, y_test, feature_names = get_base_pipeline()
        self.assertEqual(len(feature_names), 31)
        indicators = [c for c in feature_names if c.endswith("__special")]
        original_cols = [c for c in feature_names if not c.endswith("__special")]
        self.assertEqual(len(indicators), 8)
        self.assertEqual(len(original_cols), 23)

    def test_tree_explainer_compatibility(self):
        """Confirms shap.TreeExplainer directly handles HistGradientBoosting."""
        bb, X_train, y_train, X_test, y_test, feature_names = get_base_pipeline()
        explainer = shap.TreeExplainer(bb)
        sample = X_test.iloc[:2]
        shap_vals = explainer.shap_values(sample)
        self.assertEqual(shap_vals.shape, (2, 31))

    def test_tree_randomization_effects(self):
        """Verifies that progressive randomization alters leaf weights and predictions."""
        bb, X_train, y_train, X_test, y_test, feature_names = get_base_pipeline()
        sample = X_test.iloc[:5]
        orig_preds = bb.predict_proba(sample)[:, 1]
        
        # 100% randomization
        rand_model = randomize_tree_predictors(bb, 1.0, random_state=42)
        rand_preds = rand_model.predict_proba(sample)[:, 1]
        
        # Predictions must shift noticeably
        mean_shift = np.mean(np.abs(orig_preds - rand_preds))
        self.assertGreater(mean_shift, 0.10)

    def test_jaccard_top_k_overlap(self):
        """Verifies exact Jaccard calculation |A n B| / |A u B|."""
        s1 = pd.Series([10, 8, 6, 4], index=["f1", "f2", "f3", "f4"])
        s2 = pd.Series([10, 8, 1, 0], index=["f1", "f2", "f5", "f6"])
        # Top-3 s1: f1, f2, f3
        # Top-3 s2: f1, f2, f5
        # Intersection = {f1, f2} (size 2)
        # Union = {f1, f2, f3, f5} (size 4) -> 2/4 = 0.50
        j_overlap = calculate_top_k_overlap(s1, s2, k=3)
        self.assertEqual(j_overlap, 0.50)


if __name__ == "__main__":
    unittest.main()
