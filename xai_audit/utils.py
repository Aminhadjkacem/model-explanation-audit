"""xai_audit/utils.py: Shared Utilities, Data Loading, and Explainers for XAI Audit.

This module provides common functionality across all XAI audit experiments:
1. Loading the HELOC dataset and baseline HistGradientBoostingClassifier model from Fico.py.
2. Standardized wrappers to compute:
   - SHAP (SHapley Additive exPlanations via TreeExplainer)
   - LIME (Local Interpretable Model-agnostic Explanations via LimeTabularExplainer)
   - Permutation Feature Importance (Global feature importance via scikit-learn)
   - Feature Variance (Model-independent baseline)
3. Helper functions for statistical comparisons (Kendall's Tau, Spearman Rank Correlation, Top-k Overlap).
4. Plotting and table export helpers.
"""

import os
import copy
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.inspection import permutation_importance
import shap
import lime
import lime.lime_tabular

# Ensure output directories exist
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Set writable matplotlib cache dir
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_xai_audit"


def get_base_pipeline():
    """Loads the trained black-box model and split datasets from Fico.py.
    
    Returns:
        bb: Trained HistGradientBoostingClassifier model.
        X_train: Training features DataFrame (30% split, search).
        y_train: Training labels Series.
        X_test: Evaluation features DataFrame (70% split, eval).
        y_test: Evaluation labels Series.
        feature_names: List of feature names.
    """
    import Fico
    bb = Fico.bb
    X_train = Fico.X_search.copy()
    y_train = Fico.y_search.copy()
    X_test = Fico.X_eval.copy()
    y_test = Fico.y_eval.copy()
    feature_names = list(X_train.columns)
    return bb, X_train, y_train, X_test, y_test, feature_names


def compute_feature_variance(X_df):
    """Computes normalized feature variance as a model-independent baseline.
    
    Features with higher variance exhibit higher dispersion in the input space,
    serving as a baseline to verify whether an explanation method is merely
    reflecting data variance rather than learned model relationships.
    """
    variances = X_df.var(numeric_only=True)
    # Ensure all columns present
    for col in X_df.columns:
        if col not in variances:
            variances[col] = 0.0
    variances = variances[X_df.columns]
    # Normalize to sum to 1.0
    total = variances.sum()
    if total > 0:
        normalized_var = variances / total
    else:
        normalized_var = variances
    return normalized_var


def compute_shap_importance(model, X_sample):
    """Computes global SHAP feature importance (mean absolute SHAP value).
    
    Uses TreeExplainer for tree ensembles. Returns a Series indexed by feature names.
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    # shap_vals shape can be (n_samples, n_features) or (n_samples, n_features, n_classes)
    if isinstance(shap_vals, list):
        # Multi-class or list of classes: take positive class (Bad = 1)
        shap_matrix = np.abs(shap_vals[1])
    elif len(shap_vals.shape) == 3:
        shap_matrix = np.abs(shap_vals[:, :, 1])
    else:
        shap_matrix = np.abs(shap_vals)
    
    mean_abs_shap = np.mean(shap_matrix, axis=0)
    return pd.Series(mean_abs_shap, index=X_sample.columns)


def compute_shap_local(model, sample_row):
    """Computes local SHAP values for an individual observation.
    
    Args:
        model: HistGradientBoostingClassifier.
        sample_row: DataFrame or Series representing a single observation.
        
    Returns:
        pd.Series of signed SHAP values for class 1 (RiskPerformance = Bad).
    """
    if isinstance(sample_row, pd.Series):
        sample_df = sample_row.to_frame().T
    else:
        sample_df = sample_row
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(sample_df)
    if isinstance(shap_vals, list):
        local_vals = shap_vals[1][0]
    elif len(shap_vals.shape) == 3:
        local_vals = shap_vals[0, :, 1]
    else:
        local_vals = shap_vals[0]
    return pd.Series(local_vals, index=sample_df.columns)


def create_lime_explainer(X_train):
    """Initializes a LimeTabularExplainer on training data.
    
    LIME expects imputed continuous numerical values during perturbation sampling.
    """
    X_train_imputed = X_train.fillna(X_train.median())
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_imputed.values,
        feature_names=list(X_train.columns),
        class_names=["Good", "Bad"],
        mode="classification",
        random_state=42
    )
    return explainer


def compute_lime_local(lime_explainer, model, sample_row, num_features=None, num_samples=1500):
    """Computes local LIME feature weights for an individual observation for class 1 (Bad).
    
    Args:
        lime_explainer: Initialized LimeTabularExplainer.
        model: Trained classifier.
        sample_row: Series representing a single instance.
        num_features: Number of features to score (defaults to all features).
        num_samples: Number of perturbation samples for surrogate training.
        
    Returns:
        pd.Series of LIME coefficients for class 1 indexed by feature names.
    """
    feature_names = lime_explainer.feature_names
    if num_features is None:
        num_features = len(feature_names)
        
    # Impute missing values for LIME's perturbation kernel
    row_values = sample_row.copy()
    if isinstance(row_values, pd.DataFrame):
        row_values = row_values.iloc[0]
    row_imputed = row_values.fillna(sample_row.median()).values.astype(float)
    
    exp = lime_explainer.explain_instance(
        data_row=row_imputed,
        predict_fn=model.predict_proba,
        labels=[1],
        num_features=num_features,
        num_samples=num_samples
    )
    
    # Map back to full feature list
    weights = pd.Series(0.0, index=feature_names)
    # exp.local_exp[1] returns list of (feature_idx, weight)
    for feat_idx, weight in exp.local_exp.get(1, []):
        feat_name = feature_names[feat_idx]
        weights[feat_name] = weight
        
    return weights


def compute_lime_global_importance(lime_explainer, model, X_sample, num_samples_lime=1000):
    """Computes global feature importance by averaging absolute LIME local weights across instances.
    
    Args:
        lime_explainer: Initialized LimeTabularExplainer.
        model: Trained model.
        X_sample: DataFrame of instances to sample and aggregate.
        num_samples_lime: Number of perturbation points per local explanation.
        
    Returns:
        pd.Series of mean absolute LIME weights indexed by feature names.
    """
    feature_names = list(X_sample.columns)
    abs_weights_matrix = []
    
    for idx in range(len(X_sample)):
        row = X_sample.iloc[idx]
        local_w = compute_lime_local(
            lime_explainer, model, row, num_features=len(feature_names), num_samples=num_samples_lime
        )
        abs_weights_matrix.append(np.abs(local_w.values))
        
    mean_abs_lime = np.mean(abs_weights_matrix, axis=0)
    return pd.Series(mean_abs_lime, index=feature_names)


def compute_permutation_importance(model, X_eval, y_eval, n_repeats=5, random_state=42):
    """Computes Permutation Feature Importance on evaluation data.
    
    Measures the drop in model performance (ROC AUC) when a feature's values are shuffled.
    
    Returns:
        pd.Series of mean importance scores indexed by feature names.
    """
    perm = permutation_importance(
        model,
        X_eval,
        y_eval,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    return pd.Series(perm.importances_mean, index=X_eval.columns)


def calculate_kendall_tau(series_a, series_b):
    """Computes Kendall's Tau rank correlation coefficient between two feature importance Series.
    
    Values range from -1 (inverted ranks) to +1 (identical ranks). 0 indicates no concordance.
    """
    common_idx = [col for col in series_a.index if col in series_b.index]
    tau, p_value = kendalltau(series_a[common_idx], series_b[common_idx])
    if np.isnan(tau):
        tau = 0.0
    return float(tau)


def calculate_spearman_rho(series_a, series_b):
    """Computes Spearman's rank correlation coefficient between two feature importance Series."""
    common_idx = [col for col in series_a.index if col in series_b.index]
    rho, p_value = spearmanr(series_a[common_idx], series_b[common_idx])
    if np.isnan(rho):
        rho = 0.0
    return float(rho)


def calculate_top_k_overlap(series_a, series_b, k=3):
    """Computes the Jaccard overlap fraction of the top-k features between two Series.
    
    Score ranges from 0.0 (disjoint top-k) to 1.0 (exact same top-k set).
    """
    top_a = set(series_a.nlargest(k).index)
    top_b = set(series_b.nlargest(k).index)
    intersection = len(top_a.intersection(top_b))
    union = len(top_a.union(top_b))
    return float(intersection / union) if union > 0 else 0.0

