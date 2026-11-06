"""01_sanity.py: Experiment 1 – Sanity Checks for XAI Explanations.

Academic Question:
-------------------
Do explanation methods actually depend on the learned parameters and genuine
relationships of the trained model, or do they produce artifacts independent of
the model's parameters (acting merely as edge detectors or input statistics)?

Background & Methodology:
-------------------------
Following the seminal work by Adebayo et al. (2018) "Sanity Checks for Saliency
Maps", a trustworthy explanation method must be sensitive to model parameters
and training data relationships. An explanation method that outputs identical or
highly correlated feature importances even when the model's learned weights are
randomized fails the basic sanity requirement.

In this experiment, we conduct two rigorous sanity checks on HistGradientBoosting:

1. Test A: Model Parameter Randomization (Progressive Tree Randomization)
   - HistGradientBoostingClassifier is an ensemble of decision trees.
   - We progressively randomize the internal tree parameters (leaf values and split thresholds)
     in three successive stages:
       * Stage 0: Original trained model (0% randomized)
       * Stage 1: Randomize the last 33% of trees (Trees 67%-100%)
       * Stage 2: Randomize the last 66% of trees (Trees 33%-100%)
       * Stage 3: Randomize 100% of trees (Fully randomized model)
   - At each stage, we compute feature importances using SHAP, LIME, and Permutation Importance.
   - We evaluate the rank similarity (Kendall's Tau and Spearman's rho) compared to the original model.
   - Expected Sanity Outcome: Similarity should drop steadily toward zero as randomization progresses.

2. Test B: Random-Label Training (Permuted Targets)
   - We shuffle the training labels y_train uniformly at random, breaking any real statistical
     dependency between features X and target y.
   - We train a new HistGradientBoostingClassifier using the identical hyperparameters and architecture.
   - We compute SHAP, LIME, Permutation Importance, and Feature Variance on the random-label model.
   - Expected Sanity Outcome:
     * A sane explanation method should produce explanations uncorrelated with the original model's
       explanations (Kendall's Tau near 0).
     * The feature variance baseline is completely independent of the model and serves as a null reference.

Outputs:
--------
- CSV tables saved in `results/tables/`:
    * sanity_parameter_randomization.csv
    * sanity_random_labels.csv
- Visualizations saved in `results/figures/`:
    * sanity_parameter_randomization_decay.png
    * sanity_random_label_comparison.png
"""

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier

from xai_audit.utils import (
    get_base_pipeline,
    compute_feature_variance,
    compute_shap_importance,
    create_lime_explainer,
    compute_lime_global_importance,
    compute_permutation_importance,
    calculate_kendall_tau,
    calculate_spearman_rho,
    calculate_top_k_overlap,
    TABLES_DIR,
    FIGURES_DIR,
)


def randomize_tree_predictors(model, fraction_to_randomize, random_state=42):
    """Safely randomizes the learned internal decision tree parameters of HistGradientBoosting.
    
    Args:
        model: Trained HistGradientBoostingClassifier.
        fraction_to_randomize: Float between 0.0 and 1.0 (portion of final boosting stages to randomize).
        random_state: Seed for reproducibility.
        
    Returns:
        randomized_model: Deep-copied model with modified tree predictors.
    """
    model_copy = copy.deepcopy(model)
    if fraction_to_randomize <= 0.0:
        return model_copy

    rng = np.random.RandomState(random_state)
    n_trees = len(model_copy._predictors)
    start_idx = int(np.floor((1.0 - fraction_to_randomize) * n_trees))
    start_idx = max(0, min(start_idx, n_trees - 1))

    for i in range(start_idx, n_trees):
        predictor = model_copy._predictors[i][0]
        nodes = predictor.nodes
        # Randomize leaf prediction values and split thresholds
        # Standard normal for leaf values (logits), uniform for thresholds
        leaf_mask = nodes["is_leaf"] == 1
        internal_mask = ~leaf_mask
        
        if np.any(leaf_mask):
            nodes["value"][leaf_mask] = rng.normal(loc=0.0, scale=1.0, size=np.sum(leaf_mask))
        if np.any(internal_mask):
            # Shuffle feature indices and randomize thresholds
            nodes["feature_idx"][internal_mask] = rng.choice(
                model_copy.n_features_in_, size=np.sum(internal_mask)
            )

    return model_copy


def run_sanity_experiment():
    """Executes Experiment 1: Model parameter randomization & random-label training."""
    print("=" * 80)
    print("EXPERIMENT 1: SANITY CHECKS (Adebayo et al. Protocol)")
    print("=" * 80)

    # 1. Load trained model and data
    bb, X_train, y_train, X_test, y_test, feature_names = get_base_pipeline()
    lime_explainer = create_lime_explainer(X_train)
    
    # Representative sample for compute efficiency (150 evaluation samples)
    eval_sample = X_test.sample(n=150, random_state=42)
    eval_sample_y = y_test.loc[eval_sample.index]

    print("\n[Step 1/4] Computing original model explanations & baseline feature variance...")
    # Baseline Feature Variance (Model-independent baseline)
    feat_var = compute_feature_variance(X_train)

    # Original model explanations
    orig_shap = compute_shap_importance(bb, eval_sample)
    orig_lime = compute_lime_global_importance(lime_explainer, bb, eval_sample, num_samples_lime=800)
    orig_perm = compute_permutation_importance(bb, eval_sample, eval_sample_y, n_repeats=5, random_state=42)

    print(f"Top 3 Original SHAP Features: {list(orig_shap.nlargest(3).index)}")
    print(f"Top 3 Original LIME Features: {list(orig_lime.nlargest(3).index)}")
    print(f"Top 3 Original Permutation Features: {list(orig_perm.nlargest(3).index)}")

    # =========================================================================
    # TEST A: Progressive Parameter Randomization
    # =========================================================================
    print("\n[Step 2/4] Test A: Progressive Parameter Randomization (0%, 33%, 66%, 100%)...")
    randomization_stages = [
        ("Original (0%)", 0.0),
        ("Stage 1 (33% Trees)", 0.33),
        ("Stage 2 (66% Trees)", 0.66),
        ("Stage 3 (100% Trees)", 1.00),
    ]

    param_results = []

    for stage_name, frac in randomization_stages:
        print(f"  Evaluating {stage_name}...")
        rand_model = randomize_tree_predictors(bb, fraction_to_randomize=frac, random_state=42)
        
        # Calculate explanations on randomized model
        stage_shap = compute_shap_importance(rand_model, eval_sample)
        stage_lime = compute_lime_global_importance(lime_explainer, rand_model, eval_sample, num_samples_lime=800)
        stage_perm = compute_permutation_importance(rand_model, eval_sample, eval_sample_y, n_repeats=5, random_state=42)

        # Correlation with Original Explanations
        shap_tau = calculate_kendall_tau(orig_shap, stage_shap)
        shap_rho = calculate_spearman_rho(orig_shap, stage_shap)
        shap_top3 = calculate_top_k_overlap(orig_shap, stage_shap, k=3)

        lime_tau = calculate_kendall_tau(orig_lime, stage_lime)
        lime_rho = calculate_spearman_rho(orig_lime, stage_lime)
        lime_top3 = calculate_top_k_overlap(orig_lime, stage_lime, k=3)

        perm_tau = calculate_kendall_tau(orig_perm, stage_perm)
        perm_rho = calculate_spearman_rho(orig_perm, stage_perm)
        perm_top3 = calculate_top_k_overlap(orig_perm, stage_perm, k=3)

        # Correlation with Feature Variance Baseline
        shap_var_tau = calculate_kendall_tau(stage_shap, feat_var)
        lime_var_tau = calculate_kendall_tau(stage_lime, feat_var)
        perm_var_tau = calculate_kendall_tau(stage_perm, feat_var)

        param_results.append({
            "Stage": stage_name,
            "Randomized_Fraction": frac,
            "SHAP_Kendall_Tau": round(shap_tau, 4),
            "SHAP_Spearman_Rho": round(shap_rho, 4),
            "SHAP_Top3_Overlap": round(shap_top3, 4),
            "LIME_Kendall_Tau": round(lime_tau, 4),
            "LIME_Spearman_Rho": round(lime_rho, 4),
            "LIME_Top3_Overlap": round(lime_top3, 4),
            "Permutation_Kendall_Tau": round(perm_tau, 4),
            "Permutation_Spearman_Rho": round(perm_rho, 4),
            "Permutation_Top3_Overlap": round(perm_top3, 4),
            "SHAP_vs_Variance_Tau": round(shap_var_tau, 4),
            "LIME_vs_Variance_Tau": round(lime_var_tau, 4),
            "Permutation_vs_Variance_Tau": round(perm_var_tau, 4),
        })

    df_param = pd.DataFrame(param_results)
    param_csv_path = os.path.join(TABLES_DIR, "sanity_parameter_randomization.csv")
    df_param.to_csv(param_csv_path, index=False)
    print(f"  -> Saved parameter randomization table to: {param_csv_path}")

    # Plot Test A: Explanation Rank Decay
    plt.figure(figsize=(9, 5))
    stages_x = [r["Stage"] for r in param_results]
    plt.plot(stages_x, df_param["SHAP_Kendall_Tau"], marker="o", linewidth=2.5, label="SHAP (Kendall's Tau)")
    plt.plot(stages_x, df_param["LIME_Kendall_Tau"], marker="s", linewidth=2.5, label="LIME (Kendall's Tau)")
    plt.plot(stages_x, df_param["Permutation_Kendall_Tau"], marker="^", linewidth=2.5, label="Permutation (Kendall's Tau)")
    plt.axhline(0.0, color="gray", linestyle="--", alpha=0.7, label="Null Association (Tau = 0)")
    plt.title("Sanity Test A: Explanation Sensitivity to Model Parameter Randomization", fontsize=12, fontweight="bold")
    plt.xlabel("Randomization Stage (Learned Trees Scrambled)", fontsize=11)
    plt.ylabel("Kendall's Tau Correlation with Original Model", fontsize=11)
    plt.ylim(-0.4, 1.1)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig_param_path = os.path.join(FIGURES_DIR, "sanity_parameter_randomization_decay.png")
    plt.savefig(fig_param_path, dpi=300)
    plt.close()
    print(f"  -> Saved parameter decay figure to: {fig_param_path}")

    # =========================================================================
    # TEST B: Random-Label Training (Permuted Targets)
    # =========================================================================
    print("\n[Step 3/4] Test B: Training Model on Random Shuffled Labels...")
    rng = np.random.RandomState(42)
    y_train_shuffled = y_train.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    # Train a new HistGradientBoostingClassifier with identical architecture on shuffled labels
    import Fico
    random_label_model = HistGradientBoostingClassifier(
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
        **Fico.BEST_PARAMS,
    )
    random_label_model.fit(X_train, y_train_shuffled)

    print("  Calculating explanations on random-label model...")
    rl_shap = compute_shap_importance(random_label_model, eval_sample)
    rl_lime = compute_lime_global_importance(lime_explainer, random_label_model, eval_sample, num_samples_lime=800)
    rl_perm = compute_permutation_importance(random_label_model, eval_sample, eval_sample_y, n_repeats=5, random_state=42)

    # Compare explanations between Original Model and Random-Label Model
    random_label_results = [
        {
            "Method": "SHAP",
            "Kendall_Tau_vs_Original": round(calculate_kendall_tau(orig_shap, rl_shap), 4),
            "Spearman_Rho_vs_Original": round(calculate_spearman_rho(orig_shap, rl_shap), 4),
            "Top3_Overlap_vs_Original": round(calculate_top_k_overlap(orig_shap, rl_shap, k=3), 4),
            "Kendall_Tau_vs_Feature_Variance": round(calculate_kendall_tau(rl_shap, feat_var), 4),
            "Sanity_Status": "PASSED" if abs(calculate_kendall_tau(orig_shap, rl_shap)) < 0.3 else "FAILED",
        },
        {
            "Method": "LIME",
            "Kendall_Tau_vs_Original": round(calculate_kendall_tau(orig_lime, rl_lime), 4),
            "Spearman_Rho_vs_Original": round(calculate_spearman_rho(orig_lime, rl_lime), 4),
            "Top3_Overlap_vs_Original": round(calculate_top_k_overlap(orig_lime, rl_lime, k=3), 4),
            "Kendall_Tau_vs_Feature_Variance": round(calculate_kendall_tau(rl_lime, feat_var), 4),
            "Sanity_Status": "PASSED" if abs(calculate_kendall_tau(orig_lime, rl_lime)) < 0.3 else "FAILED",
        },
        {
            "Method": "Permutation Importance",
            "Kendall_Tau_vs_Original": round(calculate_kendall_tau(orig_perm, rl_perm), 4),
            "Spearman_Rho_vs_Original": round(calculate_spearman_rho(orig_perm, rl_perm), 4),
            "Top3_Overlap_vs_Original": round(calculate_top_k_overlap(orig_perm, rl_perm, k=3), 4),
            "Kendall_Tau_vs_Feature_Variance": round(calculate_kendall_tau(rl_perm, feat_var), 4),
            "Sanity_Status": "PASSED" if abs(calculate_kendall_tau(orig_perm, rl_perm)) < 0.3 else "FAILED",
        },
        {
            "Method": "Feature Variance (Baseline)",
            "Kendall_Tau_vs_Original": round(calculate_kendall_tau(orig_shap, feat_var), 4),
            "Spearman_Rho_vs_Original": round(calculate_spearman_rho(orig_shap, feat_var), 4),
            "Top3_Overlap_vs_Original": round(calculate_top_k_overlap(orig_shap, feat_var, k=3), 4),
            "Kendall_Tau_vs_Feature_Variance": 1.0000,
            "Sanity_Status": "BASELINE",
        },
    ]

    df_random_label = pd.DataFrame(random_label_results)
    rl_csv_path = os.path.join(TABLES_DIR, "sanity_random_labels.csv")
    df_random_label.to_csv(rl_csv_path, index=False)
    print(f"  -> Saved random-label table to: {rl_csv_path}")

    # Plot Test B: Correlation Bar Chart
    plt.figure(figsize=(8, 4.5))
    methods = [r["Method"] for r in random_label_results if r["Method"] != "Feature Variance (Baseline)"]
    tau_vals = [r["Kendall_Tau_vs_Original"] for r in random_label_results if r["Method"] != "Feature Variance (Baseline)"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    bars = plt.bar(methods, tau_vals, color=colors, width=0.45, edgecolor="black")
    plt.axhline(0.0, color="black", linestyle="-", linewidth=0.8)
    plt.axhspan(-0.2, 0.2, color="green", alpha=0.15, label="Expected Sanity Band (-0.2 <= Tau <= 0.2)")
    plt.title("Sanity Test B: Explanation Similarity to Model Trained on Random Labels", fontsize=11, fontweight="bold")
    plt.ylabel("Kendall's Tau with Original Explanations", fontsize=10)
    plt.ylim(-0.5, 0.8)
    for bar in bars:
        height = bar.get_height()
        y_pos = height + 0.03 if height >= 0 else height - 0.08
        plt.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f"Tau = {height:.3f}", ha="center", fontsize=10, fontweight="bold")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig_rl_path = os.path.join(FIGURES_DIR, "sanity_random_label_comparison.png")
    plt.savefig(fig_rl_path, dpi=300)
    plt.close()
    print(f"  -> Saved random-label comparison figure to: {fig_rl_path}")

    print("\n[Step 4/4] Sanity Summary Findings:")
    print("--------------------------------------------------------------------------------")
    for r in random_label_results:
        print(f"  * {r['Method']:<25} | Kendall Tau: {r['Kendall_Tau_vs_Original']:>6.3f} | Sanity Status: {r['Sanity_Status']}")
    print("--------------------------------------------------------------------------------")
    print("Sanity experiment successfully completed.\n")


if __name__ == "__main__":
    run_sanity_experiment()

