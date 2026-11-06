"""03_stability.py: Experiment 3 – Stability (Local Robustness under Perturbations).

Academic Question:
-------------------
Do explanations remain stable and consistent when small, realistic perturbations
are introduced to the input data?

Background & Distinction:
-------------------------
In interpretable machine learning, we must distinguish between two concepts:
1. Prediction Stability: The percentage of times the model's discrete prediction
   (predicted class or broad probability regime) remains unchanged when inputs are perturbed.
2. Explanation Stability (Local Lipschitz Continuity / Robustness): The degree to
   which the generated feature importance rankings remain identical when the model's
   prediction is virtually unchanged.

An explanation method that produces wildly fluctuating explanations for two nearly
identical applicant profiles (e.g. an applicant whose credit balance differs by $1)
is untrustworthy in financial risk auditing.

Experimental Design:
--------------------
1. Select test instances from X_test.
2. Handling Special Codes and Binary Indicators:
   - HELOC contains continuous numerical features (e.g., ExternalRiskEstimate, AverageMInFile)
     and binary indicator flags (e.g., MSOptional__special, MaxDelq2PublicRecLast12M__special).
   - Rule: We ONLY perturb genuine continuous numerical features.
   - We strictly leave binary indicator columns (0 or 1) and missing values (NaN) untouched
     to maintain distributional validity.
3. Apply small epsilon noise to continuous features:
   x_perturbed = x + epsilon * std(x)
   where epsilon in [0.01, 0.03, 0.05] (1% to 5% standard deviation noise).
4. Evaluate:
   - Prediction Stability Percentage: fraction of perturbed instances where predicted class (threshold 0.5) is identical.
   - Explanation Stability (Spearman Rank Correlation): Rank correlation between pre- and post-perturbation feature importances.
   - Top-3 Feature Overlap (Jaccard Index): Fraction of top-3 most influential features that remain in top-3 after perturbation.
5. Compare local stability across SHAP and LIME.

Outputs:
--------
- CSV tables in `results/tables/`:
    * stability_metrics.csv
- Figures in `results/figures/`:
    * stability_rank_correlation_comparison.png
    * stability_top3_overlap.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xai_audit.utils import (
    get_base_pipeline,
    compute_shap_local,
    create_lime_explainer,
    compute_lime_local,
    calculate_spearman_rho,
    calculate_top_k_overlap,
    TABLES_DIR,
    FIGURES_DIR,
)


def run_stability_experiment():
    """Executes Experiment 3: Explanation stability against small input perturbations."""
    print("=" * 80)
    print("EXPERIMENT 3: STABILITY (Explanation Robustness Under Small Perturbations)")
    print("=" * 80)

    # 1. Load pipeline and data
    bb, X_train, y_train, X_test, y_test, feature_names = get_base_pipeline()
    lime_explainer = create_lime_explainer(X_train)

    # Identify continuous numerical vs binary/indicator features
    continuous_cols = [
        col for col in feature_names
        if not col.endswith("__special") and X_train[col].dropna().nunique() > 10
    ]
    binary_cols = [col for col in feature_names if col not in continuous_cols]
    
    print(f"\n[Step 1/4] Feature taxonomy:")
    print(f"  - Continuous numerical features subject to small noise: {len(continuous_cols)}")
    print(f"  - Binary indicator / discrete features held fixed: {len(binary_cols)}")

    # Continuous standard deviations
    feat_stds = X_train[continuous_cols].std()

    # Select representative test instances (e.g. 80 instances)
    test_sample = X_test.sample(n=80, random_state=42).copy()
    base_preds = bb.predict(test_sample)
    base_probs = bb.predict_proba(test_sample)[:, 1]

    # Pre-compute original local explanations
    print("\n[Step 2/4] Pre-computing original local explanations for test instances...")
    orig_shap_list = []
    orig_lime_list = []

    for i in range(len(test_sample)):
        row = test_sample.iloc[i]
        orig_shap_list.append(compute_shap_local(bb, row))
        orig_lime_list.append(compute_lime_local(lime_explainer, bb, row, num_samples=800))

    # =========================================================================
    # Step 3: Apply Perturbations across noise levels (epsilon = 1%, 3%, 5%)
    # =========================================================================
    epsilon_levels = [0.01, 0.03, 0.05]
    stability_records = []

    print("\n[Step 3/4] Testing stability across noise levels epsilon in [1%, 3%, 5%]...")
    rng = np.random.RandomState(42)

    for eps in epsilon_levels:
        print(f"\n  Evaluating noise level epsilon = {int(eps * 100)}% of feature standard deviation...")

        perturbed_instances = test_sample.copy().astype(float)
        for col in continuous_cols:
            # Gaussian noise scaled by feature standard deviation
            noise = rng.normal(0.0, eps * feat_stds[col], size=len(perturbed_instances))
            # Preserve missing values (NaN) if present
            non_na_mask = perturbed_instances[col].notna()
            perturbed_instances.loc[non_na_mask, col] = (
                perturbed_instances.loc[non_na_mask, col] + noise[non_na_mask]
            )

        # Check prediction stability
        perturbed_preds = bb.predict(perturbed_instances)
        perturbed_probs = bb.predict_proba(perturbed_instances)[:, 1]
        pred_stability_pct = float(np.mean(base_preds == perturbed_preds) * 100.0)
        mean_prob_shift = float(np.mean(np.abs(base_probs - perturbed_probs)))

        # Evaluate explanation stability
        shap_spearman_scores = []
        shap_top3_overlaps = []
        lime_spearman_scores = []
        lime_top3_overlaps = []

        for i in range(len(test_sample)):
            orig_row = test_sample.iloc[i]
            pert_row = perturbed_instances.iloc[i]

            # SHAP on perturbed instance
            pert_shap = compute_shap_local(bb, pert_row)
            s_spearman = calculate_spearman_rho(orig_shap_list[i].abs(), pert_shap.abs())
            s_top3 = calculate_top_k_overlap(orig_shap_list[i].abs(), pert_shap.abs(), k=3)
            shap_spearman_scores.append(s_spearman)
            shap_top3_overlaps.append(s_top3)

            # LIME on perturbed instance
            pert_lime = compute_lime_local(lime_explainer, bb, pert_row, num_samples=800)
            l_spearman = calculate_spearman_rho(orig_lime_list[i].abs(), pert_lime.abs())
            l_top3 = calculate_top_k_overlap(orig_lime_list[i].abs(), pert_lime.abs(), k=3)
            lime_spearman_scores.append(l_spearman)
            lime_top3_overlaps.append(l_top3)

        # Add record for SHAP
        stability_records.append({
            "Noise_Level_Epsilon": f"{int(eps * 100)}%",
            "Explanation_Method": "SHAP",
            "Prediction_Stability_Pct": round(pred_stability_pct, 2),
            "Mean_Abs_Probability_Shift": round(mean_prob_shift, 4),
            "Explanation_Stability_Spearman": round(float(np.mean(shap_spearman_scores)), 4),
            "Explanation_Top3_Overlap": round(float(np.mean(shap_top3_overlaps)), 4),
        })

        # Add record for LIME
        stability_records.append({
            "Noise_Level_Epsilon": f"{int(eps * 100)}%",
            "Explanation_Method": "LIME",
            "Prediction_Stability_Pct": round(pred_stability_pct, 2),
            "Mean_Abs_Probability_Shift": round(mean_prob_shift, 4),
            "Explanation_Stability_Spearman": round(float(np.mean(lime_spearman_scores)), 4),
            "Explanation_Top3_Overlap": round(float(np.mean(lime_top3_overlaps)), 4),
        })

    # Note on Permutation Feature Importance:
    # Permutation Importance is a global dataset-level validation metric (computed over the entire test set).
    # We report it as a global reference.
    from xai_audit.utils import compute_permutation_importance
    global_perm_orig = compute_permutation_importance(bb, test_sample, y_test.loc[test_sample.index], random_state=42)
    # Permutation on slightly perturbed sample
    global_perm_pert = compute_permutation_importance(bb, perturbed_instances, y_test.loc[test_sample.index], random_state=42)
    perm_spearman = calculate_spearman_rho(global_perm_orig, global_perm_pert)
    perm_top3 = calculate_top_k_overlap(global_perm_orig, global_perm_pert, k=3)

    stability_records.append({
        "Noise_Level_Epsilon": "5% (Global Dataset)",
        "Explanation_Method": "Permutation Importance",
        "Prediction_Stability_Pct": round(pred_stability_pct, 2),
        "Mean_Abs_Probability_Shift": round(mean_prob_shift, 4),
        "Explanation_Stability_Spearman": round(perm_spearman, 4),
        "Explanation_Top3_Overlap": round(perm_top3, 4),
    })

    df_stability = pd.DataFrame(stability_records)
    csv_stab_path = os.path.join(TABLES_DIR, "stability_metrics.csv")
    df_stability.to_csv(csv_stab_path, index=False)
    print(f"\n[Step 4/4] Saved stability table to: {csv_stab_path}")

    # =========================================================================
    # Visualizations
    # =========================================================================
    df_local = df_stability[df_stability["Explanation_Method"].isin(["SHAP", "LIME"])]

    # Figure 1: Spearman Rank Correlation Decay
    plt.figure(figsize=(8, 4.5))
    for method, color, marker in [("SHAP", "#2b5c8f", "o"), ("LIME", "#d95f02", "s")]:
        sub = df_local[df_local["Explanation_Method"] == method]
        plt.plot(
            sub["Noise_Level_Epsilon"],
            sub["Explanation_Stability_Spearman"],
            marker=marker,
            linewidth=2.5,
            color=color,
            label=f"{method} Explanation Stability"
        )
    plt.title("Explanation Stability: Feature Ranking Spearman Correlation vs. Input Noise", fontsize=11, fontweight="bold")
    plt.xlabel("Input Perturbation Noise Level (% of Feature Std)", fontsize=10)
    plt.ylabel("Mean Spearman Rank Correlation (Values near 1.0 = Highly Stable)", fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="lower left")
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "stability_rank_correlation_comparison.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig1_path}")

    # Figure 2: Top-3 Overlap vs Prediction Stability
    plt.figure(figsize=(8, 4.5))
    for method, color, marker in [("SHAP", "#2b5c8f", "o"), ("LIME", "#d95f02", "s")]:
        sub = df_local[df_local["Explanation_Method"] == method]
        plt.plot(
            sub["Noise_Level_Epsilon"],
            sub["Explanation_Top3_Overlap"],
            marker=marker,
            linewidth=2.5,
            color=color,
            label=f"{method} Top-3 Overlap"
        )
    plt.title("Explanation Stability: Top-3 Feature Jaccard Overlap under Perturbations", fontsize=11, fontweight="bold")
    plt.xlabel("Input Perturbation Noise Level (% of Feature Std)", fontsize=10)
    plt.ylabel("Top-3 Feature Overlap Fraction (Jaccard)", fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="lower left")
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "stability_top3_overlap.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig2_path}")

    print("\nStability Results Summary:")
    print("--------------------------------------------------------------------------------")
    print(df_stability.to_string(index=False))
    print("--------------------------------------------------------------------------------")
    print("Stability experiment successfully completed.\n")


if __name__ == "__main__":
    run_stability_experiment()

