"""04_agreement.py: Experiment 4 – Agreement Between Explanation Methods.

Academic Question:
-------------------
Do different explanation methods (SHAP, LIME, and Permutation Feature Importance)
identify the same set of important features, or do they give conflicting answers
to domain experts and loan officers?

Background & Methodology:
-------------------------
A major open problem in Explainable AI is the "Disagreement Problem" (Krishna et al., 2022).
Even on the exact same dataset and trained model, different interpretability frameworks
often highlight completely different features as "most critical".

In this experiment, we:
1. Compute the global feature rankings produced by:
   - SHAP (Mean Absolute SHAP values across evaluation instances)
   - LIME (Mean Absolute LIME weights across evaluation instances)
   - Permutation Feature Importance (Drop in ROC AUC on evaluation data)
2. Pairwise Agreement Metrics:
   - Top-1 Agreement: Boolean flag (1 or 0) indicating whether both methods agree on the single most important feature.
   - Top-3 Agreement (Jaccard Overlap): Fraction of overlapping features in the top-3 sets.
   - Kendall's Tau: Rank concordance metric across all features (-1.0 to +1.0).
   - Spearman's Rank Correlation: Monotonic rank correlation (-1.0 to +1.0).
3. Hypothesis Investigation for the Lowest-Agreement Pair:
   - We identify which pair of explanation methods exhibits the lowest agreement.
   - We formulate a clear academic hypothesis:
     * Hypothesis: "LIME's local linear surrogate in high-dimensional tabular spaces is distorted
       by multi-collinearity and sampling in off-manifold regions, whereas SHAP's TreeExplainer
       and Permutation Importance directly follow true tree split paths and marginal feature distributions."
   - Investigation Experiment: We evaluate how feature collinearity (correlation matrix) and LIME sample size
     (perturbation variance) affect the disagreement between LIME and SHAP/Permutation.

Outputs:
--------
- CSV tables in `results/tables/`:
    * agreement_pairwise_metrics.csv
    * agreement_feature_rankings.csv
    * agreement_hypothesis_investigation.csv
- Figures in `results/figures/`:
    * agreement_top_features_comparison.png
    * agreement_rank_correlation_heatmap.png
    * agreement_hypothesis_disagreement_analysis.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau, spearmanr

from xai_audit.utils import (
    get_base_pipeline,
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


def run_agreement_experiment():
    """Executes Experiment 4: Inter-method agreement and hypothesis investigation."""
    print("=" * 80)
    print("EXPERIMENT 4: INTER-METHOD AGREEMENT & DISAGREEMENT INVESTIGATION")
    print("=" * 80)

    # 1. Load pipeline and data
    bb, X_train, y_train, X_test, y_test, feature_names = get_base_pipeline()
    lime_explainer = create_lime_explainer(X_train)

    eval_sample = X_test.sample(n=150, random_state=42)
    eval_sample_y = y_test.loc[eval_sample.index]

    print("\n[Step 1/4] Calculating global feature importance across methods...")
    shap_imp = compute_shap_importance(bb, eval_sample)
    lime_imp = compute_lime_global_importance(lime_explainer, bb, eval_sample, num_samples_lime=800)
    perm_imp = compute_permutation_importance(bb, eval_sample, eval_sample_y, n_repeats=5, random_state=42)

    # Compile into DataFrame
    df_rankings = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Score": shap_imp.values,
        "SHAP_Rank": shap_imp.rank(ascending=False).astype(int).values,
        "LIME_Score": lime_imp.values,
        "LIME_Rank": lime_imp.rank(ascending=False).astype(int).values,
        "Permutation_Score": perm_imp.values,
        "Permutation_Rank": perm_imp.rank(ascending=False).astype(int).values,
    }).sort_values("SHAP_Rank").reset_index(drop=True)

    csv_ranks_path = os.path.join(TABLES_DIR, "agreement_feature_rankings.csv")
    df_rankings.to_csv(csv_ranks_path, index=False)
    print(f"  -> Saved all feature rankings to: {csv_ranks_path}")

    # Top features summary
    print("\n  Top 5 Features by Method:")
    print("  - SHAP:       ", list(df_rankings.sort_values("SHAP_Rank")["Feature"].head(5)))
    print("  - LIME:       ", list(df_rankings.sort_values("LIME_Rank")["Feature"].head(5)))
    print("  - Permutation:", list(df_rankings.sort_values("Permutation_Rank")["Feature"].head(5)))

    # =========================================================================
    # Step 2: Pairwise Agreement Metrics
    # =========================================================================
    pairs = [
        ("SHAP", "LIME", shap_imp, lime_imp),
        ("SHAP", "Permutation", shap_imp, perm_imp),
        ("LIME", "Permutation", lime_imp, perm_imp),
    ]

    agreement_results = []
    for name_a, name_b, series_a, series_b in pairs:
        top1_a = series_a.idxmax()
        top1_b = series_b.idxmax()
        top1_agree = 1.0 if top1_a == top1_b else 0.0

        top3_overlap = calculate_top_k_overlap(series_a, series_b, k=3)
        top5_overlap = calculate_top_k_overlap(series_a, series_b, k=5)
        k_tau = calculate_kendall_tau(series_a, series_b)
        s_rho = calculate_spearman_rho(series_a, series_b)

        agreement_results.append({
            "Pair": f"{name_a} vs {name_b}",
            "Top1_Agreement": top1_agree,
            "Top1_Feature_A": top1_a,
            "Top1_Feature_B": top1_b,
            "Top3_Overlap": round(top3_overlap, 4),
            "Top5_Overlap": round(top5_overlap, 4),
            "Kendall_Tau": round(k_tau, 4),
            "Spearman_Rho": round(s_rho, 4),
        })

    df_agreement = pd.DataFrame(agreement_results)
    csv_agree_path = os.path.join(TABLES_DIR, "agreement_pairwise_metrics.csv")
    df_agreement.to_csv(csv_agree_path, index=False)
    print(f"\n[Step 2/4] Saved pairwise agreement metrics to: {csv_agree_path}")

    # Identify lowest agreement pair
    lowest_row = df_agreement.sort_values("Kendall_Tau").iloc[0]
    lowest_pair = lowest_row["Pair"]
    lowest_tau = lowest_row["Kendall_Tau"]
    print(f"\n  Lowest Agreement Pair: {lowest_pair} (Kendall's Tau = {lowest_tau:.4f})")

    # =========================================================================
    # Step 3: Hypothesis Investigation for Disagreement
    # =========================================================================
    print("\n[Step 3/4] Investigating Root Cause Hypothesis for Disagreement...")
    print("--------------------------------------------------------------------------------")
    print("HYPOTHESIS:")
    print("In tabular credit scoring data, highly collinear features (such as delinquency counts")
    print("and credit inquiries) cause LIME's localized ridge regression surrogate to exhibit high")
    print("variance, splitting credit weights inconsistently across correlated dimensions.")
    print("Conversely, Tree SHAP efficiently computes exact path Shapley values, and Permutation")
    print("Importance directly penalizes model accuracy across the empirical feature manifold.")
    print("--------------------------------------------------------------------------------")

    # Investigation Experiment:
    # 1. Compute pairwise collinearity matrix of features
    corr_matrix = X_train.corr().abs()
    # Compute average correlation with other features for each feature
    mean_collinearity = corr_matrix.mean()

    # 2. Check disagreement gap: absolute difference in normalized ranks between LIME and SHAP
    df_rankings["Rank_Diff_LIME_SHAP"] = (df_rankings["LIME_Rank"] - df_rankings["SHAP_Rank"]).abs()
    df_rankings["Rank_Diff_LIME_Perm"] = (df_rankings["LIME_Rank"] - df_rankings["Permutation_Rank"]).abs()
    df_rankings["Collinearity_Score"] = df_rankings["Feature"].map(mean_collinearity)

    # Correlation between collinearity and rank disagreement
    corr_collin_shap_lime, p_val = spearmanr(
        df_rankings["Collinearity_Score"].dropna(),
        df_rankings["Rank_Diff_LIME_SHAP"].dropna()
    )

    # 3. Test LIME perturbation sample size sensitivity (sample sizes: 200, 800, 2000)
    print("  Testing LIME sample size sensitivity on disagreement...")
    lime_sample_sizes = [200, 800, 2000]
    lime_sample_sensitivity = []
    sub_eval = eval_sample.head(40)
    
    for n_s in lime_sample_sizes:
        sub_lime = compute_lime_global_importance(lime_explainer, bb, sub_eval, num_samples_lime=n_s)
        tau_with_shap = calculate_kendall_tau(sub_lime, shap_imp)
        tau_with_perm = calculate_kendall_tau(sub_lime, perm_imp)
        lime_sample_sensitivity.append({
            "LIME_Perturbation_Samples": n_s,
            "Kendall_Tau_with_SHAP": round(tau_with_shap, 4),
            "Kendall_Tau_with_Permutation": round(tau_with_perm, 4),
        })

    df_hypothesis = pd.DataFrame(lime_sample_sensitivity)
    csv_hyp_path = os.path.join(TABLES_DIR, "agreement_hypothesis_investigation.csv")
    df_hypothesis.to_csv(csv_hyp_path, index=False)
    print(f"  -> Saved hypothesis experiment results to: {csv_hyp_path}")

    # =========================================================================
    # Step 4: Visualizations
    # =========================================================================
    # Figure 1: Top 10 Features Side-by-Side Comparison
    top_feats = list(df_rankings.sort_values("SHAP_Rank")["Feature"].head(10))
    plot_df = df_rankings[df_rankings["Feature"].isin(top_feats)].copy()
    plot_df = plot_df.sort_values("SHAP_Rank")

    fig, ax = plt.subplots(figsize=(10, 5))
    x_indices = np.arange(len(plot_df))
    width = 0.25

    ax.bar(x_indices - width, plot_df["SHAP_Rank"], width, label="SHAP Rank", color="#2b5c8f", edgecolor="black")
    ax.bar(x_indices, plot_df["LIME_Rank"], width, label="LIME Rank", color="#d95f02", edgecolor="black")
    ax.bar(x_indices + width, plot_df["Permutation_Rank"], width, label="Permutation Rank", color="#7570b3", edgecolor="black")

    ax.set_ylabel("Assigned Rank (Lower = More Important)", fontsize=11)
    ax.set_title("Top-10 Features: Disagreement in Assigned Ranks Across XAI Methods", fontsize=12, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(plot_df["Feature"], rotation=40, ha="right", fontsize=9)
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "agreement_top_features_comparison.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig1_path}")

    # Figure 2: Pairwise Rank Correlation Heatmap
    corr_matrix_methods = pd.DataFrame(
        [
            [1.0, df_agreement.loc[0, "Kendall_Tau"], df_agreement.loc[1, "Kendall_Tau"]],
            [df_agreement.loc[0, "Kendall_Tau"], 1.0, df_agreement.loc[2, "Kendall_Tau"]],
            [df_agreement.loc[1, "Kendall_Tau"], df_agreement.loc[2, "Kendall_Tau"], 1.0],
        ],
        index=["SHAP", "LIME", "Permutation"],
        columns=["SHAP", "LIME", "Permutation"],
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix_methods, annot=True, fmt=".3f", cmap="Blues", cbar=True, vmin=0.0, vmax=1.0)
    plt.title("Inter-Method Agreement Matrix (Kendall's Tau)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "agreement_rank_correlation_heatmap.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig2_path}")

    # Figure 3: Hypothesis Investigation Graph
    plt.figure(figsize=(7, 4.5))
    plt.plot(
        df_hypothesis["LIME_Perturbation_Samples"],
        df_hypothesis["Kendall_Tau_with_SHAP"],
        marker="o",
        linewidth=2.2,
        color="#2b5c8f",
        label="Agreement with SHAP (Tau)"
    )
    plt.plot(
        df_hypothesis["LIME_Perturbation_Samples"],
        df_hypothesis["Kendall_Tau_with_Permutation"],
        marker="s",
        linewidth=2.2,
        color="#7570b3",
        label="Agreement with Permutation (Tau)"
    )
    plt.title("Hypothesis Test: Effect of LIME Perturbation Sample Size on Agreement", fontsize=11, fontweight="bold")
    plt.xlabel("LIME Perturbation Sample Size (Surrogate Quality)", fontsize=10)
    plt.ylabel("Kendall's Tau Agreement", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "agreement_hypothesis_disagreement_analysis.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig3_path}")

    print("\nAgreement Results Summary:")
    print("--------------------------------------------------------------------------------")
    print(df_agreement.to_string(index=False))
    print("--------------------------------------------------------------------------------")
    print("Agreement experiment successfully completed.\n")


if __name__ == "__main__":
    run_agreement_experiment()

