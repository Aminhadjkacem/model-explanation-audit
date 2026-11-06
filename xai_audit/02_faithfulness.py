"""02_faithfulness.py: Experiment 2 – Faithfulness (Feature Removal / Replacement).

Academic Question:
-------------------
Does removing or replacing the features identified as "most important" by an
explanation method actually cause a significant change in the model's predicted
probabilities?

Background & Methodology:
-------------------------
Faithfulness (also referred to as Fidelity or Monotonicity in XAI literature)
measures whether the explanation reflects the true underlying decision logic
of the machine learning model. If a feature is ranked as highly important for a
loan applicant, replacing or perturbing that feature should drastically shift
the predicted risk probability P(RiskPerformance = Bad).

Experimental Design:
--------------------
1. Select a representative test sample of instances from X_test.
2. For each instance and each explanation method (SHAP, LIME, Permutation Importance):
   - Determine the individual feature ranking from top-1 to top-k (k in [1, 2, 3, 5, 8, 10, 15]).
3. Feature Replacement Protocol (Two Background Strategies):
   - Strategy 1: Replace top-k features with the Training-set Median (neutral point replacement).
   - Strategy 2: Replace top-k features with Randomly Sampled values from the training distribution
     (empirical marginal distribution sampling, averaged over Monte Carlo iterations).
4. Measure the Mean Absolute Probability Degradation:
   Delta_k = (1 / N) * sum(| P(y=1 | x) - P(y=1 | x_replaced_k) |)
   A more faithful explanation method will cause a steeper, faster drop or change in prediction
   as top features are masked compared to a random feature replacement baseline.
5. Compute Rank Correlation (Kendall's Tau) between the degradation curves and theoretical rankings.

Outputs:
--------
- CSV tables in `results/tables/`:
    * faithfulness_degradation_curves.csv
    * faithfulness_summary_metrics.csv
- Figures in `results/figures/`:
    * faithfulness_replacement_median.png
    * faithfulness_replacement_random.png
    * faithfulness_strategy_comparison.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau

from xai_audit.utils import (
    get_base_pipeline,
    compute_shap_local,
    create_lime_explainer,
    compute_lime_local,
    compute_permutation_importance,
    TABLES_DIR,
    FIGURES_DIR,
)


def run_faithfulness_experiment():
    """Executes Experiment 2: Feature replacement faithfulness evaluation."""
    print("=" * 80)
    print("EXPERIMENT 2: FAITHFULNESS (Feature Removal / Replacement Curves)")
    print("=" * 80)

    # 1. Load pipeline and data
    bb, X_train, y_train, X_test, y_test, feature_names = get_base_pipeline()
    lime_explainer = create_lime_explainer(X_train)

    # Median background reference
    train_medians = X_train.median()

    # Pre-compute global permutation importance as the benchmark rank order for Permutation
    print("\n[Step 1/4] Pre-computing reference rankings for test samples...")
    global_perm = compute_permutation_importance(bb, X_test.sample(200, random_state=42), y_test.sample(200, random_state=42))
    perm_ranking = list(global_perm.sort_values(ascending=False).index)

    # Select representative test instances (e.g. 100 diverse test samples)
    eval_instances = X_test.sample(n=100, random_state=42).copy()
    original_probs = bb.predict_proba(eval_instances)[:, 1]

    # Precompute local rankings for each sample under SHAP, LIME, and Permutation
    sample_rankings = {"SHAP": [], "LIME": [], "Permutation": [], "Random_Baseline": []}

    print("  Calculating individual sample rankings for SHAP and LIME...")
    rng = np.random.RandomState(42)
    for idx in range(len(eval_instances)):
        row = eval_instances.iloc[idx]
        
        # SHAP local ranking (by absolute magnitude of contribution)
        s_vals = compute_shap_local(bb, row)
        shap_ranked = list(s_vals.abs().sort_values(ascending=False).index)
        sample_rankings["SHAP"].append(shap_ranked)

        # LIME local ranking
        l_vals = compute_lime_local(lime_explainer, bb, row, num_samples=800)
        lime_ranked = list(l_vals.abs().sort_values(ascending=False).index)
        sample_rankings["LIME"].append(lime_ranked)

        # Permutation ranking (shared global ranking)
        sample_rankings["Permutation"].append(perm_ranking)

        # Random baseline ranking (arbitrary feature ranking)
        shuffled_cols = list(rng.permutation(feature_names))
        sample_rankings["Random_Baseline"].append(shuffled_cols)

    # =========================================================================
    # Step 2: Progressive Feature Replacement
    # =========================================================================
    k_values = [1, 2, 3, 5, 8, 10, 15, 20]
    methods = ["SHAP", "LIME", "Permutation", "Random_Baseline"]
    strategies = ["Median_Replacement", "Random_Sample_Replacement"]

    results_records = []

    print("\n[Step 2/4] Testing feature replacement curves across both background strategies...")

    for strategy in strategies:
        print(f"\n  Running strategy: {strategy}...")
        for method in methods:
            degradations_per_k = []
            signed_shifts_per_k = []

            for k in k_values:
                abs_diffs = []
                signed_diffs = []

                for i in range(len(eval_instances)):
                    row = eval_instances.iloc[i].copy()
                    top_k_features = sample_rankings[method][i][:k]
                    orig_p = original_probs[i]

                    if strategy == "Median_Replacement":
                        # Direct median replacement
                        perturbed_row = row.copy()
                        for feat in top_k_features:
                            perturbed_row[feat] = train_medians[feat]
                        p_new = bb.predict_proba(perturbed_row.to_frame().T)[0, 1]
                        abs_diffs.append(abs(orig_p - p_new))
                        signed_diffs.append(orig_p - p_new)

                    elif strategy == "Random_Sample_Replacement":
                        # Monte Carlo average over 5 random draws from the empirical training distribution
                        mc_diffs = []
                        mc_signed = []
                        for draw in range(5):
                            perturbed_row = row.copy()
                            random_donor = X_train.sample(n=1, random_state=42 + draw + i * 10).iloc[0]
                            for feat in top_k_features:
                                perturbed_row[feat] = random_donor[feat]
                            p_draw = bb.predict_proba(perturbed_row.to_frame().T)[0, 1]
                            mc_diffs.append(abs(orig_p - p_draw))
                            mc_signed.append(orig_p - p_draw)
                        abs_diffs.append(np.mean(mc_diffs))
                        signed_diffs.append(np.mean(mc_signed))

                mean_abs_diff = float(np.mean(abs_diffs))
                std_abs_diff = float(np.std(abs_diffs))
                
                results_records.append({
                    "Strategy": strategy,
                    "Method": method,
                    "Top_k_Replaced": k,
                    "Mean_Absolute_Probability_Change": round(mean_abs_diff, 4),
                    "Std_Absolute_Probability_Change": round(std_abs_diff, 4),
                })

    df_degradation = pd.DataFrame(results_records)
    csv_deg_path = os.path.join(TABLES_DIR, "faithfulness_degradation_curves.csv")
    df_degradation.to_csv(csv_deg_path, index=False)
    print(f"\n[Step 3/4] Saved degradation curves to: {csv_deg_path}")

    # Compute Summary Faithfulness Metrics: Area Under Degradation Curve (AUDC) and Kendall's Tau Monotonicity
    summary_metrics = []
    for (strategy, method), group in df_degradation.groupby(["Strategy", "Method"]):
        group_sorted = group.sort_values("Top_k_Replaced")
        k_arr = group_sorted["Top_k_Replaced"].values
        deg_arr = group_sorted["Mean_Absolute_Probability_Change"].values
        # AUDC: higher indicates that removing fewer top features causes larger immediate impact
        # In NumPy 2.0+, np.trapezoid replaces np.trapz
        if hasattr(np, "trapezoid"):
            integral_val = np.trapezoid(deg_arr, k_arr)
        else:
            integral_val = getattr(np, "trapz", lambda y, x: np.sum(y))(deg_arr, k_arr)
        audc = float(integral_val / (k_arr[-1] - k_arr[0]))
        # Monotonicity: Kendall Tau between k and degradation (should be positive and close to +1.0)
        tau, _ = kendalltau(k_arr, deg_arr)
        
        summary_metrics.append({
            "Strategy": strategy,
            "Method": method,
            "Mean_Change_At_Top1": round(deg_arr[0], 4),
            "Mean_Change_At_Top5": round(deg_arr[3], 4),
            "Mean_Change_At_Top10": round(deg_arr[5], 4),
            "AUDC_Score": round(audc, 4),
            "Monotonicity_Kendall_Tau": round(float(tau), 4),
        })

    df_summary = pd.DataFrame(summary_metrics)
    csv_sum_path = os.path.join(TABLES_DIR, "faithfulness_summary_metrics.csv")
    df_summary.to_csv(csv_sum_path, index=False)
    print(f"Saved faithfulness summary metrics to: {csv_sum_path}")

    # =========================================================================
    # Step 4: Visualizations
    # =========================================================================
    palette = {
        "SHAP": "#2b5c8f",
        "LIME": "#d95f02",
        "Permutation": "#7570b3",
        "Random_Baseline": "#999999",
    }
    markers = {"SHAP": "o", "LIME": "s", "Permutation": "^", "Random_Baseline": "x"}

    for strategy in strategies:
        plt.figure(figsize=(8.5, 5))
        strat_df = df_degradation[df_degradation["Strategy"] == strategy]
        
        for method in methods:
            m_df = strat_df[strat_df["Method"] == method].sort_values("Top_k_Replaced")
            plt.plot(
                m_df["Top_k_Replaced"],
                m_df["Mean_Absolute_Probability_Change"],
                marker=markers[method],
                linewidth=2.2,
                color=palette[method],
                label=f"{method} (AUDC: {df_summary[(df_summary['Strategy']==strategy) & (df_summary['Method']==method)]['AUDC_Score'].values[0]:.3f})"
            )
        
        strat_clean = strategy.replace("_", " ")
        plt.title(f"Faithfulness: Feature Removal Impact ({strat_clean})", fontsize=12, fontweight="bold")
        plt.xlabel("Number of Top Important Features Replaced (k)", fontsize=11)
        plt.ylabel("Mean Absolute Change in Predicted Risk |P(Bad) - P_new|", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True, loc="lower right")
        plt.tight_layout()
        
        fname = f"faithfulness_{strategy.lower()}.png"
        fig_path = os.path.join(FIGURES_DIR, fname)
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved figure: {fig_path}")

    print("\n[Step 4/4] Faithfulness Results Summary:")
    print("--------------------------------------------------------------------------------")
    print(df_summary.to_string(index=False))
    print("--------------------------------------------------------------------------------")
    print("Faithfulness experiment successfully completed.\n")


if __name__ == "__main__":
    run_faithfulness_experiment()

