"""run_all.py: Master Pipeline Runner for the Week 9 XAI Audit Assignment.

This script executes all five audit experiments sequentially:
1. Experiment 1: Sanity Checks (01_sanity.py)
2. Experiment 2: Faithfulness / Feature Removal (02_faithfulness.py)
3. Experiment 3: Local Stability / Robustness (03_stability.py)
4. Experiment 4: Inter-Method Agreement & Hypothesis Test (04_agreement.py)
5. Experiment 5: Final Synthesis & Audit Report Table (05_summary.py)

Usage:
------
From the project root (/home/amin/xAi/ODEV6):
    python -m xai_audit.run_all
"""

import importlib
import time


def main():
    print("*" * 80)
    print("STARTING FULL WEEK 9 XAI AUDIT EXECUTION PIPELINE")
    print("Target Model: HistGradientBoostingClassifier (HELOC Credit Risk)")
    print("*" * 80)

    start_time = time.time()

    # Dynamically import numeric-prefixed experiment modules
    exp1 = importlib.import_module("xai_audit.01_sanity")
    exp2 = importlib.import_module("xai_audit.02_faithfulness")
    exp3 = importlib.import_module("xai_audit.03_stability")
    exp4 = importlib.import_module("xai_audit.04_agreement")
    exp5 = importlib.import_module("xai_audit.05_summary")

    # Experiment 1
    t0 = time.time()
    exp1.run_sanity_experiment()
    print(f">> Experiment 1 completed in {time.time() - t0:.2f} seconds.\n")

    # Experiment 2
    t0 = time.time()
    exp2.run_faithfulness_experiment()
    print(f">> Experiment 2 completed in {time.time() - t0:.2f} seconds.\n")

    # Experiment 3
    t0 = time.time()
    exp3.run_stability_experiment()
    print(f">> Experiment 3 completed in {time.time() - t0:.2f} seconds.\n")

    # Experiment 4
    t0 = time.time()
    exp4.run_agreement_experiment()
    print(f">> Experiment 4 completed in {time.time() - t0:.2f} seconds.\n")

    # Experiment 5
    t0 = time.time()
    exp5.generate_summary()
    print(f">> Experiment 5 completed in {time.time() - t0:.2f} seconds.\n")

    total_time = time.time() - start_time
    print("*" * 80)
    print(f"ALL EXPERIMENTS SUCCESSFULLY EXECUTED IN {total_time:.2f} SECONDS!")
    print("All tables saved in:   results/tables/")
    print("All figures saved in:  results/figures/")
    print("*" * 80)


if __name__ == "__main__":
    main()
