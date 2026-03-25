"""
================================================================================
PRESENT EXPERIMENT RUNNER
================================================================================

OVERVIEW:
This script is the main entry point for running Machine Learning experiments 
on the PRESENT block cipher. It evaluates how well various models can 
approximate the cipher's behavior at reduced rounds (1 to 5).

PRESENT CONTEXT:
PRESENT is an ultra-lightweight block cipher designed for extremely 
constrained environments like RFID tags. It is a Substitution-Permutation 
Network (SPN) and is widely used as a benchmark in lightweight 
cryptography research due to its simplicity and efficiency.

THE PIPELINE:
1. DATASET GENERATION:
   - Creates datasets for PRESENT across rounds 1 to 5.
2. MULTI-MODEL COMPETITION:
   - Runs experiments for Logistic Regression, MLP, CNN, MINE, and Random Forest.
3. RESULTS CONSOLIDATION:
   - Saves metrics, plots, and a final comparison report.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.generate_dataset import save_dataset
from models.logistic_regression import run_experiment as run_logistic
from models.mlp import run_experiment as run_mlp
from models.cnn import run_experiment as run_cnn
from models.mine import run_experiment as run_mine
from models.random_forest import run_experiment as run_random_forest
from models.common import save_summary
from results.comparison_report import generate_report
from results.generate_all_plots import generate_all_plots


ROUNDS = [1, 2, 3, 4, 5]
CIPHER = "present"
MODELS = {
    "logistic": run_logistic,
    "mlp": run_mlp,
    "cnn": run_cnn,
    "mine": run_mine,
    "random_forest": run_random_forest,
}


def main():
    print("=" * 65)
    print("  PRESENT 64/80  x  Logistic, MLP, CNN, MINE, RF  x  r = 1,2,3,4,5")
    print("=" * 65)

    print(f"\n[Step 1] Generating datasets  ->  data/{CIPHER}/")
    for r in ROUNDS:
        save_dataset(CIPHER, r, base_dir="data")

    print("\n[Step 2] Training all models ...")
    all_model_results = {}
    for model_name, runner in MODELS.items():
        model_results = {}
        for r in ROUNDS:
            model_results[r] = runner(r, data_dir="data", cipher=CIPHER)
        all_model_results[model_name] = model_results
        save_summary(model_results, CIPHER, model_name)

    print("\n[Step 3] Generating plots + comparison report ...")
    generate_all_plots(cipher=CIPHER, models=list(MODELS.keys()))
    generate_report(cipher=CIPHER, models=list(MODELS.keys()), rounds=ROUNDS)

    print("\n" + "=" * 65)
    print("  Files saved:")
    print("=" * 65)
    for root, dirs, files in os.walk(f"data/{CIPHER}"):
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f))
            print(f"    {rel}")
    for root, dirs, files in os.walk("results"):
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f))
            if rel.startswith(f"comparison/{CIPHER}") or rel.startswith("metrics") or rel.startswith("plots"):
                print(f"    {rel}")

    print("\nDone!")


if __name__ == "__main__":
    main()
