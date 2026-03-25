"""
================================================================================
GRAIN-128a EXPERIMENT RUNNER
================================================================================

OVERVIEW:
This script is the main entry point for running Machine Learning experiments 
on the Grain-128a stream cipher. It evaluates how well various models can 
approximate the cipher's behavior at reduced rounds.

GRAIN-128a CONTEXT:
Grain-128a is a lightweight stream cipher designed for efficient hardware 
implementation. It belongs to the eSTREAM portfolio and uses a combination 
of a Linear Feedback Shift Register (LFSR) and a Non-Linear Feedback Shift 
Register (NFSR) to generate a keystream.

THE PIPELINE:
1. DATASET GENERATION:
   - Uses the 'grain128a' module in 'ciphers/' to generate plaintext-ciphertext pairs.
   - Datasets are created for several round configurations.
2. MULTI-MODEL ANALYSIS:
   - Executes experiments for five different architectures:
     * Logistic Regression (The Linear Baseline)
     * Multi-Layer Perceptron (The Vanilla Neural Net)
     * Convolutional Neural Network (The Pattern Finder)
     * MINE (The Mutual Information Estimator)
     * Random Forest (The Ensemble Learner)
3. EVALUATION & REPORTING:
   - Collects bitwise accuracy, Hamming distance, and word accuracy.
   - Generates comparative plots and a final JSON comparison report.
================================================================================
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
CIPHER = "grain128a"
MODELS = {
    "logistic": run_logistic,
    "mlp": run_mlp,
    "cnn": run_cnn,
    "mine": run_mine,
    "random_forest": run_random_forest,
}


def main():
    print("=" * 72)
    print("  GRAIN-128A  x  Logistic, MLP, CNN, MINE, RF  x  r = 1,2,3,4,5")
    print("=" * 72)

    print(f"\n[Step 1] Generating datasets  ->  data/{CIPHER}/")
    for r in ROUNDS:
        save_dataset(CIPHER, r, base_dir="data")

    print("\n[Step 2] Training all models ...")
    for model_name, runner in MODELS.items():
        model_results = {}
        for r in ROUNDS:
            model_results[r] = runner(r, data_dir="data", cipher=CIPHER)
        save_summary(model_results, CIPHER, model_name)

    print("\n[Step 3] Generating plots + comparison report ...")
    generate_all_plots(cipher=CIPHER, models=list(MODELS.keys()))
    generate_report(cipher=CIPHER, models=list(MODELS.keys()), rounds=ROUNDS)

    print("\nDone!")


if __name__ == "__main__":
    main()
