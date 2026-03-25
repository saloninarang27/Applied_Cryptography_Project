"""
================================================================================
SIMON EXPERIMENT RUNNER
================================================================================

OVERVIEW:
This script is the main entry point for Machine Learning experiments on the 
SIMON 32/64 block cipher. SIMON is a lightweight cipher designed by the NSA. 
Because of its simple Feistel structure and bitwise operations, it is a 
classic target for AI-based cryptanalysis research.

THE PIPELINE:
1. DATASET GENERATION:
   - Generates plaintext-ciphertext pairs for SIMON across rounds 1 to 5.
2. MULTI-MODEL COMPETITION:
   - We run five different AI architectures (Logistic, MLP, CNN, MINE, RF).
   - We look for the "cut-off" round—the point where the models can no 
     longer distinguish SIMON's output from random noise.
3. DETAILED LOGGING:
   - This runner provides more verbose output than others, listing all 
     generated files and saved metrics paths.

SIMON SPECIFIC CONTEXT:
SIMON is a Feistel cipher. In each round, only half of the bits are modified. 
This means the models might find patterns more easily than in a non-Feistel 
cipher like Speck or LED.
================================================================================
"""

import os
import sys

# Project Path Setup.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.generate_dataset import save_dataset
from models.logistic_regression import run_experiment as run_logistic, plot_results as plot_logistic
from models.mlp import run_experiment as run_mlp
from models.cnn import run_experiment as run_cnn
from models.mine import run_experiment as run_mine
from models.random_forest import run_experiment as run_random_forest
from models.common import save_summary
from results.comparison_report import generate_report
from results.generate_all_plots import generate_all_plots

# Experiment Parameters.
ROUNDS = [1, 2, 3, 4, 5]
CIPHER = "simon"
MODELS = {
    "logistic": run_logistic,
    "mlp": run_mlp,
    "cnn": run_cnn,
    "mine": run_mine,
    "random_forest": run_random_forest,
}


def main():
    """
    Main entry point for SIMON 32/64 analysis.
    """
    print("=" * 55)
    print("  SIMON 32/64  x  Logistic, MLP, CNN, MINE, RF  x  r = 1,2,3,4,5")
    print("=" * 55)

    # STEP 1: Generate the Bitwise Datasets.
    print(f"\n[Step 1] Generating datasets  ->  data/{CIPHER}/")
    for round_id in ROUNDS:
        save_dataset(CIPHER, round_id, base_dir="data")

    # STEP 2: Train the Competitive Models.
    print("\n[Step 2] Training all models ...")
    all_model_results = {}
    for model_name, runner in MODELS.items():
        model_results = {}
        for round_id in ROUNDS:
            # Runner returns a dict of metrics for this specific model/round.
            model_results[round_id] = runner(round_id, data_dir="data")
        all_model_results[model_name] = model_results
        
        # Save summary JSON for this specific model architecture.
        save_summary(model_results, CIPHER, model_name)

    # STEP 3: Initial Visualization.
    # We generate a quick plot for the baseline model (Logistic) first.
    plot_logistic(all_model_results["logistic"], CIPHER, "logistic")
    
    # Then generate plots for all other models.
    generate_all_plots(cipher=CIPHER, models=list(MODELS.keys()))

    # STEP 4: Final Comparison Report.
    print("\n[Step 3] Saving comparison report ...")
    generate_report(cipher=CIPHER, models=list(MODELS.keys()), rounds=ROUNDS)

    # STEP 5: File Summary.
    # List all the artifacts created during the run.
    print("\n" + "=" * 55)
    print("  Execution Summary (Files Created):")
    print("=" * 55)
    
    # 5a. List Data.
    for root, _, files in os.walk(f"data/{CIPHER}"):
        for filename in sorted(files):
            rel = os.path.relpath(os.path.join(root, filename))
            print(f"    {rel}")

    # 5b. List Metrics.
    metrics_dir = os.path.join("results", "metrics")
    if os.path.isdir(metrics_dir):
        for filename in sorted(os.listdir(metrics_dir)):
            if filename.startswith(f"{CIPHER}_"):
                print(f"    results/metrics/{filename}")

    # 5c. List Plots.
    plots_dir = os.path.join("results", "plots")
    if os.path.isdir(plots_dir):
        for filename in sorted(os.listdir(plots_dir)):
            if filename.startswith(f"{CIPHER}_"):
                print(f"    results/plots/{filename}")

    print("\nDone!")


if __name__ == "__main__":
    main()
