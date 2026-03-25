"""
================================================================================
AES-128 EXPERIMENT RUNNER
================================================================================

OVERVIEW:
This script is the main entry point for running Machine Learning experiments 
on the AES-128 (Advanced Encryption Standard) cipher. AES is the global 
standard for encryption, and this experiment tests how well different 
models can "break" or approximate its behavior at reduced rounds (1 to 5).

THE PIPELINE:
1. DATASET GENERATION:
   - Uses the 'aes' module in 'ciphers/' to generate plaintext-ciphertext pairs.
   - Datasets are created for 1, 2, 3, 4, and 5 rounds.
2. MULTI-MODEL TRAINING:
   - Executes experiments for five different architectures:
     * Logistic Regression (The Linear Baseline)
     * Multi-Layer Perceptron (The Vanilla Neural Net)
     * Convolutional Neural Network (The Pattern Finder)
     * MINE (The Mutual Information Estimator)
     * Random Forest (The Ensemble Learner)
3. EVALUATION & REPORTING:
   - Collects bitwise accuracy, Hamming distance, and word accuracy.
   - Generates comparative plots to visualize how AES security scales.
   - Produces a final JSON comparison report for all models.

AES SPECIFIC CONTEXT:
AES is highly non-linear due to its S-box and MixColumns operations. 
Even at 1 or 2 rounds, it is expected to be much harder for models to 
learn than lightweight ciphers like Simon or Speck.
================================================================================
"""

import os
import sys

# Add project root to path.
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

# Define the rounds and models for this experiment.
ROUNDS = [1, 2, 3, 4, 5]
CIPHER = "aes"
MODELS = {
    "logistic": run_logistic,
    "mlp": run_mlp,
    "cnn": run_cnn,
    "mine": run_mine,
    "random_forest": run_random_forest,
}


def main():
    """
    Orchestrates the AES experiment pipeline.
    """
    print("=" * 68)
    print("  AES-128  x  Logistic, MLP, CNN, MINE, RF  x  r = 1,2,3,4,5")
    print("=" * 68)

    # STEP 1: Dataset Generation.
    # We ensure we have fresh data for every round level.
    print(f"\n[Step 1] Generating datasets  ->  data/{CIPHER}/")
    for r in ROUNDS:
        save_dataset(CIPHER, r, base_dir="data")

    # STEP 2: Model Training.
    # We iterate through each model and run the experiment for all rounds.
    print("\n[Step 2] Training all models ...")
    for model_name, runner in MODELS.items():
        model_results = {}
        for r in ROUNDS:
            # Each 'runner' handles its own training, logging, and metric saving.
            model_results[r] = runner(r, data_dir="data", cipher=CIPHER)
        
        # Save a combined summary for this specific model architecture.
        save_summary(model_results, CIPHER, model_name)

    # STEP 3: Analysis & Visualization.
    # Generate charts and a final comparison report.
    print("\n[Step 3] Generating plots + comparison report ...")
    generate_all_plots(cipher=CIPHER, models=list(MODELS.keys()))
    generate_report(cipher=CIPHER, models=list(MODELS.keys()), rounds=ROUNDS)

    print("\nExperiment Complete!")


if __name__ == "__main__":
    main()
