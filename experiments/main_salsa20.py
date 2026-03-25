"""
================================================================================
SALSA20 EXPERIMENT RUNNER
================================================================================

OVERVIEW:
This script executes Machine Learning experiments on the Salsa20 stream 
cipher mapping. Designed by Daniel J. Bernstein, Salsa20 is a 
high-performance stream cipher that was a finalist in the eSTREAM project.

SALSA20 CONTEXT:
Salsa20 operates on a 512-bit state and uses a core permutation based on 
ARX (Addition-Rotation-XOR) operations. It is the predecessor to ChaCha20. 
This experiment evaluates how modern AI architectures can learn the 
diffusion patterns of the Salsa20 permutation at reduced rounds.

THE PIPELINE:
1. DATASET GENERATION:
   - Creates Salsa20 datasets for rounds 1-5.
2. MULTI-MODEL ANALYSIS:
   - Tests Logistic Regression, MLP, CNN, MINE, and Random Forest.
   - Compares the ability of linear vs. non-linear models to approximate 
     the cipher's output.
3. VISUALIZATION & COMPARISON:
   - Generates accuracy plots and a comprehensive JSON comparison report.
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


ROUNDS = [1, 2, 3, 4, 5]
CIPHER = "salsa20"
MODELS = {
    "logistic": run_logistic,
    "mlp": run_mlp,
    "cnn": run_cnn,
    "mine": run_mine,
    "random_forest": run_random_forest,
}


def main():
    """
    Orchestrates the Salsa20 experiment pipeline.
    """
    print("=" * 68)
    print("  SALSA20  x  Logistic, MLP, CNN, MINE, RF  x  r = 1,2,3,4,5")
    print("=" * 68)

    # STEP 1: Dataset Generation.
    print(f"\n[Step 1] Generating datasets  ->  data/{CIPHER}/")
    for r in ROUNDS:
        save_dataset(CIPHER, r, base_dir="data")

    # STEP 2: Model Training & Evaluation.
    print("\n[Step 2] Training all models ...")
    for model_name, runner in MODELS.items():
        model_results = {}
        for r in ROUNDS:
            model_results[r] = runner(r, data_dir="data", cipher=CIPHER)
        
        # Save summary for each model.
        save_summary(model_results, CIPHER, model_name)

    # STEP 3: Analysis & Visualization.
    print("\n[Step 3] Generating plots + comparison report ...")
    generate_all_plots(cipher=CIPHER, models=list(MODELS.keys()))
    generate_report(cipher=CIPHER, models=list(MODELS.keys()), rounds=ROUNDS)

    print("\nExperiment Complete!")


if __name__ == "__main__":
    main()