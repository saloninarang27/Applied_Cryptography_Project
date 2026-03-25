"""
================================================================================
ASCON EXPERIMENT RUNNER
================================================================================

OVERVIEW:
This script executes Machine Learning experiments on the ASCON-style 
permutation mapping. ASCON is the NIST standard for lightweight 
cryptography, chosen for its efficiency and security on small devices.

ASCON CONTEXT:
ASCON is a permutation-based cipher. Unlike AES, which uses blocks and 
keys in a fixed structure, ASCON uses a 320-bit internal state that 
is repeatedly scrambled. This experiment tests how well modern AI models 
can approximate the resulting output mapping after 1 to 5 scrambling rounds.

THE PIPELINE:
1. DATASET GENERATION:
   - Creates ASCON datasets for rounds 1-5.
2. MULTI-MODEL ANALYSIS:
   - Tests five different AI architectures (Logistic, MLP, CNN, MINE, RF).
   - This variety allows us to see which architecture is best at 
     detecting the specific bitwise patterns created by ASCON.
3. VISUALIZATION & COMPARISON:
   - Produces detailed plots and a comprehensive JSON report.
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

# Experiment Parameters.
ROUNDS = [1, 2, 3, 4, 5]
CIPHER = "ascon"
MODELS = {
    "logistic": run_logistic,
    "mlp": run_mlp,
    "cnn": run_cnn,
    "mine": run_mine,
    "random_forest": run_random_forest,
}


def main():
    """
    Main orchestration loop for the ASCON experiment.
    """
    print("=" * 65)
    print("  ASCON (reduced-round mapping)  x  Logistic, MLP, CNN, MINE, RF  x  r = 1,2,3,4,5")
    print("=" * 65)

    # STEP 1: Generate Datasets.
    print(f"\n[Step 1] Generating datasets  ->  data/{CIPHER}/")
    for r in ROUNDS:
        save_dataset(CIPHER, r, base_dir="data")

    # STEP 2: Train & Evaluate Models.
    print("\n[Step 2] Training all models ...")
    for model_name, runner in MODELS.items():
        model_results = {}
        for r in ROUNDS:
            model_results[r] = runner(r, data_dir="data", cipher=CIPHER)
        save_summary(model_results, CIPHER, model_name)

    # STEP 3: Generate Analysis ArtifactS.
    print("\n[Step 3] Generating plots + comparison report ...")
    generate_all_plots(cipher=CIPHER, models=list(MODELS.keys()))
    generate_report(cipher=CIPHER, models=list(MODELS.keys()), rounds=ROUNDS)

    print("\nExperiment Complete!")


if __name__ == "__main__":
    main()
