"""
================================================================================
PRINCE EXPERIMENT RUNNER
================================================================================

OVERVIEW:
This script is the main entry point for running Machine Learning experiments 
on the PRINCE block cipher. It evaluates how well various models can 
approximate the cipher's behavior at reduced rounds (1 to 5).

PRINCE CONTEXT:
PRINCE is a low-latency block cipher designed for high-speed hardware 
implementations. It is known for its "FX-construction" and the 
$\alpha$-reflection property, which allows the same circuit to be used 
for both encryption and decryption with minimal overhead.

THE PIPELINE:
1. DATASET GENERATION:
   - Generates PRINCE datasets for rounds 1 to 5.
2. MULTI-MODEL ANALYSIS:
   - Tests Logistic Regression, MLP, CNN, MINE, and Random Forest.
3. VISUALIZATION:
   - Produces plots and reports to analyze model performance.
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
CIPHER = "prince"
MODELS = {
    "logistic": run_logistic,
    "mlp": run_mlp,
    "cnn": run_cnn,
    "mine": run_mine,
    "random_forest": run_random_forest,
}


def main():
    print("=" * 65)
    print("  PRINCE  x  Logistic, MLP, CNN, MINE, RF  x  r = 1,2,3,4,5")
    print("=" * 65)

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
