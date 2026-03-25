"""
================================================================================
LOGISTIC REGRESSION FOR CRYPTOGRAPHIC ANALYSIS
================================================================================

OVERVIEW:
This module provides a Logistic Regression baseline for analyzing reduced-round 
ciphers. While neural networks (CNNs, MLPs) are powerful, Logistic Regression 
is a vital baseline because it represents the "Linear" limit. If a cipher can 
be predicted by Logistic Regression, it means its diffusion and confusion 
layers are not yet providing strong mathematical security.

HOW IT WORKS:
1. BINARY CLASSIFICATION PER BIT:
   - We train a separate Logistic Regression model for every single bit of 
     the ciphertext.
   - For a 64-bit cipher, we train 64 independent models.
2. LINEAR DECISION BOUNDARY:
   - The model tries to find a linear "cut" in the input bit space that 
     best separates 0s from 1s in the output.
3. FEATURE AUGMENTATION:
   - To give the linear model a "fighting chance" against non-linear 
     ciphers, we use 'augment_plaintext_features' to add non-linear terms 
     (like bitwise XORs) to the input.
4. CONSTANT BIT DETECTION:
   - For very low rounds, some ciphertext bits might always be 0 or 1. 
     The module detects this and uses a "constant predictor" instead of 
     trying to fit a regression model.

WHY USE THIS?
- SPEED: It is significantly faster than training neural networks.
- INTERPRETABILITY: If bitwise accuracy is high, we know the cipher is 
  mathematically "shallow" at that round level.
- BASELINE: It provides the "floor" for performance. Any model that 
  doesn't beat Logistic Regression is not learning complex features.

THIS MODULE:
- Implements the multi-bit training and prediction logic.
- Includes a sophisticated plotting engine to visualize results over rounds.
- Automatically handles data scaling and feature engineering.
================================================================================
"""

import numpy as np
import json
import os
import sys
import time
import matplotlib.pyplot as plt

# Add project root to path.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data.generate_dataset import load_dataset, save_dataset
from models.common import augment_plaintext_features

# Global experiment settings.
MODEL_NAME = "logistic"
MAX_TRAIN_SAMPLES = 140_000
MAX_TEST_SAMPLES = 30_000


def _fast_mode_enabled() -> bool:
    """Checks if the fast execution flag is set."""
    return os.getenv("AC_FAST_MODE", "0") == "1"


# ------------------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------------------

def train_logistic(X_train, y_train):
    """
    Trains one Logistic Regression model for each bit in the ciphertext.
    """
    block_bits = y_train.shape[1]
    models = []
    
    for bit in range(block_bits):
        y_bit = y_train[:, bit]
        unique_classes = np.unique(y_bit)

        # Handle bits that are constant (always 0 or always 1).
        if unique_classes.size < 2:
            models.append(int(unique_classes[0]))
            continue

        # Standard Logistic Regression with L2 regularization.
        clf = LogisticRegression(
            max_iter=900, 
            solver="lbfgs", 
            C=1.5, 
            class_weight="balanced", 
            random_state=42
        )
        clf.fit(X_train, y_bit)
        models.append(clf)
        
    return models


def predict(models, X):
    """
    Generates full ciphertext predictions by querying each bit-model.
    """
    preds = []
    for m in models:
        if isinstance(m, (int, np.integer)):
            # Use the constant value if no model was needed.
            preds.append(np.full(X.shape[0], int(m), dtype=np.uint8))
        else:
            preds.append(m.predict(X).astype(np.uint8))
    # Stack bit predictions into a full block matrix.
    return np.stack(preds, axis=1).astype(np.uint8)


def compute_metrics(y_true, y_pred):
    """Calculates accuracy and error metrics."""
    return {
        "bitwise_accuracy": float((y_true == y_pred).mean()),
        "hamming_distance": float((y_true != y_pred).sum(axis=1).mean()),
        "word_accuracy":    float(np.all(y_true == y_pred, axis=1).mean()),
    }


# ------------------------------------------------------------------------------
# DATA & PERSISTENCE
# ------------------------------------------------------------------------------

def save_metrics(metrics, cipher, model, num_rounds, out_dir="results/metrics"):
    """Saves the results of a single experiment to a JSON file."""
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{cipher}_{model}_r{num_rounds}.json"
    payload = {"cipher": cipher, "model": model, "rounds": num_rounds, **metrics}
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Metrics saved  ->  {out_dir}/{fname}")


def save_summary(all_results, cipher, model, out_dir="results/metrics"):
    """Saves a summary of multiple round experiments."""
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{cipher}_{model}_summary.json"
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Summary saved  ->  {out_dir}/{fname}")


# ------------------------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------------------------

def plot_results(all_results, cipher, model, out_dir="results/plots"):
    """
    Generates high-quality charts showing how model performance 
    changes as the number of cipher rounds increases.
    """
    os.makedirs(out_dir, exist_ok=True)

    rounds = sorted(all_results.keys())
    bit_acc = [all_results[r]["bitwise_accuracy"] * 100 for r in rounds]
    hamming = [all_results[r]["hamming_distance"] for r in rounds]
    block_bits = int(all_results[rounds[0]].get("block_bits", 32)) if rounds else 32
    random_hamming = block_bits / 2.0

    # Dynamic scaling for the Y-axis to ensure details are visible.
    def _tight_accuracy_ylim(values, random_ref=50.0):
        if not values: return (40.0, 100.0)
        lo, hi = min(values), max(values)
        padding = max(1.0, (hi - lo) * 0.1)
        return (max(0, lo - padding), min(100, hi + padding))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{cipher.upper()}  x  {model.capitalize()} Regression", fontsize=14, fontweight="bold")

    # Chart 1: Bitwise Accuracy
    ax = axes[0]
    ax.plot(rounds, bit_acc, "o-", color="#2196F3", lw=2.5, ms=9, label="Model")
    ax.axhline(50, ls="--", color="red", lw=1.5, label="Random (50%)")
    ax.fill_between(rounds, 50, bit_acc, alpha=0.1, color="#2196F3")
    ax.set(xlabel="Rounds (r)", ylabel="Bitwise Accuracy (%)", title="Accuracy vs Complexity")
    ax.legend(); ax.grid(alpha=0.3)

    # Chart 2: Hamming Distance
    ax = axes[1]
    ax.plot(rounds, hamming, "s-", color="#E91E63", lw=2.5, ms=9, label="Model")
    ax.axhline(random_hamming, ls="--", color="red", lw=1.5, label=f"Random ({random_hamming:.1f})")
    ax.fill_between(rounds, hamming, random_hamming, alpha=0.1, color="#E91E63")
    ax.set(xlabel="Rounds (r)", ylabel="Hamming Distance (bits)", title="Hamming Distance vs Complexity")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cipher}_{model}_results.png"), dpi=150)
    print(f"  Plot saved     ->  {out_dir}")
    plt.show()


# ------------------------------------------------------------------------------
# EXPERIMENT RUNNER
# ------------------------------------------------------------------------------

def run_experiment(num_rounds, data_dir="data", cipher="simon"):
    """
    Main entry point for running a logistic regression test.
    """
    print(f"\n{'='*55}")
    print(f"  {cipher.upper()}  |  Logistic Regression  |  r = {num_rounds}")
    print(f"{'='*55}")

    # Load the specific dataset for this cipher/round combination.
    try:
        X, y = load_dataset(cipher, num_rounds, base_dir=data_dir)
    except FileNotFoundError:
        X, y = save_dataset(cipher, num_rounds, base_dir=data_dir)

    # Split into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply caps to keep runtime manageable.
    train_cap = 8_000 if _fast_mode_enabled() else MAX_TRAIN_SAMPLES
    test_cap = 2_000 if _fast_mode_enabled() else MAX_TEST_SAMPLES
    
    X_train, y_train = X_train[:train_cap], y_train[:train_cap]
    X_test, y_test = X_test[:test_cap], y_test[:test_cap]

    # Feature engineering.
    X_train = augment_plaintext_features(X_train, cipher=cipher)
    X_test = augment_plaintext_features(X_test, cipher=cipher)
    
    # Scale features.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    block_bits = int(y_train.shape[1])
    random_hamming = block_bits / 2.0

    # Train and time the process.
    t0 = time.time()
    models = train_logistic(X_train, y_train)
    t_train = time.time() - t0

    # Evaluate.
    y_pred = predict(models, X_test)
    metrics = compute_metrics(y_test, y_pred)
    metrics["block_bits"] = block_bits
    metrics["train_time_s"] = round(t_train, 2)

    print(f"  Train time        : {t_train:.1f}s")
    print(f"  Bitwise accuracy  : {metrics['bitwise_accuracy']*100:.2f}%")

    save_metrics(metrics, cipher, MODEL_NAME, num_rounds)
    return metrics


if __name__ == "__main__":
    # Standard sweep from 1 to 5 rounds.
    all_results = {}
    for r in [1, 2, 3, 4, 5]:
        all_results[r] = run_experiment(r, cipher="simon")

    save_summary(all_results, "simon", MODEL_NAME)
    plot_results(all_results, "simon", MODEL_NAME)
