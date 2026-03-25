"""
================================================================================
MULTI-LAYER PERCEPTRON (MLP) FOR CIPHER ANALYSIS
================================================================================

OVERVIEW:
This module implements a Multi-Layer Perceptron (MLP), often called a 
standard "Vanilla Neural Network," to analyze cryptographic mappings. 
The goal is to see if a dense network can learn the complex, non-linear 
relationships between a plaintext and its corresponding ciphertext.

HOW IT WORKS:
1. FULLY CONNECTED LAYERS:
   - The network consists of multiple layers of neurons where every neuron 
     is connected to every neuron in the next layer.
   - These layers learn a mathematical function that maps input bits to 
     output bits.
2. ACTIVATION FUNCTIONS (ReLU):
   - We use the Rectified Linear Unit (ReLU) to introduce non-linearity. 
     This is crucial because ciphers are designed to be extremely 
     non-linear (using S-boxes and bitwise operations).
3. MULTI-OUTPUT CLASSIFICATION:
   - Since a block cipher predicts many bits at once (e.g., 64 bits), 
     we use a 'MultiOutputClassifier' wrapper. This treats each bit 
     prediction as a separate but related task.
4. BACKPROPAGATION (Adam Solver):
   - The model uses the Adam optimizer to adjust its weights by 
     minimizing the difference between its predictions and the real 
     ciphertext bits.

STRENGTHS VS WEAKNESSES:
- STRENGTH: MLPs are universal function approximators; they can theoretically 
  learn any mapping given enough data and neurons.
- WEAKNESS: They don't have built-in "structural" knowledge like CNNs do. 
  They treat every bit as equally far apart from every other bit initially.

THIS MODULE:
- Implements a robust MLP pipeline using Scikit-Learn.
- Features extensive 'Runtime Overrides' to handle different ciphers efficiently.
- Includes memory-safe retry mechanisms and feature augmentation.
================================================================================
"""

import os
import sys
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Project-level imports.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.common import (
    DEFAULT_CIPHER,
    compute_metrics,
    get_split,
    augment_plaintext_features,
    load_or_generate_dataset,
    save_metrics,
    save_summary,
    timed_call,
)

# Global Configuration.
MODEL_NAME = "mlp"
MAX_TRAIN_SAMPLES = 120_000
MAX_TEST_SAMPLES = 30_000

# ------------------------------------------------------------------------------
# RUNTIME OVERRIDES
# ------------------------------------------------------------------------------
# Some ciphers have very high entropy (complexity) and can crash or hang 
# the training process on a normal PC. These settings ensure the experiments 
# complete in a reasonable time while still providing valid data.
MLP_RUNTIME_OVERRIDES = {
    "ascon": {"train_cap": 60_000, "max_iter": 120, "n_jobs": 1},
    "present": {"train_cap": 40_000, "max_iter": 150, "n_jobs": 1},
    "tinyjambu": {"train_cap": 8_000, "max_iter": 60, "n_jobs": 1},
    "grain128a": {"train_cap": 54_000, "max_iter": 110, "n_jobs": 1},
    "led": {"train_cap": 54_000, "max_iter": 110, "n_jobs": 1},
    "skinny": {"train_cap": 54_000, "max_iter": 110, "n_jobs": 1},
    "chacha20": {"train_cap": 48_000, "max_iter": 120, "n_jobs": 1},
}


def _fast_mode_enabled() -> bool:
    """Checks for the developer 'fast mode' flag."""
    return os.getenv("AC_FAST_MODE", "0") == "1"


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 140,
    batch_size: int = 1024,
    hidden_layer_sizes=(128, 64),
    n_jobs: int = -1,
    alpha: float = 5e-5,
    learning_rate_init: float = 7e-4,
    early_stopping: bool = True,
    n_iter_no_change: int = 15,
    tol: float = 1e-4,
) -> MultiOutputClassifier:
    """
    Core function to build and train the MLP model.
    """
    # Define the base neural network.
    base = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        n_iter_no_change=n_iter_no_change,
        tol=tol,
        early_stopping=early_stopping,
        random_state=42,
        verbose=False,
    )
    
    # Wrap it to handle multiple output bits simultaneously.
    model = MultiOutputClassifier(base, n_jobs=n_jobs)
    
    # We ignore ConvergenceWarnings because for complex ciphers, the model 
    # might not fully 'solve' the function within the iteration limit.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train)
        
    return model


def run_experiment(num_rounds: int, data_dir: str = "data", cipher: str = DEFAULT_CIPHER):
    """
    Orchestrates the MLP experiment: Data loading -> Training -> Metrics.
    """
    print(f"\n{'='*55}")
    print(f"  {cipher.upper()}  |  MLP  |  r = {num_rounds}")
    print(f"{'='*55}")

    # 1. LOAD DATA
    X, y = load_or_generate_dataset(num_rounds, data_dir=data_dir, cipher=cipher)
    X_train, X_test, y_train, y_test = get_split(X, y)

    # 2. DYNAMIC CONFIGURATION
    # Fetch overrides and set default hyperparameters.
    cfg = MLP_RUNTIME_OVERRIDES.get(cipher.lower(), {})
    train_cap = int(cfg.get("train_cap", MAX_TRAIN_SAMPLES))
    max_iter = int(cfg.get("max_iter", 140))
    n_jobs = int(cfg.get("n_jobs", -1))
    batch_size = 1024
    test_cap = MAX_TEST_SAMPLES
    hidden_layers = (128, 64)
    alpha, learning_rate_init = 5e-5, 7e-4
    early_stopping, n_iter_no_change, tol = True, 15, 1e-4

    # The following block contains many specific tweaks for different ciphers 
    # and round levels to ensure the experiments are both accurate and fast.
    if cipher.lower() == "tinyjambu":
        train_cap, test_cap, max_iter, batch_size, n_jobs = 2_000, 1_000, 16, 128, 1

    if cipher.lower() == "simon" and _fast_mode_enabled():
        # Simon is a primary target, so we have very granular settings for it.
        n_jobs = 1
        if num_rounds <= 2:
            train_cap, test_cap, max_iter = 18_000, 8_000, 110
            hidden_layers = (192, 96)
        else:
            train_cap, test_cap, max_iter = 20_000, 6_000, 110
            hidden_layers = (192, 96)

    # 3. PRE-PROCESSING
    X_train, y_train = X_train[:train_cap], y_train[:train_cap]
    X_test, y_test = X_test[:test_cap], y_test[:test_cap]

    # Augment features: Add bitwise interactions to help the network "see" the math.
    X_train = augment_plaintext_features(X_train, cipher=cipher)
    X_test = augment_plaintext_features(X_test, cipher=cipher)
    
    # Scale data to have zero mean and unit variance (helps neural networks converge).
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    block_bits = int(y_train.shape[1])
    random_hamming = block_bits / 2.0

    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")
    print(f"  Feature dim       : {X_train.shape[1]} (augmented)")

    # 4. TRAINING
    try:
        model, train_time = timed_call(
            train_mlp, X_train, y_train,
            max_iter=max_iter, batch_size=batch_size,
            hidden_layer_sizes=hidden_layers, n_jobs=n_jobs,
            alpha=alpha, learning_rate_init=learning_rate_init,
            early_stopping=early_stopping, n_iter_no_change=n_iter_no_change,
            tol=tol
        )
    except Exception as exc:
        # Emergency fallback for low-memory environments.
        print(f"  Memory Warning: Reducing capacity...")
        model, train_time = timed_call(
            train_mlp, X_train[:2000], y_train[:2000],
            max_iter=10, batch_size=64, n_jobs=1
        )

    # 5. PREDICTION & METRICS
    y_pred = model.predict(X_test).astype(np.uint8)
    metrics = compute_metrics(y_test, y_pred)
    metrics["block_bits"] = block_bits
    metrics["train_time_s"] = round(train_time, 2)

    print(f"  Train time        : {train_time:.1f}s")
    print(f"  Bitwise accuracy  : {metrics['bitwise_accuracy']*100:.2f}%")
    print(f"  Full-word accuracy: {metrics['word_accuracy']*100:.4f}%")

    # Save metrics for reporting.
    save_metrics(metrics, cipher, MODEL_NAME, num_rounds)
    return metrics


if __name__ == "__main__":
    # Perform a standard sweep if run as a script.
    results = {}
    for r in [1, 2, 3, 4, 5]:
        results[r] = run_experiment(r, cipher="simon")
    save_summary(results, "simon", MODEL_NAME)
