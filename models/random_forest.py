"""
================================================================================
RANDOM FOREST CLASSIFIER FOR CIPHER ANALYSIS
================================================================================

OVERVIEW:
This module implements a Random Forest Classifier to analyze reduced-round 
ciphers. Random Forest is an 'Ensemble Learning' method that builds many 
individual Decision Trees and merges their predictions to get a more 
accurate and stable result.

HOW IT WORKS:
1. DECISION TREES:
   - A Decision Tree splits the bitstream data by asking questions (e.g., 
     "Is bit 5 equal to 1?").
   - By combining many of these splits, the tree can approximate complex 
     logical functions.
2. BAGGING (Bootstrap Aggregating):
   - We train each tree on a random subset of the data. 
   - This ensures that the trees are different from each other and prevents 
     the model from simply 'memorizing' the training data (overfitting).
3. FEATURE SUBSETTING:
   - Each time a tree makes a split, it only looks at a random subset of 
     the input bits. This is crucial for ciphers, as it forces the model 
     to find multiple different ways to predict the output.
4. MULTI-OUTPUT HANDLING:
   - Scikit-learn's Random Forest can natively handle multiple output bits 
     simultaneously by finding splits that are good for the entire 
     ciphertext block.

STRENGTHS:
- Extremely robust to noise and outliers.
- Can capture complex non-linear interactions without needing a specific 
  mathematical model (unlike Logistic Regression).
- Naturally handles binary data (bits) very well.

THIS MODULE:
- Implements a parallelized Random Forest pipeline.
- Includes specific optimizations for high-entropy ciphers like AES and PRESENT.
- Uses feature augmentation to help the trees find cryptographic patterns.
================================================================================
"""

import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Add project root to path.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.common import (
    DEFAULT_CIPHER,
    compute_metrics,
    get_split,
    infer_block_bits,
    augment_plaintext_features,
    load_or_generate_dataset,
    save_metrics,
    save_summary,
    timed_call,
)

# Global Configuration.
MODEL_NAME = "random_forest"
MAX_TRAIN_SAMPLES = 70_000
MAX_TEST_SAMPLES = 15_000


def _fast_mode_enabled() -> bool:
    """Checks if the project is running in fast-testing mode."""
    return os.getenv("AC_FAST_MODE", "0") == "1"


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 180,
    max_depth = None,
    min_samples_leaf: int = 2,
) -> RandomForestClassifier:
    """
    Core function to initialize and train the Random Forest.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,    # Number of trees in the forest.
        max_depth=max_depth,          # How deep each tree can go.
        min_samples_leaf=min_samples_leaf, # Minimum samples required at a leaf node.
        max_features="sqrt",          # Number of features to consider at each split.
        random_state=42,              # For reproducible results.
        n_jobs=-1,                    # Use all available CPU cores.
    )
    model.fit(X_train, y_train)
    return model


def run_experiment(num_rounds: int, data_dir: str = "data", cipher: str = DEFAULT_CIPHER):
    """
    Executes a Random Forest experiment for a given cipher and round count.
    """
    print(f"\n{'='*55}")
    print(f"  {cipher.upper()}  |  Random Forest  |  r = {num_rounds}")
    print(f"{'='*55}")

    # 1. DATA PREPARATION
    X, y = load_or_generate_dataset(num_rounds, data_dir=data_dir, cipher=cipher)
    X_train, X_test, y_train, y_test = get_split(X, y)
    block_bits = infer_block_bits(y_train)

    # 2. HYPERPARAMETER TUNING
    # Set default values based on whether we are in "fast mode".
    train_cap = 16_000 if _fast_mode_enabled() else MAX_TRAIN_SAMPLES
    test_cap = 4_000 if _fast_mode_enabled() else MAX_TEST_SAMPLES
    n_estimators = 60 if _fast_mode_enabled() else 180
    max_depth = 16 if _fast_mode_enabled() else None
    min_samples_leaf = 2

    # Specific overrides for certain ciphers to ensure we get a good look 
    # at their security properties.
    if cipher.lower() == "simon" and _fast_mode_enabled():
        train_cap, test_cap, n_estimators, max_depth = 24_000, 6_000, 180, None

    if cipher.lower() == "aes":
        # AES is very complex; we use more trees and depth when not in fast mode.
        if _fast_mode_enabled():
            train_cap, test_cap, n_estimators, max_depth = 8_000, 2_000, 40, 14
        else:
            train_cap = min(MAX_TRAIN_SAMPLES, len(X_train))
            test_cap = min(MAX_TEST_SAMPLES, len(X_test))
            n_estimators, max_depth = 220, None

    random_hamming = block_bits / 2.0

    # 3. PRE-PROCESSING
    X_train, y_train = X_train[:train_cap], y_train[:train_cap]
    X_test, y_test = X_test[:test_cap], y_test[:test_cap]
    
    # Feature Augmentation: Help the trees see bitwise XOR patterns.
    X_train = augment_plaintext_features(X_train, cipher=cipher)
    X_test = augment_plaintext_features(X_test, cipher=cipher)

    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")
    print(f"  Feature dim       : {X_train.shape[1]} (augmented)")
    print(f"  RF config         : trees={n_estimators}, max_depth={max_depth}")

    # 4. TRAINING & PREDICTION
    model, train_time = timed_call(
        train_random_forest, X_train, y_train,
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf
    )

    y_pred = model.predict(X_test).astype(np.uint8)
    
    # 5. RESULTS & LOGGING
    metrics = compute_metrics(y_test, y_pred)
    metrics["block_bits"] = block_bits
    metrics["train_time_s"] = round(train_time, 2)

    print(f"  Train time        : {train_time:.1f}s")
    print(f"  Bitwise accuracy  : {metrics['bitwise_accuracy']*100:.2f}%")
    print(f"  Full-word accuracy: {metrics['word_accuracy']*100:.4f}%")

    save_metrics(metrics, cipher, MODEL_NAME, num_rounds)
    return metrics


if __name__ == "__main__":
    # Standard 5-round sweep for the Simon cipher.
    results = {}
    for r in [1, 2, 3, 4, 5]:
        results[r] = run_experiment(r, cipher="simon")
    save_summary(results, "simon", MODEL_NAME)
