"""
================================================================================
1D CONVOLUTIONAL NEURAL NETWORK (CNN) FOR CIPHER APPROXIMATION
================================================================================

OVERVIEW:
This module implements a 1D Convolutional Neural Network (CNN) designed to 
approximate the behavior of reduced-round block ciphers. Unlike traditional 
cryptanalysis, which uses mathematical proofs, this approach treats the 
ciphertext as a complex "signal" and tries to find patterns using Deep Learning.

HOW IT WORKS:
1. DATA REPRESENTATION:
   - Plaintext bits are treated as a 1D sequence (a "signal") of 0s and 1s.
   - The CNN "slides" a filter over these bits to detect local dependencies 
     (e.g., how bit 1 affects bit 3).
2. CONVOLUTIONAL LAYERS:
   - Multiple filters learn to extract features from the bitstream.
   - ReLU activation functions add non-linearity, allowing the model to 
     capture complex XOR-based transformations.
3. ADAPTIVE POOLING:
   - Reduces the dimensionality of the features while keeping the most 
     important information.
4. FULLY CONNECTED (DENSE) LAYERS:
   - These layers act as the "brain," combining the extracted features 
     to predict the final ciphertext bits.

WHY CNN?
CNNs are excellent at finding local patterns and spatial hierarchies. In 
ciphers, bits often interact with their neighbors in specific ways (like 
in S-boxes or bit-shifts). A CNN is naturally suited to pick up these 
structural patterns.

THIS MODULE:
- Provides a PyTorch-based CNN architecture.
- Includes a training and evaluation pipeline for various ciphers.
- Features a fallback to a Random Forest model if PyTorch is not available.
================================================================================
"""

import os
import sys
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Add the project root to the path so we can import local modules.
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
)

# Configuration for the CNN model.
MODEL_NAME = "cnn"
EPOCHS = 12
BATCH_SIZE = 1024
MAX_TRAIN_SAMPLES = 90000
MAX_TEST_SAMPLES = 25000


def _fast_mode_enabled() -> bool:
    """Checks if the AC_FAST_MODE environment variable is set for quicker testing."""
    return os.getenv("AC_FAST_MODE", "0") == "1"


def _torch_available():
    """Verifies if PyTorch is installed on the system."""
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
        from torch.utils.data import DataLoader, TensorDataset  # noqa: F401
        return True
    except Exception as exc:
        print(f"  Torch unavailable ({exc.__class__.__name__}) - using fallback CNN baseline")
        return False


def _run_fallback(X_train, y_train, X_test):
    """Fallback: Uses Random Forest if PyTorch is not available."""
    model = RandomForestClassifier(n_estimators=80, max_depth=14, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test).astype(np.uint8)


def run_experiment(num_rounds: int, data_dir: str = "data", cipher: str = DEFAULT_CIPHER):
    """
    Executes a single CNN experiment for a specific cipher and round count.
    """
    print(f"\n{'='*55}")
    print(f"  {cipher.upper()}  |  CNN  |  r = {num_rounds}")
    print(f"{'='*55}")

    # 1. DATA LOADING
    # Load the bits from the generated dataset.
    X, y = load_or_generate_dataset(num_rounds, data_dir=data_dir, cipher=cipher)
    X_train, X_test, y_train, y_test = get_split(X, y)
    block_bits = infer_block_bits(y_train)

    # 2. RUNTIME OPTIMIZATION
    # Adjust sample sizes if we are in "fast mode" to save time during development.
    train_cap = 20_000 if _fast_mode_enabled() else MAX_TRAIN_SAMPLES
    test_cap = 6_000 if _fast_mode_enabled() else MAX_TEST_SAMPLES
    epochs = 5 if _fast_mode_enabled() else EPOCHS

    # Specific tweaks for certain ciphers to ensure meaningful results.
    if cipher.lower() == "simon" and _fast_mode_enabled():
        train_cap, test_cap, epochs = 24_000, 6_000, 16

    if cipher.lower() == "aes":
        if _fast_mode_enabled():
            train_cap, test_cap, epochs = 6_000, 2_000, max(epochs, 2)
        else:
            train_cap = min(MAX_TRAIN_SAMPLES, len(X_train))
            test_cap = min(MAX_TEST_SAMPLES, len(X_test))

    # Apply the caps.
    X_train, y_train = X_train[:train_cap], y_train[:train_cap]
    X_test, y_test = X_test[:test_cap], y_test[:test_cap]

    # Feature Augmentation: Add extra bitwise info (like XORs) to help the model learn.
    X_train = augment_plaintext_features(X_train, cipher=cipher)
    X_test = augment_plaintext_features(X_test, cipher=cipher)
    
    random_hamming = block_bits / 2.0 # Theoretical average distance for random data.

    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")
    print(f"  Feature dim       : {X_train.shape[1]} (augmented)")

        # 3. MODEL TRAINING (PyTorch Path)
    # We use PyTorch for the CNN because of its excellent support for 
    # parallelizing bitwise computations on the GPU.
    if _torch_available():
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        class CipherCNN(nn.Module):
            """
            A simple 1D CNN architecture for bitwise classification.
            """
            def __init__(self, out_bits: int):
                super().__init__()
                # The 'Features' part: Extracting patterns from the bitstream.
                # We use 1D Convolutions because the plaintext is a sequence of bits.
                self.features = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(8), # Squashes feature map to size 8.
                )
                # The 'Classifier' part: Making the final prediction.
                # It takes the features and maps them to the output bits.
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32 * 8, 128),
                    nn.ReLU(),
                    nn.Linear(128, out_bits), # One output neuron per ciphertext bit.
                )

            def forward(self, x):
                # CNN expects (Batch, Channels, Length). We add the Channel dimension.
                # Input: [Batch, Bits] -> Output: [Batch, 1, Bits]
                x = x.unsqueeze(1)
                # Pass through features then classifier.
                return self.classifier(self.features(x))

        def _fit(model, loader, optimizer, criterion, device):
            """Standard PyTorch training loop for one epoch."""
            model.train()
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)

        def _predict(model, X_np, device):
            """Generates predictions (0 or 1) for the input data."""
            model.eval()
            with torch.no_grad():
                xb = torch.tensor(X_np, dtype=torch.float32, device=device)
                logits = model(xb)
                # Apply sigmoid to get probabilities between 0 and 1.
                probs = torch.sigmoid(logits)
            # Threshold at 0.5 to get binary bits (0 or 1).
            return (probs.cpu().numpy() >= 0.5).astype(np.uint8)

        # Move model to GPU if available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CipherCNN(out_bits=block_bits).to(device)

        # Prepare DataLoaders for efficient batch processing.
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        # Use BCEWithLogitsLoss (includes sigmoid internally) for bitwise classification.
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train the model.
        print(f"  [Model] Training CNN on {device} for {epochs} epochs...")
        t0 = time.time()
        for epoch in range(epochs):
            avg_loss = _fit(model, train_loader, optimizer, criterion, device)
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
        train_time = time.time() - t0
        
        # Predict on test set to evaluate final performance.
        y_pred = _predict(model, X_test, device)
        device_label = str(device)
        
    else:
        # 4. FALLBACK PATH (Scikit-Learn)
        # If PyTorch is missing, we use a Random Forest as a baseline.
        print("  [Model] Torch missing - using Random Forest fallback...")
        t0 = time.time()
        y_pred = _run_fallback(X_train, y_train, X_test)
        train_time = time.time() - t0
        device_label = "fallback-sklearn"

    # 5. METRICS & LOGGING
    # Final cleanup and metric calculation.
    y_pred = y_pred.astype(np.uint8)
    metrics = compute_metrics(y_test, y_pred)
    metrics["block_bits"] = block_bits
    metrics["train_time_s"] = round(train_time, 2)

    # Detailed summary for the user.
    print(f"  [Model] Performance Summary:")
    print(f"    - Device            : {device_label}")
    print(f"    - Total Train Time  : {train_time:.1f}s")
    print(f"    - Bitwise Accuracy  : {metrics['bitwise_accuracy']*100:.2f}%  (Baseline: 50.0%)")
    print(f"    - Hamming Distance  : {metrics['hamming_distance']:.3f} bits (Baseline: {random_hamming:.1f})")
    print(f"    - Full-word Matches : {metrics['word_accuracy']*100:.4f}%")

    # Save results to disk for the final comparison report.
    save_metrics(metrics, cipher, MODEL_NAME, num_rounds)
    return metrics


if __name__ == "__main__":
    # If run directly, perform a sweep over rounds 1 to 5 for the Simon cipher.
    results = {}
    for r in [1, 2, 3, 4, 5]:
        results[r] = run_experiment(r, cipher="simon")
    save_summary(results, "simon", MODEL_NAME)
