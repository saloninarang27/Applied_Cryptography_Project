"""
================================================================================
MINE-INSPIRED NEURAL ESTIMATOR FOR CIPHER APPROXIMATION
================================================================================

OVERVIEW:
This module implements a hybrid model inspired by Mutual Information Neural 
Estimation (MINE). It combines a standard Deep Neural Network for prediction 
with an auxiliary network that tries to maximize the Mutual Information (MI) 
between the network's internal features and the real ciphertext bits.

HOW IT WORKS:
1. ENCODER-PREDICTOR:
   - Takes plaintext bits and maps them to an internal "latent" representation.
   - A classification head then predicts the ciphertext bits from this representation.
2. MINE NETWORK (The Regularizer):
   - A second network that estimates the Mutual Information between the 
     Encoder's features and the actual output.
   - By maximizing this MI, we force the Encoder to keep as much relevant 
     information as possible about the cipher's transformation.
3. JOINT OPTIMIZATION:
   - The model is trained using a combined loss function: 
     Loss = Classification_Error - (Weight * Mutual_Information)
   - This "pulls" the model to be both accurate and information-theoretically 
     aligned with the cipher.

WHY MINE?
Ciphers are designed to be "zero-information" (indistinguishable from random). 
Standard neural networks can sometimes "give up" if the mapping looks too 
random. MINE provides a stronger training signal by explicitly looking for 
statistical dependencies that a normal loss function might miss.

THIS MODULE:
- Implements the MINE-style architecture in PyTorch.
- Features complex hyperparameter schedules for different ciphers (Speck, 
  Present, Trivium, etc.).
- Includes a Mutual Information lower-bound estimator based on the 
  Donsker-Varadhan representation.
================================================================================
"""

import os
import sys
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Project Path Setup.
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

# Global Configuration.
MODEL_NAME = "mine"
EPOCHS = 14
BATCH_SIZE = 1024
MAX_TRAIN_SAMPLES = 90000
MAX_TEST_SAMPLES = 25000
MINE_WEIGHT = 0.12 # Weight of the Mutual Information regularizer.


def _fast_mode_enabled() -> bool:
    """Checks if developer fast-mode is active."""
    return os.getenv("AC_FAST_MODE", "0") == "1"


def _torch_available():
    """Verifies PyTorch presence."""
    try:
        import torch # noqa: F401
        return True
    except Exception:
        print(f"  Torch unavailable - using fallback baseline")
        return False


def _run_fallback(X_train, y_train, X_test):
    """Fallback to Random Forest if PyTorch is missing."""
    model = RandomForestClassifier(n_estimators=80, max_depth=14, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test).astype(np.uint8)


def run_experiment(num_rounds: int, data_dir: str = "data", cipher: str = DEFAULT_CIPHER):
    """
    Executes the MINE experiment for a specific cipher and round count.
    """
    print(f"\n{'='*55}")
    print(f"  {cipher.upper()}  |  MINE  |  r = {num_rounds}")
    print(f"{'='*55}")

    # 1. DATA LOADING
    X, y = load_or_generate_dataset(num_rounds, data_dir=data_dir, cipher=cipher)
    X_train, X_test, y_train, y_test = get_split(X, y)
    block_bits = infer_block_bits(y_train)

    # 2. HYPERPARAMETER HEURISTICS
    # Different ciphers require different "MI weights" and network sizes.
    train_cap = 20_000 if _fast_mode_enabled() else MAX_TRAIN_SAMPLES
    test_cap = 6_000 if _fast_mode_enabled() else MAX_TEST_SAMPLES
    epochs = 6 if _fast_mode_enabled() else EPOCHS
    mine_weight = MINE_WEIGHT
    hidden_size = 128

    # The following blocks contain fine-tuned settings for specific ciphers.
    if cipher.lower() == "speck":
        if num_rounds <= 2:
            hidden_size, mine_weight = 192, 0.08
            epochs = 10 if _fast_mode_enabled() else 20
        else:
            mine_weight = 0.06

    if cipher.lower() == "trivium":
        # Trivium is a stream-cipher; it needs a higher weight to find the state bits.
        hidden_size, mine_weight = 224, 0.12
        epochs = 16 if _fast_mode_enabled() else 28

    # 3. PRE-PROCESSING
    X_train, y_train = X_train[:train_cap], y_train[:train_cap]
    X_test, y_test = X_test[:test_cap], y_test[:test_cap]
    X_train = augment_plaintext_features(X_train, cipher=cipher)
    X_test = augment_plaintext_features(X_test, cipher=cipher)
    
    random_hamming = block_bits / 2.0

    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")
    print(f"  Feature dim       : {X_train.shape[1]} (augmented)")

    # 4. PYTORCH IMPLEMENTATION
    if _torch_available():
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        class MinePredictor(nn.Module):
            """
            The Dual-Network Architecture.
            """
            def __init__(self, input_dim: int, hidden: int = 128, out_bits: int = 32):
                super().__init__()
                # Part A: The Encoder.
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                )
                # Part B: The Bit Predictor.
                self.head = nn.Linear(hidden, out_bits)
                
                # Part C: The MINE (Mutual Information) network.
                # It takes features + actual bits and outputs a score.
                self.mine_net = nn.Sequential(
                    nn.Linear(hidden + out_bits, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )

            def forward(self, x):
                z = self.encoder(x)
                logits = self.head(z)
                return z, logits

            def mi_lower_bound(self, z, y):
                """
                Estimates the Mutual Information lower bound using the 
                Donsker-Varadhan representation.
                """
                # Joint Distribution (real pairs of features and bits).
                joint = torch.cat([z, y], dim=1)
                # Marginal Distribution (shuffled pairs).
                y_perm = y[torch.randperm(y.size(0), device=y.device)]
                marginal = torch.cat([z, y_perm], dim=1)
                
                t_joint = self.mine_net(joint).squeeze(-1)
                t_marginal = self.mine_net(marginal).squeeze(-1)
                
                # MI ≈ Mean(T_joint) - Log(Mean(Exp(T_marginal)))
                log_mean_exp = torch.logsumexp(t_marginal, dim=0) - torch.log(
                    torch.tensor(float(t_marginal.shape[0]), device=t_marginal.device)
                )
                return t_joint.mean() - log_mean_exp

        def _fit(model, loader, optimizer, criterion, device):
            """Specialized training loop with MI maximization."""
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                z, logits = model(xb)
                
                # Combined Loss.
                pred_loss = criterion(logits, yb)
                mi_lb = model.mi_lower_bound(z, yb)
                loss = pred_loss - (mine_weight * mi_lb) # Maximize MI by subtracting it.
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        def _predict(model, X_np, device):
            model.eval()
            with torch.no_grad():
                xb = torch.tensor(X_np, dtype=torch.float32, device=device)
                _, logits = model(xb)
                probs = torch.sigmoid(logits)
            return (probs.cpu().numpy() >= 0.5).astype(np.uint8)

        # Execution.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MinePredictor(input_dim=X_train.shape[1], hidden=hidden_size, out_bits=block_bits).to(device)

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        t0 = time.time()
        for _ in range(epochs):
            _fit(model, train_loader, optimizer, criterion, device)
        train_time = time.time() - t0
        y_pred = _predict(model, X_test, device)
        device_label = str(device)
        
    else:
        # Fallback to Random Forest.
        t0 = time.time()
        y_pred = _run_fallback(X_train, y_train, X_test)
        train_time = time.time() - t0
        device_label = "fallback-sklearn"

    # 5. RESULTS
    y_pred = y_pred.astype(np.uint8)
    metrics = compute_metrics(y_test, y_pred)
    metrics["block_bits"] = block_bits
    metrics["train_time_s"] = round(train_time, 2)

    print(f"  Device            : {device_label}")
    print(f"  Train time        : {train_time:.1f}s")
    print(f"  Bitwise accuracy  : {metrics['bitwise_accuracy']*100:.2f}%")

    save_metrics(metrics, cipher, MODEL_NAME, num_rounds)
    return metrics


if __name__ == "__main__":
    results = {}
    for r in [1, 2, 3, 4, 5]:
        results[r] = run_experiment(r, cipher="simon")
    save_summary(results, "simon", MODEL_NAME)
