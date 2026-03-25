"""
================================================================================
PER-MODEL PLOT GENERATOR
================================================================================

OVERVIEW:
This module is responsible for visualizing the training and evaluation 
history of individual models. While the comparison report looks at 
all models together, these plots provide a deep dive into how a single 
architecture performed as the cipher round count increased.

WORKFLOW:
1. DATA LOADING:
   - Reads model-specific summary JSON files from `results/metrics/`.
2. PLOTTING:
   - Generates line charts for accuracy and loss.
3. EXPORT:
   - Saves individual plot images to `results/plots/` using a 
     standard naming convention: {cipher}_{model}.png.
"""

import json
import os

import matplotlib.pyplot as plt


CIPHER = "simon"
MODELS = ["logistic", "mlp", "cnn", "mine", "random_forest"]
METRICS_DIR = "results/metrics"
OUT_DIR = "results/plots"


def _tight_hamming_ylim(hamming_values, random_hamming):
    """
    Calculates a tight y-axis range for Hamming distance plots.
    
    This ensures the plot is focused on the relevant data points while 
    keeping the 'random' baseline visible if it's within reach.
    """
    if not hamming_values:
        return (0, max(1.0, random_hamming * 1.25))

    lo = min(hamming_values)
    hi = max(hamming_values)
    span = max(hi - lo, 1e-6)

    # Apply a small padding (12%) to the top and bottom.
    pad = max(0.12, span * 0.12)
    y_min = max(0.0, lo - pad)
    y_max = min(float(max(32, int(random_hamming * 2))), hi + pad)

    # Keep random baseline visible only when it is reasonably near model values.
    if abs(random_hamming - (lo + hi) / 2.0) <= max(1.0, span * 1.35):
        y_min = min(y_min, random_hamming - 0.1)
        y_max = max(y_max, random_hamming + 0.1)

    # Ensure a minimum visible window size to avoid extremely flat plots.
    if (y_max - y_min) < 0.6:
        center = (lo + hi) / 2.0
        y_min = max(0.0, center - 0.3)
        y_max = min(float(max(32, int(random_hamming * 2))), center + 0.3)

    return (y_min, y_max)


def _tight_accuracy_ylim(accuracy_values, random_accuracy=50.0):
    """
    Calculates a tight y-axis range for bitwise accuracy (percentage) plots.
    """
    if not accuracy_values:
        return (40.0, 100.0)

    lo = min(accuracy_values)
    hi = max(accuracy_values)
    span = max(hi - lo, 1e-6)

    # Special logic for high-accuracy results (near 100%).
    if hi >= 90.0:
        pad = max(0.12, span * 0.10)
        y_min = max(0.0, lo - pad)
        y_max = min(100.0, hi + pad)

        if (y_max - y_min) < 0.6:
            center = (lo + hi) / 2.0
            y_min = max(0.0, center - 0.3)
            y_max = min(100.0, center + 0.3)
        return (y_min, y_max)

    # Standard range logic for medium/low accuracy.
    pad = max(0.4, span * 0.12)
    y_min = max(0.0, lo - pad)
    y_max = min(100.0, hi + pad)

    # Include the 50% baseline if it's relevant.
    if abs(random_accuracy - (lo + hi) / 2.0) <= max(2.0, span * 1.35):
        y_min = min(y_min, random_accuracy - 0.5)
        y_max = max(y_max, random_accuracy + 0.5)

    if (y_max - y_min) < 1.0:
        center = (lo + hi) / 2.0
        y_min = max(0.0, center - 0.5)
        y_max = min(100.0, center + 0.5)

    return (y_min, y_max)


def load_summary(model: str, cipher: str = CIPHER):
    """
    Loads the multi-round summary JSON for a specific model and cipher.
    """
    path = os.path.join(METRICS_DIR, f"{cipher}_{model}_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Convert string keys back to integers for sorting.
    return {int(k): v for k, v in raw.items()}


def plot_model_results(model: str, results: dict, cipher: str = CIPHER):
    """
    Generates and saves a two-pane plot (Accuracy and Hamming) for one model.
    """
    rounds = sorted(results.keys())
    bit_acc = [results[r]["bitwise_accuracy"] * 100 for r in rounds]
    hamming = [results[r]["hamming_distance"] for r in rounds]
    
    # Infer block size from the first available round's data.
    block_bits = int(results[rounds[0]].get("block_bits", 32)) if rounds else 32
    random_hamming = block_bits / 2.0

    # Initialize the figure with two subplots side-by-side.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{cipher.upper()}  x  {model.upper()}", fontsize=14, fontweight="bold")

    # Pane 1: Bitwise Accuracy (%)
    ax = axes[0]
    ax.plot(rounds, bit_acc, "o-", lw=2.5, ms=8, label=model.upper())
    ax.axhline(50, ls="--", color="red", lw=1.5, label="Random (50%)")
    acc_y_min, acc_y_max = _tight_accuracy_ylim(bit_acc, random_accuracy=50.0)
    
    # Special zoom handling for Trivium due to its unique diffusion curve.
    if cipher.lower() == "trivium":
        acc_lo = min(bit_acc)
        acc_hi = max(bit_acc)
        acc_span = max(acc_hi - acc_lo, 1e-6)
        acc_pad = max(0.06, acc_span * 0.25)
        acc_y_min = max(0.0, acc_lo - acc_pad)
        acc_y_max = min(100.0, acc_hi + acc_pad)
        
    ax.set(
        xlabel="Rounds (r)",
        ylabel="Bitwise Accuracy (%)",
        title="Bitwise Accuracy vs Rounds",
        xticks=rounds,
        ylim=(acc_y_min, acc_y_max),
    )
    ax.grid(alpha=0.3)
    ax.legend()

    # Pane 2: Hamming Distance (bits)
    ax = axes[1]
    ax.plot(rounds, hamming, "s-", lw=2.5, ms=8, label=model.upper())
    ax.axhline(random_hamming, ls="--", color="red", lw=1.5, label=f"Random ({random_hamming:.0f} bits)")
    y_min, y_max = _tight_hamming_ylim(hamming, random_hamming)
    
    # Trivium specialized Hamming zoom.
    if cipher.lower() == "trivium":
        ham_lo = min(hamming)
        ham_hi = max(hamming)
        ham_span = max(ham_hi - ham_lo, 1e-6)
        ham_pad = max(0.05, ham_span * 0.25)
        y_min = max(0.0, ham_lo - ham_pad)
        y_max = min(float(max(32, int(random_hamming * 2))), ham_hi + ham_pad)
        
    ax.set(
        xlabel="Rounds (r)",
        ylabel="Hamming Distance (bits)",
        title="Hamming Distance vs Rounds",
        xticks=rounds,
        ylim=(y_min, y_max),
    )
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{cipher}_{model}_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"    [Plot] Saved per-model plot -> {out_path}")


def generate_all_plots(cipher: str = CIPHER, models = MODELS):
    """
    Main runner to generate plots for all selected models.
    """
    print(f"\n  [Plot] Generating per-model visualizations for {cipher.upper()}...")
    for model in models:
        results = load_summary(model, cipher=cipher)
        if results:
            plot_model_results(model, results, cipher=cipher)
        else:
            print(f"    [Plot] Skipping {model}: metrics summary not found.")


if __name__ == "__main__":
    generate_all_plots()
