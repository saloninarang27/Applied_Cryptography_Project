"""
================================================================================
CIPHER EXPERIMENT COMPARISON GENERATOR
================================================================================

OVERVIEW:
This script aggregates results from all models trained on a specific 
cipher and produces comparative reports. It allows researchers to 
easily see which model is best at identifying patterns in the bitstream 
across different rounds.

OUTPUTS:
1. JSON SUMMARIES:
   - A consolidated JSON file containing accuracy and loss for all models.
2. MARKDOWN TABLES:
   - A human-readable table summarizing performance per round.
3. COMPARISON PLOTS:
   - A multi-line graph comparing the "round vs accuracy" curves of all models.

ISOLATION:
- All outputs are stored in `results/comparison/{cipher}/` to ensure 
  experiments for different ciphers do not overlap.
"""

import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt


CIPHER = "simon"
MODELS = ["logistic", "mlp", "cnn", "mine", "random_forest"]
ROUNDS = [1, 2, 3, 4, 5]
METRICS_DIR = "results/metrics"
OUT_ROOT = "results/comparison"

MODEL_COLORS = {
    "logistic": "#1f77b4",
    "mlp": "#ff7f0e",
    "cnn": "#2ca02c",
    "mine": "#d62728",
    "random_forest": "#9467bd",
}


def _tight_hamming_ylim(all_hamming_values, random_hamming):
    """
    Calculates a tight y-axis range for Hamming distance plots.
    
    This ensures the plot is focused on the relevant data points while 
    keeping the 'random' baseline visible if it's within reach.
    """
    if not all_hamming_values:
        return (0, max(1.0, random_hamming * 1.25))

    lo = min(all_hamming_values)
    hi = max(all_hamming_values)
    span = max(hi - lo, 1e-6)

    # Apply a small padding (12%) to the top and bottom.
    pad = max(0.12, span * 0.12)
    y_min = max(0.0, lo - pad)
    y_max = min(float(max(32, int(random_hamming * 2))), hi + pad)

    # If the random baseline is near the data range, expand the axis to include it.
    if abs(random_hamming - (lo + hi) / 2.0) <= max(1.0, span * 1.35):
        y_min = min(y_min, random_hamming - 0.1)
        y_max = max(y_max, random_hamming + 0.1)

    # Ensure a minimum visible window size to avoid extremely flat plots.
    if (y_max - y_min) < 0.6:
        center = (lo + hi) / 2.0
        y_min = max(0.0, center - 0.3)
        y_max = min(float(max(32, int(random_hamming * 2))), center + 0.3)

    return (y_min, y_max)


def _focused_hamming_ylim(all_hamming_values, random_hamming):
    """
    Heuristic for 'zoomed-in' plots when data points are clustered 
    near the random baseline.
    """
    if not all_hamming_values:
        return _tight_hamming_ylim(all_hamming_values, random_hamming), False

    span = max(all_hamming_values) - min(all_hamming_values)
    # If the spread is already large, we don't need a special focus.
    if span <= 1.5:
        return _tight_hamming_ylim(all_hamming_values, random_hamming), False

    # Count how many points are actually near the random line.
    near_random = [v for v in all_hamming_values if abs(v - random_hamming) <= 1.25]
    if len(near_random) < max(6, int(0.55 * len(all_hamming_values))):
        return _tight_hamming_ylim(all_hamming_values, random_hamming), False

    # Focus on the 'near-random' cluster.
    lo = min(near_random)
    hi = max(near_random)
    pad = max(0.25, (hi - lo) * 0.35)
    y_min = max(0.0, lo - pad)
    y_max = min(float(max(32, int(random_hamming * 2))), hi + pad)

    if (y_max - y_min) < 1.0:
        center = (lo + hi) / 2.0
        y_min = max(0.0, center - 0.5)
        y_max = min(float(max(32, int(random_hamming * 2))), center + 0.5)

    return (y_min, y_max), True


def _tight_accuracy_ylim(all_accuracy_values, random_accuracy=50.0):
    """
    Calculates a tight y-axis range for bitwise accuracy (percentage) plots.
    """
    if not all_accuracy_values:
        return (40.0, 100.0)

    lo = min(all_accuracy_values)
    hi = max(all_accuracy_values)
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


def _focused_accuracy_ylim(all_accuracy_values, random_accuracy=50.0):
    """
    Heuristic to 'zoom' into accuracy plots when results are near 50%.
    """
    if not all_accuracy_values:
        return _tight_accuracy_ylim(all_accuracy_values, random_accuracy=random_accuracy), False

    span = max(all_accuracy_values) - min(all_accuracy_values)
    if span <= 4.0:
        return _tight_accuracy_ylim(all_accuracy_values, random_accuracy=random_accuracy), False

    near_random = [v for v in all_accuracy_values if abs(v - random_accuracy) <= 2.5]
    if len(near_random) < max(6, int(0.55 * len(all_accuracy_values))):
        return _tight_accuracy_ylim(all_accuracy_values, random_accuracy=random_accuracy), False

    lo = min(near_random)
    hi = max(near_random)
    pad = max(0.5, (hi - lo) * 0.35)
    y_min = max(0.0, lo - pad)
    y_max = min(100.0, hi + pad)

    # Check if there are high-accuracy values (e.g., round 1 peaks)
    high_values = [v for v in all_accuracy_values if v >= 85.0]
    if high_values:
        y_max = max(y_max, max(high_values) + 1.0)

    if (y_max - y_min) < 1.5:
        center = (lo + hi) / 2.0
        y_min = max(0.0, center - 0.75)
        y_max = min(100.0, center + 0.75)

    return (y_min, y_max), True


def _infer_block_bits(data: Dict[str, Dict[str, Dict]]) -> int:
    for model_data in data.values():
        for payload in model_data.values():
            if payload:
                return int(payload.get("block_bits", 32))
    return 32


def _read_model_round(cipher: str, model: str, round_id: int) -> Dict:
    path = os.path.join(METRICS_DIR, f"{cipher}_{model}_r{round_id}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_comparison(cipher: str = CIPHER, models: List[str] = MODELS, rounds: List[int] = ROUNDS) -> Dict[str, Dict[str, Dict]]:
    data: Dict[str, Dict[str, Dict]] = {}
    for model in models:
        data[model] = {}
        for r in rounds:
            payload = _read_model_round(cipher, model, r)
            if payload:
                data[model][str(r)] = payload
    return data


def save_round_reports(data: Dict[str, Dict[str, Dict]], cipher: str = CIPHER, models: List[str] = MODELS, rounds: List[int] = ROUNDS) -> None:
    out_dir = os.path.join(OUT_ROOT, cipher)
    os.makedirs(out_dir, exist_ok=True)
    for r in rounds:
        row = {}
        for model in models:
            if str(r) in data.get(model, {}):
                row[model] = data[model][str(r)]
        payload = {
            "cipher": cipher,
            "round": r,
            "models": row,
        }
        with open(os.path.join(out_dir, f"round_{r}_comparison.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def save_all_summary(data: Dict[str, Dict[str, Dict]], cipher: str = CIPHER) -> None:
    out_dir = os.path.join(OUT_ROOT, cipher)
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "cipher": cipher,
        "results_by_model": data,
    }
    with open(os.path.join(out_dir, "all_models_summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_markdown_table(data: Dict[str, Dict[str, Dict]], cipher: str = CIPHER, models: List[str] = MODELS, rounds: List[int] = ROUNDS) -> None:
    out_dir = os.path.join(OUT_ROOT, cipher)
    os.makedirs(out_dir, exist_ok=True)

    lines: List[str] = []
    lines.append(f"# {cipher.upper()} Model Comparison")
    lines.append("")
    for r in rounds:
        lines.append(f"## Cipher: {cipher.upper()} | Round {r}")
        lines.append("")
        lines.append("| Model | Bitwise Accuracy | Hamming Distance | Train Time (s) |")
        lines.append("|---|---:|---:|---:|")
        for model in models:
            item = data.get(model, {}).get(str(r), {})
            if not item:
                lines.append(f"| {model} | N/A | N/A | N/A |")
                continue
            lines.append(
                f"| {model} | {item['bitwise_accuracy']*100:.2f}% | {item['hamming_distance']:.3f} | {item.get('train_time_s', 0):.2f} |"
            )
        lines.append("")

    with open(os.path.join(out_dir, "comparison_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_plot(data: Dict[str, Dict[str, Dict]], cipher: str = CIPHER, models: List[str] = MODELS, rounds: List[int] = ROUNDS) -> None:
    out_dir = os.path.join(OUT_ROOT, cipher)
    os.makedirs(out_dir, exist_ok=True)
    block_bits = _infer_block_bits(data)
    random_hamming = block_bits / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cipher_key = cipher.lower()

    all_hamming_values = []
    all_accuracy_values = []
    visible_models = [model for model in models if any(str(r) in data.get(model, {}) for r in rounds)]
    offset_span = 0.24
    model_offsets = {}
    if visible_models:
        if len(visible_models) == 1:
            model_offsets[visible_models[0]] = 0.0
        else:
            step = offset_span / (len(visible_models) - 1)
            start = -offset_span / 2.0
            for i, model_name in enumerate(visible_models):
                model_offsets[model_name] = start + i * step

    for model in models:
        model_data = data.get(model, {})
        xs = [r for r in rounds if str(r) in model_data]
        if not xs:
            continue
        x_offset = model_offsets.get(model, 0.0)
        xs_plot = [x + x_offset for x in xs]
        bit_acc = [model_data[str(r)]["bitwise_accuracy"] * 100 for r in xs]
        hamming = [model_data[str(r)]["hamming_distance"] for r in xs]
        all_accuracy_values.extend(bit_acc)
        all_hamming_values.extend(hamming)
        color = MODEL_COLORS.get(model)
        axes[0].plot(xs_plot, bit_acc, marker="o", linewidth=2, markersize=6, label=model, color=color)
        axes[1].plot(xs_plot, hamming, marker="s", linewidth=2, markersize=6, label=model, color=color)

    axes[0].axhline(50, linestyle="--", color="red", linewidth=1)
    axes[0].set_title(f"{cipher.upper()} Bitwise Accuracy vs Rounds")
    axes[0].set_xlabel("Rounds")
    axes[0].set_ylabel("Bitwise Accuracy (%)")
    axes[0].set_xticks(rounds)
    (acc_y_min, acc_y_max), acc_zoomed = _focused_accuracy_ylim(all_accuracy_values, random_accuracy=50.0)
    if cipher.lower() == "trivium" and all_accuracy_values:
        acc_lo = min(all_accuracy_values)
        acc_hi = max(all_accuracy_values)
        acc_span = max(acc_hi - acc_lo, 1e-6)
        acc_pad = max(0.08, acc_span * 0.25)
        acc_y_min = max(0.0, acc_lo - acc_pad)
        acc_y_max = min(100.0, acc_hi + acc_pad)
        acc_zoomed = False
    if cipher.lower() == "chacha20" and all_accuracy_values:
        acc_lo = min(all_accuracy_values)
        acc_hi = max(all_accuracy_values)
        acc_span = max(acc_hi - acc_lo, 1e-6)
        acc_pad = max(2.0, acc_span * 0.20)
        acc_y_min = max(0.0, acc_lo - acc_pad)
        acc_y_max = min(100.0, acc_hi + acc_pad)
        acc_zoomed = False
    axes[0].set_ylim(acc_y_min, acc_y_max)
    if acc_zoomed:
        axes[0].set_title(f"{cipher.upper()} Bitwise Accuracy vs Rounds (zoomed)")
    if cipher.lower() == "trivium":
        axes[0].set_title(f"{cipher.upper()} Bitwise Accuracy vs Rounds")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].axhline(random_hamming, linestyle="--", color="red", linewidth=1)
    axes[1].set_title(f"{cipher.upper()} Hamming Distance vs Rounds")
    axes[1].set_xlabel("Rounds")
    axes[1].set_ylabel("Hamming Distance")
    axes[1].set_xticks(rounds)
    (y_min, y_max), ham_zoomed = _focused_hamming_ylim(all_hamming_values, random_hamming)
    if cipher.lower() == "trivium":
        trivium_lo = min(all_hamming_values) if all_hamming_values else random_hamming
        trivium_hi = max(all_hamming_values) if all_hamming_values else random_hamming
        span = max(trivium_hi - trivium_lo, 1e-6)
        pad = max(0.06, span * 0.25)
        y_min = max(0.0, trivium_lo - pad)
        y_max = min(float(max(32, int(random_hamming * 2))), trivium_hi + pad)
        ham_zoomed = False
    elif cipher.lower() == "present":
        present_lo = min(all_hamming_values) if all_hamming_values else 0.0
        y_min = max(0.0, present_lo - 0.8)
        y_max = 35.0
        if y_max <= y_min:
            y_max = y_min + 1.0
        ham_zoomed = True
    elif cipher.lower() == "speck":
        speck_lo = min(all_hamming_values) if all_hamming_values else 0.0
        y_min = max(0.0, speck_lo - 0.6)
        y_max = 18.0
        if y_max <= y_min:
            y_max = y_min + 1.0
        ham_zoomed = True
    elif cipher.lower() == "chacha20" and all_hamming_values:
        ham_lo = min(all_hamming_values)
        ham_hi = max(all_hamming_values)
        ham_span = max(ham_hi - ham_lo, 1e-6)
        ham_pad = max(0.8, ham_span * 0.20)
        y_min = max(0.0, ham_lo - ham_pad)
        y_max = min(float(max(32, int(random_hamming * 2))), ham_hi + ham_pad)
        ham_zoomed = False
    axes[1].set_ylim(y_min, y_max)
    if ham_zoomed:
        axes[1].set_title(f"{cipher.upper()} Hamming Distance vs Rounds (zoomed)")
    if cipher.lower() == "trivium":
        axes[1].set_title(f"{cipher.upper()} Hamming Distance vs Rounds")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, "all_models_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Comparison plot saved -> {out_path}")


def generate_report(cipher: str = CIPHER, models: List[str] = MODELS, rounds: List[int] = ROUNDS) -> Dict[str, Dict[str, Dict]]:
    data = build_comparison(cipher=cipher, models=models, rounds=rounds)
    save_round_reports(data, cipher=cipher, models=models, rounds=rounds)
    save_all_summary(data, cipher=cipher)
    save_markdown_table(data, cipher=cipher, models=models, rounds=rounds)
    save_plot(data, cipher=cipher, models=models, rounds=rounds)
    print(f"  Comparison report saved -> {os.path.join(OUT_ROOT, cipher)}/")
    return data


if __name__ == "__main__":
    generate_report()
