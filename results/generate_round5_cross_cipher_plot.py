"""
Generate cross-cipher visualizations from saved metrics.

Supported outputs:
1) Single-round grouped bar chart for all 8 ciphers x 5 models.
2) Optional two-round side-by-side comparison chart.
3) Optional all-rounds line-chart grid (8 panels, one per cipher).
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt


CIPHERS: List[str] = [
    "aes",
    "simon",
    "speck",
    "present",
    "katan",
    "xoodoo",
    "trivium",
    "tinyjambu",
]

MODELS: List[str] = ["logistic", "mlp", "cnn", "mine", "random_forest"]

METRICS_DIR = os.path.join("results", "metrics")
OUT_DIR = os.path.join("results", "comparison", "round5")

MODEL_COLORS = {
    "logistic": "#1f77b4",
    "mlp": "#ff7f0e",
    "cnn": "#2ca02c",
    "mine": "#d62728",
    "random_forest": "#9467bd",
}

CIPHER_COLORS = {
    "aes": "#1f77b4",
    "simon": "#ff7f0e",
    "speck": "#2ca02c",
    "present": "#d62728",
    "katan": "#9467bd",
    "xoodoo": "#8c564b",
    "trivium": "#e377c2",
    "tinyjambu": "#7f7f7f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one graph for all ciphers x all models at a target round"
    )
    parser.add_argument("--round", type=int, default=5, help="Target round (default: 5)")
    parser.add_argument(
        "--compare-with-round",
        type=int,
        default=None,
        help="Optional second round for side-by-side comparison (example: 1)",
    )
    parser.add_argument(
        "--all-rounds-grid",
        action="store_true",
        help="Generate one line-chart image for rounds 1-5 across all 8 ciphers",
    )
    parser.add_argument(
        "--all-rounds-heatmap",
        action="store_true",
        help="Generate one heatmap image for all ciphers/models across rounds 1-5",
    )
    parser.add_argument(
        "--all-rounds-model-panels",
        action="store_true",
        help="Generate one line-chart image with 5 model panels and all cipher curves",
    )
    return parser.parse_args()


def _read_metric(cipher: str, model: str, round_id: int) -> Dict:
    path = os.path.join(METRICS_DIR, f"{cipher}_{model}_r{round_id}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_round_accuracy(round_id: int) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    for cipher in CIPHERS:
        table[cipher] = {}
        for model in MODELS:
            payload = _read_metric(cipher, model, round_id)
            if payload and "bitwise_accuracy" in payload:
                table[cipher][model] = float(payload["bitwise_accuracy"]) * 100.0
    return table


def collect_all_rounds_accuracy(rounds: List[int]) -> Dict[str, Dict[str, Dict[int, float]]]:
    table: Dict[str, Dict[str, Dict[int, float]]] = {}
    for cipher in CIPHERS:
        table[cipher] = {}
        for model in MODELS:
            table[cipher][model] = {}
            for round_id in rounds:
                payload = _read_metric(cipher, model, round_id)
                if payload and "bitwise_accuracy" in payload:
                    table[cipher][model][round_id] = float(payload["bitwise_accuracy"]) * 100.0
    return table


def plot_grouped_bars(table: Dict[str, Dict[str, float]], round_id: int) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)

    ciphers = list(CIPHERS)
    x = list(range(len(ciphers)))
    width = 0.15
    offsets = {
        "logistic": -2 * width,
        "mlp": -1 * width,
        "cnn": 0,
        "mine": 1 * width,
        "random_forest": 2 * width,
    }

    plt.figure(figsize=(15, 7))

    for model in MODELS:
        xs = [xi + offsets[model] for xi in x]
        ys = [table.get(cipher, {}).get(model, float("nan")) for cipher in ciphers]
        plt.bar(xs, ys, width=width, label=model.upper(), color=MODEL_COLORS.get(model))

    plt.axhline(50.0, linestyle="--", linewidth=1.5, color="black", alpha=0.65, label="Random (50%)")
    plt.xticks(x, [c.upper() for c in ciphers])
    plt.ylim(40, 100)
    plt.ylabel("Bitwise Accuracy (%)")
    plt.xlabel("Cipher")
    plt.title(f"Round {round_id}: 5 ML Models Across 8 Ciphers")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncols=3)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"all_ciphers_5_models_round{round_id}_accuracy.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


def _draw_grouped_bars(ax, table: Dict[str, Dict[str, float]], round_id: int) -> None:
    ciphers = list(CIPHERS)
    x = list(range(len(ciphers)))
    width = 0.15
    offsets = {
        "logistic": -2 * width,
        "mlp": -1 * width,
        "cnn": 0,
        "mine": 1 * width,
        "random_forest": 2 * width,
    }

    for model in MODELS:
        xs = [xi + offsets[model] for xi in x]
        ys = [table.get(cipher, {}).get(model, float("nan")) for cipher in ciphers]
        ax.bar(xs, ys, width=width, label=model.upper(), color=MODEL_COLORS.get(model))

    ax.axhline(50.0, linestyle="--", linewidth=1.5, color="black", alpha=0.65, label="Random (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in ciphers])
    ax.set_ylim(40, 100)
    ax.set_ylabel("Bitwise Accuracy (%)")
    ax.set_xlabel("Cipher")
    ax.set_title(f"Round {round_id}")
    ax.grid(axis="y", alpha=0.25)


def plot_two_rounds(round_a: int, round_b: int) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    table_a = collect_round_accuracy(round_a)
    table_b = collect_round_accuracy(round_b)

    fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
    _draw_grouped_bars(axes[0], table_a, round_a)
    _draw_grouped_bars(axes[1], table_b, round_b)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6)
    fig.suptitle(f"All 8 Ciphers x 5 Models: Round {round_a} vs Round {round_b}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_path = os.path.join(OUT_DIR, f"all_ciphers_5_models_round{round_a}_vs_round{round_b}.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all_rounds_grid(rounds: List[int]) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    table = collect_all_rounds_accuracy(rounds)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, cipher in enumerate(CIPHERS):
        ax = axes[idx]
        for model in MODELS:
            model_data = table.get(cipher, {}).get(model, {})
            xs = [r for r in rounds if r in model_data]
            ys = [model_data[r] for r in xs]
            if xs:
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=2,
                    markersize=5,
                    color=MODEL_COLORS.get(model),
                    label=model.upper(),
                )
        ax.axhline(50.0, linestyle="--", linewidth=1.2, color="black", alpha=0.6)
        ax.set_title(cipher.upper())
        ax.set_xticks(rounds)
        ax.set_ylim(45, 100)
        ax.grid(alpha=0.25)

    for ax in axes[:]:
        ax.set_xlabel("Round")
        ax.set_ylabel("Bitwise Accuracy (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6)
    fig.suptitle("All 8 Ciphers x 5 Models: Bitwise Accuracy Across Rounds 1-5", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_path = os.path.join(OUT_DIR, "all_ciphers_5_models_rounds1to5_line_grid.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all_rounds_heatmap(rounds: List[int]) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    table = collect_all_rounds_accuracy(rounds)

    row_labels: List[str] = []
    matrix: List[List[float]] = []
    for cipher in CIPHERS:
        for model in MODELS:
            row_labels.append(f"{cipher.upper()}-{model.upper()}")
            row_values = []
            for r in rounds:
                v = table.get(cipher, {}).get(model, {}).get(r, float("nan"))
                row_values.append(v)
            matrix.append(row_values)

    fig, ax = plt.subplots(figsize=(12, 18))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=49.0, vmax=100.0)

    ax.set_xticks(list(range(len(rounds))))
    ax.set_xticklabels([f"R{r}" for r in rounds])
    ax.set_yticks(list(range(len(row_labels))))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cipher-Model")
    ax.set_title("Single Image: All 8 Ciphers x 5 Models Across Rounds 1-5")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Bitwise Accuracy (%)")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "all_ciphers_5_models_rounds1to5_single_heatmap.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all_rounds_model_panels(rounds: List[int]) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    table = collect_all_rounds_accuracy(rounds)

    fig, axes = plt.subplots(2, 3, figsize=(22, 11), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        for cipher in CIPHERS:
            series = table.get(cipher, {}).get(model, {})
            xs = [r for r in rounds if r in series]
            ys = [series[r] for r in xs]
            if xs:
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=2,
                    markersize=5,
                    color=CIPHER_COLORS.get(cipher),
                    label=cipher.upper(),
                )

        ax.axhline(50.0, linestyle="--", linewidth=1.2, color="black", alpha=0.65)
        ax.set_title(model.upper())
        ax.set_xticks(rounds)
        ax.set_ylim(48, 100)
        ax.grid(alpha=0.25)

    # Hide the unused 6th panel (5 models only).
    axes[-1].axis("off")

    for ax in axes[:-1]:
        ax.set_xlabel("Round")
        ax.set_ylabel("Bitwise Accuracy (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8)
    fig.suptitle("Single Image: All 8 Ciphers x 5 Models Across Rounds 1-5", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_path = os.path.join(OUT_DIR, "all_ciphers_5_models_rounds1to5_single_line_panels.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_round_table(table: Dict[str, Dict[str, float]], round_id: int) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"all_ciphers_5_models_round{round_id}_accuracy.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    return out_path


def main() -> None:
    args = parse_args()

    # Standalone all-rounds outputs.
    if args.all_rounds_grid:
        grid_path = plot_all_rounds_grid([1, 2, 3, 4, 5])
        print(f"Saved grid    -> {grid_path}")
    if args.all_rounds_heatmap:
        heatmap_path = plot_all_rounds_heatmap([1, 2, 3, 4, 5])
        print(f"Saved heatmap -> {heatmap_path}")
    if args.all_rounds_model_panels:
        panel_path = plot_all_rounds_model_panels([1, 2, 3, 4, 5])
        print(f"Saved panels  -> {panel_path}")

    # If any standalone all-round output is requested, do not also emit round bar chart.
    if args.all_rounds_grid or args.all_rounds_heatmap or args.all_rounds_model_panels:
        return

    round_id = args.round
    table = collect_round_accuracy(round_id)
    json_path = save_round_table(table, round_id)
    plot_path = plot_grouped_bars(table, round_id)
    print(f"Saved summary -> {json_path}")
    print(f"Saved graph   -> {plot_path}")
    if args.compare_with_round is not None:
        compare_path = plot_two_rounds(args.compare_with_round, round_id)
        print(f"Saved compare -> {compare_path}")


if __name__ == "__main__":
    main()
