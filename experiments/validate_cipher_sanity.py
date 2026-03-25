"""
================================================================================
CIPHER SANITY VALIDATOR - DIAGNOSTIC TOOL FOR DATA/IMPLEMENTATION QUALITY
================================================================================

OVERVIEW:
This script performs safety checks before trusting model results in the report.
It is designed to catch subtle issues such as:

1) Implementation problems in cipher functions (non-determinism, weak key effect)
2) Dataset leakage patterns (ciphertext accidentally resembling plaintext)
3) Model metric anomalies (near-perfect behavior or suspiciously random behavior)

WHY THIS EXISTS:
In ML-based cryptanalysis, high accuracy can be caused by true structure OR by
pipeline bugs. Similarly, near-random behavior may be healthy diffusion OR a
broken cipher path. This validator gives quick, consistent checks across ciphers
to separate healthy signals from suspicious ones.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.generate_dataset import CIPHER_CONFIG


def _hamming(a: int, b: int, bits: int) -> int:
    """Return Hamming distance between two integers limited to `bits` width.

    We XOR values, mask to active bit-width, then count set bits.
    """
    # Step 1: XOR marks positions where bits differ.
    diff = a ^ b
    # Step 2: Keep only the active block width.
    masked = diff & ((1 << bits) - 1)
    # Step 3: Count differing positions.
    return masked.bit_count()


def _load_summary(metrics_dir: Path, cipher: str, model: str) -> dict | None:
    """Load summary JSON for one (cipher, model) pair if present.

    Returns:
    - Parsed dictionary when file exists.
    - None when file is missing.
    """
    path = metrics_dir / f"{cipher}_{model}_summary.json"
    # Step 1: Guard if artifact does not exist yet.
    if not path.exists():
        return None
    # Step 2: Read JSON text.
    raw = path.read_text(encoding="utf-8")
    # Step 3: Parse JSON payload.
    return json.loads(raw)


def _determinism_and_key_influence(cipher: str, rounds: list[int]) -> dict:
    """Probe core cipher behavior across rounds.

    Checks:
    - Determinism: same input/key/round must always produce same output.
    - Output diversity: number of unique outputs over a small plaintext set.
    - Key influence: fraction of test plaintexts changed when flipping one key bit
      in each key word.
    - Plaintext↔ciphertext average HD for a quick diffusion snapshot.
    """
    # Step 1: Read cipher configuration from central registry.
    cfg = CIPHER_CONFIG[cipher]
    # Step 2: Extract callable encryption function.
    encrypt = cfg["encrypt"]
    # Step 3: Extract declared block size.
    bits = int(cfg["block_bits"])
    # Step 4: Normalize key words to plain Python ints.
    key = [int(v) for v in cfg["key"]]

    # A compact but diverse plaintext probe set:
    # - small values
    # - edge values
    # - structured patterns
    test_pts = [
        0,
        1,
        2,
        3,
        (1 << (bits - 1)) - 1,
        (1 << (bits - 1)),
        (1 << bits) - 1,
        0x123456789ABCDEF & ((1 << bits) - 1),
        0x0F0F0F0F0F0F0F0F & ((1 << bits) - 1),
        0xAAAAAAAAAAAAAAAA & ((1 << bits) - 1),
    ]

    # Container keyed by round number.
    round_info: dict[int, dict] = {}
    for r in rounds:
        # Baseline outputs for this round.
        outs = [encrypt(p, key, r) for p in test_pts]

        # Determinism sanity check.
        deterministic = all(encrypt(p, key, r) == encrypt(p, key, r) for p in test_pts)

        # Flip one LSB per key word and measure how often ciphertext changes.
        # Step-by-step key sensitivity profile per key word.
        key_influence = []
        for idx in range(len(key)):
            # Copy base key.
            k2 = key[:]
            # Flip one bit (LSB) in one word.
            k2[idx] ^= 0x1
            # Count affected plaintext probes.
            changed = sum(1 for p in test_pts if encrypt(p, key, r) != encrypt(p, k2, r))
            # Convert count to ratio in [0,1].
            key_influence.append(changed / len(test_pts))

        # Mean Hamming distance between plaintext and ciphertext on probe set.
        avg_hd_pt_ct = mean(_hamming(p, c, bits) for p, c in zip(test_pts, outs))

        # Save one compact round record.
        round_info[r] = {
            "deterministic": deterministic,
            "unique_outputs": len(set(outs)),
            "avg_hd_pt_ct": avg_hd_pt_ct,
            "key_influence_ratio_per_word": key_influence,
        }

    # Return consolidated object for this cipher.
    return {
        "cipher": cipher,
        "block_bits": bits,
        "rounds": round_info,
    }


def _dataset_copy_plaintext_signal(data_dir: Path, cipher: str, rounds: list[int]) -> dict[int, dict]:
    """Check dataset-level plaintext-copy leakage indicators.

    For each round dataset:
    - `copy_plaintext_bit_acc`: bitwise match rate between X and y.
    - `avg_hd`: average per-sample Hamming distance between X and y.

    If copy accuracy is unexpectedly high, dataset creation may be leaking signal.
    """
    # Output keyed by round -> leakage indicators.
    out: dict[int, dict] = {}
    cipher_dir = data_dir / cipher
    # If cipher dataset folder is absent, return empty result.
    if not cipher_dir.exists():
        return out

    for r in rounds:
        x_path = cipher_dir / f"X_r{r}.npy"
        y_path = cipher_dir / f"y_r{r}.npy"
        # Skip rounds that are not generated.
        if not x_path.exists() or not y_path.exists():
            continue

        # Step 1: Load inputs/labels.
        x = np.load(x_path)
        y = np.load(y_path)
        # Step 2: Compute bitwise equality rate if labels accidentally copy inputs.
        bit_acc_if_copy = float((x == y).mean())
        # Step 3: Compute average Hamming distance across samples.
        avg_hd = float((x != y).sum(axis=1).mean())

        out[r] = {
            "copy_plaintext_bit_acc": bit_acc_if_copy,
            "avg_hd": avg_hd,
            "bits": int(x.shape[1]),
            "samples": int(x.shape[0]),
        }

    return out


def _model_anomalies(metrics_dir: Path, cipher: str) -> list[str]:
    """Flag model-level suspicious patterns from stored summaries.

    Alerts currently include:
    - Near-perfect R1: potentially too good to be true (leakage/shortcut risk)
    - Near-random across all rounds: possible over-diffusion or broken pipeline
    """
    # Collect all anomaly text entries for final report.
    alerts: list[str] = []
    models = ["logistic", "mlp", "cnn", "mine", "random_forest"]

    for model in models:
        summary = _load_summary(metrics_dir, cipher, model)
        # If model summary is missing or does not include R1, skip.
        if not summary or "1" not in summary:
            continue

        r1 = summary["1"]
        block_bits = int(r1.get("block_bits", 0))
        acc = float(r1["bitwise_accuracy"])
        hd = float(r1["hamming_distance"])

        # Alert Rule A: suspiciously near-perfect at R1.
        if acc >= 0.98 and hd <= max(1.0, block_bits * 0.05):
            alerts.append(
                f"{cipher}:{model}:R1 near-perfect learnability (acc={acc:.4f}, hd={hd:.3f}/{block_bits})"
            )

        rounds = sorted(int(k) for k in summary.keys() if str(k).isdigit())
        if rounds:
            # Alert Rule B: model stays near-random for all available rounds.
            near_random_all = all(abs(float(summary[str(r)]["bitwise_accuracy"]) - 0.5) <= 0.01 for r in rounds)
            if near_random_all:
                alerts.append(
                    f"{cipher}:{model}:all rounds near random (possible over-diffusion or implementation issue)"
                )

    return alerts


def main() -> None:
    """CLI entry point for sanity diagnostics.

    Typical usage:
    - Default: checks speck/trivium/tinyjambu/katan rounds 1..5
    - Custom: pass --ciphers and --rounds for focused validation
    """
    # Step 1: Build CLI parser.
    parser = argparse.ArgumentParser(description="Cipher sanity diagnostics for report validation")
    parser.add_argument(
        "--ciphers",
        nargs="*",
        default=["speck", "trivium", "tinyjambu", "katan"],
        help="Cipher names to validate",
    )
    parser.add_argument("--rounds", nargs="*", type=int, default=[1, 2, 3, 4, 5], help="Rounds to probe")
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--data-dir", default="data")
    # Step 2: Parse CLI arguments.
    args = parser.parse_args()

    # Step 3: Resolve core paths.
    metrics_dir = Path(args.metrics_dir)
    data_dir = Path(args.data_dir)

    # Global header so logs are easy to parse in CI/terminal output.
    print("=== Cipher Sanity Diagnostics ===")
    print(f"ciphers={args.ciphers} rounds={args.rounds}\n")

    # Cross-cipher alert accumulator.
    global_alerts: list[str] = []

    for cipher in args.ciphers:
        # Skip unknown ciphers gracefully.
        if cipher not in CIPHER_CONFIG:
            print(f"[SKIP] Unknown cipher: {cipher}")
            continue

        # Run three independent check families.
        print(f"--- {cipher.upper()} ---")
        core = _determinism_and_key_influence(cipher, args.rounds)
        ds = _dataset_copy_plaintext_signal(data_dir, cipher, args.rounds)
        alerts = _model_anomalies(metrics_dir, cipher)

        # Print round-by-round core metrics.
        # Render per-round diagnostics.
        for r in args.rounds:
            if r not in core["rounds"]:
                continue
            info = core["rounds"][r]
            key_inf = ", ".join(f"k{i}:{v:.2f}" for i, v in enumerate(info["key_influence_ratio_per_word"]))
            print(
                f"R{r}: deterministic={info['deterministic']} unique={info['unique_outputs']}/10 "
                f"avgHD(pt,ct)={info['avg_hd_pt_ct']:.2f} keyInfluence[{key_inf}]"
            )

            # Render dataset leakage diagnostics if this round has .npy artifacts.
            if r in ds:
                ds_info = ds[r]
                print(
                    f"    dataset: copy-pt-acc={ds_info['copy_plaintext_bit_acc']:.4f} "
                    f"avgHD={ds_info['avg_hd']:.3f}/{ds_info['bits']} n={ds_info['samples']}"
                )

        # Emit and accumulate alerts for summary.
        if alerts:
            for alert in alerts:
                print(f"ALERT: {alert}")
                global_alerts.append(alert)
        else:
            print("No model-level anomaly alerts triggered.")

        print()

    # Final compact summary for quick pass/fail signal.
    print("=== Summary ===")
    # Final pass/fail style output.
    if global_alerts:
        print(f"ALERT COUNT: {len(global_alerts)}")
        for alert in global_alerts:
            print(f"- {alert}")
    else:
        print("No alerts triggered.")


if __name__ == "__main__":
    main()
