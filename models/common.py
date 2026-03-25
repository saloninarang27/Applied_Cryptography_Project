"""
================================================================================
SHARED UTILITIES FOR CRYPTOGRAPHIC MODELING
================================================================================

OVERVIEW:
This module provides a unified interface for data handling, metric 
calculation, and model persistence across all supported architectures. 
By centralizing these functions, we ensure that every model—from a simple 
Logistic Regression to a complex CNN—is evaluated fairly using the exact 
same procedures.

CORE COMPONENTS:
1. DATASET MANAGEMENT:
   - Handles the loading and splitting of binary bit-matrices into training 
     and validation sets.
2. BITWISE METRICS:
   - Implements specialized metrics like bitwise accuracy and Hamming 
     distance, which are more meaningful than standard cross-entropy 
     for cryptographic analysis.
3. FORMATTED PERSISTENCE:
   - Saves model results and evaluation summaries in a standardized JSON 
     format, which is then read by the reporting tools.

NOTE ON FAIRNESS:
While this module contains legacy feature engineering tools, the active 
training path currently uses raw bits only. This ensures that the results 
reflect the inherent difficulty of the cipher, not the quality of manually 
engineered features.
"""

import json
import os
import time
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from data.generate_dataset import load_dataset, save_dataset
from ciphers.aes import SBOX as AES_SBOX
from ciphers.present import SBOX as PRESENT_SBOX
from ciphers.simon import bits_to_int, int_to_bits


DEFAULT_CIPHER = "simon"


def load_or_generate_dataset(
    num_rounds: int,
    data_dir: str = "data",
    cipher: str = DEFAULT_CIPHER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load pre-generated dataset for a cipher/round, or generate if missing.

    This provides a stable, single entry point for all model scripts.
    """
    # Step 1: Try loading an existing dataset artifact.
    try:
        X, y = load_dataset(cipher, num_rounds, base_dir=data_dir)
        print(f"  Dataset loaded from data/{cipher}/")
        return X, y
    except FileNotFoundError:
        # Step 2 (fallback): generate and persist dataset if not present.
        print("  Dataset not found - generating ...")
        return save_dataset(cipher, num_rounds, base_dir=data_dir)


def infer_block_bits(y: np.ndarray) -> int:
    """Infer cipher block size from label tensor shape.

    Convention: labels are stored as [n_samples, block_bits].
    """
    # Labels are expected to be [samples, block_bits].
    return int(y.shape[1])


def get_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic train/test split for reproducible experiments."""
    # Keep split deterministic via fixed random_state.
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def _byte_features_from_bits(X_bits: np.ndarray) -> np.ndarray:
    """Build generic byte-level interactions from bit arrays.

    Output is float32 bit features derived from:
    - byte XOR with 1-byte roll
    - byte XOR with 2-byte roll
    - byte AND with 1-byte roll
    """
    # Step 1: Read input shape.
    n_samples, block_bits = X_bits.shape
    # Step 2: Require byte-aligned block size.
    if block_bits % 8 != 0:
        return np.empty((n_samples, 0), dtype=np.float32)

    # Step 3: Reinterpret bits as bytes.
    n_bytes = block_bits // 8
    as_bytes = np.packbits(X_bits.reshape(n_samples, n_bytes, 8), axis=2, bitorder="big").reshape(n_samples, n_bytes)

    # Step 4: Build local neighbor views.
    roll1 = np.roll(as_bytes, 1, axis=1)
    roll2 = np.roll(as_bytes, 2, axis=1)

    # Step 5: Compute simple byte interactions.
    byte_xor_1 = np.bitwise_xor(as_bytes, roll1)
    byte_xor_2 = np.bitwise_xor(as_bytes, roll2)
    byte_and_1 = np.bitwise_and(as_bytes, roll1)

    # Step 6: Stack and unpack interactions back to bit-level features.
    stacked = np.concatenate([byte_xor_1, byte_xor_2, byte_and_1], axis=1).astype(np.uint8, copy=False)
    unpacked = np.unpackbits(stacked[:, :, None], axis=2, bitorder="big").reshape(n_samples, -1)
    return unpacked.astype(np.float32, copy=False)


def _aes_structural_features(X_bits: np.ndarray) -> np.ndarray:
    """Compute AES-inspired structural transforms from 128-bit input bits.

    Includes approximate stages:
    - AddRoundKey proxy
    - SubBytes via AES S-Box
    - ShiftRows permutation
    - Local mixing proxy
    """
    # Step 1: Validate AES-compatible block width.
    n_samples, block_bits = X_bits.shape
    if block_bits != 128:
        return np.empty((n_samples, 0), dtype=np.float32)

    # Step 2: Convert bits -> 16 plaintext bytes.
    n_bytes = 16
    pt_bytes = np.packbits(X_bits.reshape(n_samples, n_bytes, 8), axis=2, bitorder="big").reshape(n_samples, n_bytes)

    # Step 3: Build fixed key bytes used for structural proxy features.
    aes_key_words = [0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F]
    key_bytes = []
    for word in aes_key_words:
        key_bytes.extend(int(word).to_bytes(4, "big"))
    key_arr = np.array(key_bytes, dtype=np.uint8).reshape(1, 16)

    # Step 4: Approximate AddRoundKey stage.
    add_round_key = np.bitwise_xor(pt_bytes, key_arr)

    # Step 5: Approximate SubBytes stage.
    sbox = np.array(AES_SBOX, dtype=np.uint8)
    sub_bytes = sbox[add_round_key]

    # Step 6: Approximate ShiftRows stage.
    shift_perm = np.array([0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11], dtype=np.int64)
    shift_rows = sub_bytes[:, shift_perm]

    # Step 7: Local byte-mixing proxy.
    local_mix = np.bitwise_xor(shift_rows, np.roll(shift_rows, 1, axis=1))

    # Step 8: Pack all stages and unfold to bit features.
    stacked = np.concatenate([add_round_key, sub_bytes, shift_rows, local_mix], axis=1)
    unpacked = np.unpackbits(stacked[:, :, None], axis=2, bitorder="big").reshape(n_samples, -1)
    return unpacked.astype(np.float32, copy=False)


def _speck_arx_features(X_bits: np.ndarray) -> np.ndarray:
    """Extract ARX-style (Add/Rotate/Xor) features for 32-bit Speck-like layout."""
    # Step 1: Validate Speck-32 style block width.
    n_samples, block_bits = X_bits.shape
    if block_bits != 32:
        return np.empty((n_samples, 0), dtype=np.float32)

    # Step 2: Convert bits into two 16-bit words.
    words = np.packbits(X_bits.reshape(n_samples, 2, 16), axis=2, bitorder="big").reshape(n_samples, 2, 2)
    words16 = ((words[:, :, 0].astype(np.uint16) << 8) | words[:, :, 1].astype(np.uint16)).astype(np.uint16)

    left = words16[:, 0]
    right = words16[:, 1]

    def _rol16(arr: np.ndarray, r: int) -> np.ndarray:
        return ((arr << r) | (arr >> (16 - r))) & 0xFFFF

    def _ror16(arr: np.ndarray, r: int) -> np.ndarray:
        return ((arr >> r) | (arr << (16 - r))) & 0xFFFF

    # Step 3: Construct ARX proxies used by Speck-like rounds.
    left_ror7 = _ror16(left, 7)
    right_rol2 = _rol16(right, 2)
    add_lr = (left_ror7 + right) & 0xFFFF
    xor_lr = left ^ right
    carry_proxy = ((left_ror7 & right) << 1) & 0xFFFF
    mix = (xor_lr ^ add_lr) & 0xFFFF
    right_mix = (right_rol2 ^ add_lr) & 0xFFFF

    # Step 4: Stack words and unfold to bit-level matrix.
    feat_words = np.stack(
        [left, right, left_ror7, right_rol2, add_lr, xor_lr, carry_proxy, mix, right_mix],
        axis=1,
    ).astype(np.uint16, copy=False)

    feat_bytes = np.stack([(feat_words >> 8) & 0xFF, feat_words & 0xFF], axis=2).astype(np.uint8, copy=False)
    unpacked = np.unpackbits(feat_bytes, axis=2, bitorder="big").reshape(n_samples, -1)
    return unpacked.astype(np.float32, copy=False)


def _present_structural_features(X_bits: np.ndarray) -> np.ndarray:
    """Extract PRESENT-like nonlinear/permutation features from 64-bit inputs."""
    # Step 1: Validate PRESENT-compatible width.
    n_samples, block_bits = X_bits.shape
    if block_bits != 64:
        return np.empty((n_samples, 0), dtype=np.float32)

    # Step 2: Convert to 16 nibbles (4-bit chunks).
    nibbles = (
        np.packbits(X_bits.reshape(n_samples, 16, 4), axis=2, bitorder="big").reshape(n_samples, 16) >> 4
    ).astype(np.uint8, copy=False)

    # Step 3: Apply PRESENT S-box to each nibble.
    present_sbox = np.array(PRESENT_SBOX, dtype=np.uint8)
    sboxed_nibbles = present_sbox[nibbles]

    # Step 4: Expand sboxed nibbles back into bit representation.
    nibble_bit_positions = np.array([3, 2, 1, 0], dtype=np.uint8)
    sboxed_bits = ((sboxed_nibbles[:, :, None] >> nibble_bit_positions) & 1).reshape(n_samples, 64).astype(
        np.uint8, copy=False
    )

    # Step 5: Build PRESENT p-layer permutation map.
    perm = np.zeros(64, dtype=np.int64)
    for i in range(63):
        perm[i] = (16 * i) % 63
    perm[63] = 63
    # Step 6: Apply permutation and derive local interactions.
    p_layer_bits = sboxed_bits[:, perm]

    p_roll1 = np.roll(p_layer_bits, 1, axis=1)
    p_roll4 = np.roll(p_layer_bits, 4, axis=1)
    p_mix_xor = np.bitwise_xor(p_layer_bits, p_roll1)
    p_mix_and = np.bitwise_and(p_layer_bits, p_roll4)

    # Step 7: Derive nibble-level xor signal and convert to bits.
    nibble_xor = np.bitwise_xor(sboxed_nibbles, np.roll(sboxed_nibbles, 1, axis=1)).astype(np.uint8, copy=False)
    nibble_xor_bits = ((nibble_xor[:, :, None] >> nibble_bit_positions) & 1).reshape(n_samples, 64).astype(
        np.uint8, copy=False
    )

    # Step 8: Concatenate all bit feature families.
    features = np.concatenate([sboxed_bits, p_layer_bits, p_mix_xor, p_mix_and, nibble_xor_bits], axis=1)
    return features.astype(np.float32, copy=False)


def _trivium_structural_features(X_bits: np.ndarray) -> np.ndarray:
    """Extract Trivium-specific shift and feedback feedback features from 64-bit plaintext.
    
    Targets the exact operations from trivium_encrypt:
    - Right shifts: 63, 62, 61, 60, 58, 57, 55, 53, 52, 49, 46, 41
    - Rotations: 7, 19 (output mixing)
    - AND gates between shifted positions
    """
    # Step 1: Validate Trivium probe width.
    n_samples, block_bits = X_bits.shape
    if block_bits != 64:
        return np.empty((n_samples, 0), dtype=np.float32)

    # Exact shift positions used in the cipher feedback path.
    shift_positions = np.array([63, 62, 61, 60, 58, 57, 55, 53, 52, 49, 46, 41], dtype=np.int64) % 64
    
    # Extract bits at key tap positions.
    shifted_bits = X_bits[:, shift_positions].astype(np.uint8, copy=False)
    
    # AND combinations between consecutive tap positions (feedback proxy).
    and_features = []
    for i in range(0, len(shift_positions) - 1, 2):
        and_result = np.bitwise_and(
            X_bits[:, shift_positions[i]], 
            X_bits[:, shift_positions[i+1]]
        ).astype(np.uint8, copy=False)
        and_features.append(and_result.reshape(n_samples, 1))
    
    # Rotation families observed in Trivium-style update/mixing steps.
    # Step 3: Build rotation views used as mixing proxies.
    rot7 = np.roll(X_bits, 7, axis=1)
    rot19 = np.roll(X_bits, 19, axis=1)
    rot11 = np.roll(X_bits, 11, axis=1)  # initialization rotation
    rot17 = np.roll(X_bits, 17, axis=1)  # initialization rotation
    
    # XOR-based mixing proxies.
    # Step 4: XOR-based interaction families.
    mix_rot7 = np.bitwise_xor(rot7, X_bits)
    mix_rot19 = np.bitwise_xor(rot19, X_bits)
    mix_rot11 = np.bitwise_xor(rot11, X_bits)
    mix_rot17 = np.bitwise_xor(rot17, X_bits)
    
    # AND-based nonlinear proxies.
    # Step 5: AND-based nonlinear interaction families.
    and_rot7 = np.bitwise_and(rot7, X_bits)
    and_rot19 = np.bitwise_and(rot19, X_bits)
    and_rot11 = np.bitwise_and(rot11, X_bits)
    
    # Left-shift simulation (<<1) approximated via roll.
    # Step 6: Approximate left shift by rolling left one bit.
    rot_left1 = np.roll(X_bits, -1, axis=1)  # Negative roll for left shift
    mix_left1 = np.bitwise_xor(X_bits, rot_left1)
    and_left1 = np.bitwise_and(X_bits, rot_left1)
    
    # Combine all feature families into one matrix.
    # Step 7: Aggregate core feature blocks.
    features = [
        shifted_bits,  # Direct shift positions: 12 features
        mix_rot7, mix_rot19, mix_rot11, mix_rot17,  # Rotation XOR mixes: 4*64 features
        and_rot7, and_rot19, and_rot11,  # Rotation AND mixes: 3*64 features  
        mix_left1, and_left1,  # Left-shift mixes: 2*64 features
    ]
    
    # Append compact tap-pair AND features.
    if and_features:
        and_stack = np.concatenate(and_features, axis=1)
        features.append(and_stack)
    
    # Step 8: Concatenate into final dense matrix.
    result = np.concatenate(features, axis=1)
    return result.astype(np.float32, copy=False)


def _chacha20_arx_features(X_bits: np.ndarray) -> np.ndarray:
    """Extract ChaCha20-inspired ARX interaction features from 64-bit inputs."""
    # Step 1: Validate 64-bit input width.
    n_samples, block_bits = X_bits.shape
    if block_bits != 64:
        return np.empty((n_samples, 0), dtype=np.float32)

    # Step 2: Convert bits to two uint32 words.
    words = np.packbits(X_bits.reshape(n_samples, 2, 32), axis=2, bitorder="big").reshape(n_samples, 2, 4)
    words32 = (
        (words[:, :, 0].astype(np.uint32) << 24)
        | (words[:, :, 1].astype(np.uint32) << 16)
        | (words[:, :, 2].astype(np.uint32) << 8)
        | words[:, :, 3].astype(np.uint32)
    ).astype(np.uint32, copy=False)

    mask = np.uint32(0xFFFFFFFF)
    w0 = words32[:, 0]
    w1 = words32[:, 1]

    def _rol32(arr: np.ndarray, r: int) -> np.ndarray:
        return ((arr << r) | (arr >> (32 - r))) & mask

    # Step 3: Build ARX primitive interactions.
    r7_w0 = _rol32(w0, 7)
    r12_w0 = _rol32(w0, 12)
    r16_w0 = _rol32(w0, 16)
    r7_w1 = _rol32(w1, 7)
    r12_w1 = _rol32(w1, 12)
    r16_w1 = _rol32(w1, 16)

    add01 = (w0 + w1) & mask
    xor01 = w0 ^ w1
    carry01 = ((w0 & w1) << 1) & mask

    # Step 4: One compact quarter-round style proxy chain.
    a = w0
    b = w1
    c = _rol32(w0 ^ w1, 13)
    d = _rol32(w1, 8)

    a1 = (a + b) & mask
    d1 = _rol32(d ^ a1, 16)
    c1 = (c + d1) & mask
    b1 = _rol32(b ^ c1, 12)
    a2 = (a1 + b1) & mask
    d2 = _rol32(d1 ^ a2, 8)
    c2 = (c1 + d2) & mask
    b2 = _rol32(b1 ^ c2, 7)

    # Step 5: Stack all word features.
    feat_words = np.stack(
        [
            w0,
            w1,
            r7_w0,
            r12_w0,
            r16_w0,
            r7_w1,
            r12_w1,
            r16_w1,
            add01,
            xor01,
            carry01,
            a1,
            b1,
            c1,
            d1,
            a2,
            b2,
            c2,
            d2,
        ],
        axis=1,
    ).astype(np.uint32, copy=False)

    # Step 6: Convert stacked words to bit-level float features.
    feat_bytes = np.stack(
        [
            (feat_words >> 24) & 0xFF,
            (feat_words >> 16) & 0xFF,
            (feat_words >> 8) & 0xFF,
            feat_words & 0xFF,
        ],
        axis=2,
    ).astype(np.uint8, copy=False)
    unpacked = np.unpackbits(feat_bytes, axis=2, bitorder="big").reshape(n_samples, -1)
    return unpacked.astype(np.float32, copy=False)


def augment_plaintext_features(X: np.ndarray, cipher: str = "") -> np.ndarray:
    """Return raw ciphertext bits WITHOUT cipher-specific engineered features.
    
    CLEANUP (March 2026): All cipher-specific structural feature engineering has
    been REMOVED to ensure fair, unbiased cryptanalysis. This includes:
    
    - AES: S-box outputs, MixColumns patterns, byte features
    - SPECK/CHACHA20: ARX operation features
    - PRESENT: S-box and permutation patterns
    - TRIVIUM: Shift/feedback position features
    
    Models now train on identical raw bitwise representation for ALL ciphers,
    ensuring no artificially engineered signal or cipher-specific advantages.
    This is scientifically rigorous: models must learn distinguishability from
    native cipher properties, not pre-computed domain knowledge.
    """
    # Step 1: Keep policy strict: no cipher-specific feature engineering.
    # Step 2: Cast to float32 to match downstream model expectations.
    return X.astype(np.float32, copy=False)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute core evaluation metrics used in all reports.

    - bitwise_accuracy: per-bit correctness
    - hamming_distance: mean number of differing bits per sample
    - word_accuracy: exact full-block match rate
    """
    # Step 1: Bit-level agreement over all elements.
    bitwise_accuracy = float((y_true == y_pred).mean())
    # Step 2: Mean Hamming distance per sample.
    hamming_distance = float((y_true != y_pred).sum(axis=1).mean())
    # Step 3: Exact full-block (all bits) correctness per sample.
    word_accuracy = float(np.all(y_true == y_pred, axis=1).mean())

    return {
        "bitwise_accuracy": bitwise_accuracy,
        "hamming_distance": hamming_distance,
        "word_accuracy": word_accuracy,
    }


def save_metrics(metrics: Dict[str, float], cipher: str, model: str, num_rounds: int, out_dir: str = "results/metrics") -> None:
    """Persist one round's metrics in standardized JSON schema."""
    # Step 1: Ensure destination folder exists.
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{cipher}_{model}_r{num_rounds}.json"
    # Step 2: Construct consistent payload schema.
    payload = {
        "cipher": cipher,
        "model": model,
        "rounds": num_rounds,
        **metrics,
    }
    # Step 3: Write JSON file with readable formatting.
    with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Metrics saved  ->  {out_dir}/{fname}")


def save_summary(all_results: Dict[int, Dict[str, float]], cipher: str, model: str, out_dir: str = "results/metrics") -> None:
    """Persist per-model multi-round summary in standardized JSON schema."""
    # Step 1: Ensure destination folder exists.
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{cipher}_{model}_summary.json"
    # Step 2: Serialize multi-round results map.
    with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Summary saved  ->  {out_dir}/{fname}")


def timed_call(fn, *args, **kwargs):
    """Execute callable and return `(result, elapsed_seconds)`."""
    # Step 1: Capture start timestamp.
    t0 = time.time()
    # Step 2: Execute wrapped function.
    out = fn(*args, **kwargs)
    # Step 3: Return output and elapsed seconds.
    return out, time.time() - t0
