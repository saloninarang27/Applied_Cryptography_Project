"""
================================================================================
CIPHER DATASET GENERATOR
================================================================================

OVERVIEW:
This module provides the data generation and persistence layer for all 
cryptographic experiments. It acts as the bridge between the raw cipher 
implementations in 'ciphers/' and the Machine Learning models in 'models/'.

HOW IT WORKS:
1. SAMPLING:
   - Generates random plaintexts based on the cipher's block size.
2. ENCRYPTION:
   - Uses the specific cipher's 'encrypt' function for a given round count.
3. VECTORIZATION:
   - Converts hex/integer results into binary bit-matrices (NumPy arrays) 
     suitable for neural network training.
4. PERSISTENCE:
   - Saves X (plaintext) and y (ciphertext) as .npy files, along with 
     a JSON metadata file for reproducibility.

DIRECTORY STRUCTURE:
- data/{cipher}/X_r{round}.npy
- data/{cipher}/y_r{round}.npy
- data/{cipher}/metadata_r{round}.json
"""

import numpy as np
import json
import os
import sys
import time
from typing import Any, Dict
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ciphers.simon import simon_encrypt, int_to_bits
from ciphers.present import present_encrypt
from ciphers.speck import speck_encrypt
from ciphers.ascon import ascon_encrypt
from ciphers.prince import prince_encrypt
from ciphers.gimli import gimli_encrypt
from ciphers.xoodoo import xoodoo_encrypt
from ciphers.tinyjambu import tinyjambu_encrypt
from ciphers.katan import katan_encrypt
from ciphers.grain128a import grain128a_encrypt
from ciphers.led import led_encrypt
from ciphers.skinny import skinny_encrypt
from ciphers.trivium import trivium_encrypt
from ciphers.chacha20 import chacha20_encrypt
from ciphers.mickey import mickey_encrypt
from ciphers.salsa20 import salsa20_encrypt
from ciphers.rectangle import rectangle_encrypt
from ciphers.aes import aes_encrypt
from ciphers.gift import gift_encrypt
from ciphers.lea import lea_encrypt


CIPHER_CONFIG: Dict[str, Dict[str, Any]] = {
    "simon": {
        "block_bits": 32,
        "num_samples": 200_000,
        "key": 0x1918,
        "encrypt": simon_encrypt,
    },
    "present": {
        "block_bits": 64,
        "num_samples": 80_000,
        "key": 0x00000000000000000000,
        "encrypt": present_encrypt,
    },
    "speck": {
        "block_bits": 32,
        "num_samples": 200_000,
        "key": [0x1918, 0x1110, 0x0908, 0x0100],
        "encrypt": speck_encrypt,
    },
    "ascon": {
        "block_bits": 64,
        "num_samples": 70_000,
        "key": [0x0123456789ABCDEF, 0x0FEDCBA987654321, 0xA1A2A3A4A5A6A7A8, 0xB1B2B3B4B5B6B7B8],
        "encrypt": ascon_encrypt,
    },
    "prince": {
        "block_bits": 64,
        "num_samples": 70_000,
        "key": [0x0123456789ABCDEF, 0xFEDCBA9876543210],
        "encrypt": prince_encrypt,
    },
    "gimli": {
        "block_bits": 64,
        "num_samples": 70_000,
        "key": [0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344],
        "encrypt": gimli_encrypt,
    },
    "xoodoo": {
        "block_bits": 64,
        "num_samples": 70_000,
        "key": [0xDEADBEEF, 0xCAFEBABE, 0x0BADF00D, 0x8BADF00D],
        "encrypt": xoodoo_encrypt,
    },
    "tinyjambu": {
        "block_bits": 64,
        "num_samples": 60_000,
        "key": [0x00112233, 0x44556677, 0x8899AABB, 0xCCDDEEFF],
        "encrypt": tinyjambu_encrypt,
    },
    "katan": {
        "block_bits": 32,
        "num_samples": 150_000,
        "key": [0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x1357],
        "encrypt": katan_encrypt,
    },
    "grain128a": {
        "block_bits": 64,
        "num_samples": 60_000,
        "key": [0x0F1E2D3C, 0x4B5A6978, 0x8796A5B4, 0xC3D2E1F0],
        "encrypt": grain128a_encrypt,
    },
    "led": {
        "block_bits": 64,
        "num_samples": 60_000,
        "key": [0x0123, 0x4567, 0x89AB, 0xCDEF],
        "encrypt": led_encrypt,
    },
    "skinny": {
        "block_bits": 64,
        "num_samples": 60_000,
        "key": [0x89AB, 0xCDEF, 0x0123, 0x4567],
        "encrypt": skinny_encrypt,
    },
    "trivium": {
        "block_bits": 64,
        "num_samples": 60_000,
        "key": [0x12345678, 0x9ABCDEF0, 0x0F1E2D3C, 0x4B5A6978],
        "encrypt": trivium_encrypt,
    },
    "chacha20": {
        "block_bits": 64,
        "num_samples": 50_000,
        "key": [
            0x03020100, 0x07060504, 0x0B0A0908, 0x0F0E0D0C,
            0x13121110, 0x17161514, 0x1B1A1918, 0x1F1E1D1C,
        ],
        "encrypt": chacha20_encrypt,
    },
    "mickey": {
        "block_bits": 64,
        "num_samples": 60_000,
        "key": [0x0BADF00D, 0xDEADBEEF, 0xCAFEBABE, 0x13579BDF],
        "encrypt": mickey_encrypt,
    },
    "salsa20": {
        "block_bits": 64,
        "num_samples": 50_000,
        "key": [
            0x61707865, 0x3320646E, 0x79622D32, 0x6B206574,
            0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,
        ],
        "encrypt": salsa20_encrypt,
    },
    "rectangle": {
        "block_bits": 64,
        "num_samples": 60_000,
        "key": [0x0123, 0x4567, 0x89AB, 0xCDEF, 0x1357],
        "encrypt": rectangle_encrypt,
    },
    "aes": {
        "block_bits": 128,
        "num_samples": 20_000,
        "key": [0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F],
        "encrypt": aes_encrypt,
    },
    "gift": {
        "block_bits": 64,
        "num_samples": 60_000,
        "key": [0x0123, 0x4567, 0x89AB, 0xCDEF, 0x1357, 0x9BDF, 0x2468, 0xACE0],
        "encrypt": gift_encrypt,
    },
    "lea": {
        "block_bits": 128,
        "num_samples": 20_000,
        "key": [0x3C2D1E0F, 0x78695A4B, 0xB4A59687, 0xF0E1D2C3],
        "encrypt": lea_encrypt,
    },
}


def _get_cipher_config(cipher_name: str) -> Dict[str, Any]:
    cipher_name = cipher_name.lower()

    if cipher_name not in CIPHER_CONFIG:
        supported = ", ".join(sorted(CIPHER_CONFIG.keys()))
        raise ValueError(f"Unsupported cipher '{cipher_name}'. Supported: {supported}")
    return CIPHER_CONFIG[cipher_name]


def _fast_mode_enabled() -> bool:
    return os.getenv("AC_FAST_MODE", "0") == "1"


def _effective_num_samples(default_samples: int, explicit_num_samples=None, cipher_name: str = "") -> int:
    if explicit_num_samples is not None:
        return int(explicit_num_samples)
    sample_multiplier = float(os.getenv("AC_SAMPLE_MULTIPLIER", "1.35"))
    boosted_samples = int(default_samples * sample_multiplier)
    if not _fast_mode_enabled():
        return boosted_samples
    if cipher_name.lower() == "simon":
        return min(max(30_000, int(boosted_samples * 0.20)), 40_000)
    if cipher_name.lower() == "aes":
        return min(max(20_000, int(boosted_samples * 0.75)), 30_000)
    fast_samples = max(8_000, int(boosted_samples * 0.12))
    return min(fast_samples, 20_000)


def generate_dataset(cipher_name, num_rounds, num_samples=None, key=None, seed=42):
    """
    Generates a collection of plaintext-ciphertext pairs for a given cipher.
    
    This is the core data engine of the project. It handles:
    1. Configuration lookup for the specific cipher.
    2. Random sampling of plaintexts (32-bit, 64-bit, or larger).
    3. Execution of the encryption function for a specific round count.
    4. Transformation of raw integers into bit-matrices for ML training.
    """
    cfg = _get_cipher_config(cipher_name)
    block_bits = int(cfg["block_bits"])
    
    # Determine how many samples we need based on config and 'fast mode' status.
    num_samples = _effective_num_samples(cfg["num_samples"], num_samples, cipher_name=cipher_name)
    key = cfg["key"] if key is None else key
    encrypt_fn = cfg["encrypt"]

    # Initialize the random number generator for reproducibility.
    rng = np.random.default_rng(seed)
    
    # Step 1: Sample random plaintexts.
    # We use different dtypes depending on the block size to optimize memory.
    if block_bits <= 32:
        plaintexts = rng.integers(0, 2**block_bits, size=num_samples, dtype=np.uint32)
    elif block_bits <= 64:
        # np.iinfo is used to get the maximum value for a 64-bit integer.
        plaintexts = rng.integers(0, np.iinfo(np.uint64).max, size=num_samples, dtype=np.uint64)
    else:
        # For very large block sizes, we sample bytes.
        byte_len = (block_bits + 7) // 8
        mask = (1 << block_bits) - 1
        plaintexts = [int.from_bytes(rng.bytes(byte_len), "big") & mask for _ in range(num_samples)]

    # Step 2: Prepare the output matrices (num_samples x block_bits).
    # X contains input bits (plaintext), y contains output bits (ciphertext).
    X = np.zeros((num_samples, block_bits), dtype=np.uint8)
    y = np.zeros((num_samples, block_bits), dtype=np.uint8)

    # Helper for bit conversion (used specifically for the Simon cipher's layout).
    def _int_to_bits_lsb(value: int, bits: int):
        return [(int(value) >> i) & 1 for i in range(bits)]

    # Step 3: Encrypt and Vectorize.
    # We loop through every plaintext, encrypt it, and store the bits.
    for i, pt in enumerate(plaintexts):
        # Ensure the plaintext fits exactly within the block size.
        masked_pt = int(pt) & ((1 << block_bits) - 1)
        
        # Call the actual cipher implementation.
        ct = encrypt_fn(masked_pt, key, num_rounds)
        
        # Convert the integer values into bit arrays (0s and 1s).
        if cipher_name.lower() == "simon":
            # Simon uses a specific bit-order in research papers, so we match it.
            X[i] = _int_to_bits_lsb(masked_pt, block_bits)
            y[i] = _int_to_bits_lsb(ct, block_bits)
        else:
            # Standard big-endian bit conversion for all other ciphers.
            X[i] = int_to_bits(masked_pt, block_bits)
            y[i] = int_to_bits(ct, block_bits)
            
    return X, y


def save_dataset(cipher_name, num_rounds, base_dir="data"):
    """
    Generates and persists a dataset to the local file system.
    
    Outputs three files:
    - X_r{rounds}.npy: The input bit matrix.
    - y_r{rounds}.npy: The output bit matrix.
    - metadata_r{rounds}.json: Details about the generation (key, samples, etc.)
    """
    cfg = _get_cipher_config(cipher_name)
    block_bits = int(cfg["block_bits"])
    
    # Calculate how many samples to generate.
    default_samples = _effective_num_samples(cfg["num_samples"], cipher_name=cipher_name)
    key = cfg["key"]
    
    out_dir = os.path.join(base_dir, cipher_name)
    os.makedirs(out_dir, exist_ok=True)

    # User feedback: Let them know the process has started with a clean, interactive print.
    print(f"  [Data] {cipher_name.upper()} (r={num_rounds}) -> Generating {default_samples:,} samples...",
          end=" ", flush=True)
    
    t0 = time.time()
    # Generate the actual data in memory.
    X, y = generate_dataset(cipher_name, num_rounds)
    elapsed = time.time() - t0

    # Define paths for the artifacts.
    x_path = os.path.join(out_dir, f"X_r{num_rounds}.npy")
    y_path = os.path.join(out_dir, f"y_r{num_rounds}.npy")
    meta_path = os.path.join(out_dir, f"metadata_r{num_rounds}.json")

    # Step 4: Save the NumPy arrays.
    np.save(x_path, X)
    np.save(y_path, y)

    # Step 5: Save metadata for audit and reproducibility.
    meta = {
        "cipher":       cipher_name,
        "rounds":       num_rounds,
        "num_samples":  int(X.shape[0]),
        "block_bits":   block_bits,
        "key_hex":      [hex(k) for k in key] if isinstance(key, list) else hex(int(key)),
        "seed":         42,
        "X_shape":      list(X.shape),
        "y_shape":      list(y.shape),
        "gen_time_s":   round(elapsed, 2),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Final User Interaction: Show the time taken and the save location.
    print(f"done in {elapsed:.1f}s | Saved to: {out_dir}/")
    
    return X, y


def load_dataset(cipher_name, num_rounds, base_dir="data"):
    """Load a previously saved dataset."""
    out_dir = os.path.join(base_dir, cipher_name)
    X = np.load(os.path.join(out_dir, f"X_r{num_rounds}.npy"))
    y = np.load(os.path.join(out_dir, f"y_r{num_rounds}.npy"))
    return X, y


if __name__ == "__main__":
    print("=" * 55)
    print("  Dataset Generation  -  SIMON / PRESENT")
    print("=" * 55)
    for cipher in ["simon", "present", "aes"]:
        for r in [1, 2, 3, 4, 5]:
            save_dataset(cipher, r, base_dir="data")
    print("\nAll datasets saved under  data/{cipher}/")
