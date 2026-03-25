"""
================================================================================
GIMLI - HIGH-PERFORMANCE PERMUTATION
================================================================================

OVERVIEW:
Gimli is a 384-bit cryptographic permutation designed for speed and 
security across many different types of computer processors. It was 
developed by a team of world-class cryptographers (including Daniel J. 
Bernstein).

HOW IT WORKS:
Gimli arranges its 384 bits as a 3x4 matrix of 32-bit words.
It repeatedly applies a 'Permutation' that swaps and mixes these words.

THE ROUND STEPS:
1. NON-LINEAR LAYER:
   - A sequence of rotations, bitwise ANDs, ORs, and XORs that mix three 
     words in a column together.
2. LINEAR LAYER (Every 2nd or 4th round):
   - Swaps words between columns to ensure that information from one 
     column spreads to the others.
3. CONSTANT INJECTION (Every 4th round):
   - Adds a unique number to the state to prevent mathematical patterns 
     from forming.

THIS MODULE:
This code provides a 64-bit mapping using Gimli. We inject a plaintext 
and a key into the Gimli state, run the permutation, and then extract 
the result. This is used to test how 'random' the output looks to a 
Machine Learning model after a certain number of rounds.
================================================================================
"""

# Masks to keep numbers within 32-bit or 64-bit boundaries.
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1


def _rotl32(x: int, r: int) -> int:
    """
    Bitwise Left Rotation:
    Moves bits of a 32-bit number 'x' to the left by 'r' places.
    Bits falling off the left side wrap around to the right.
    """
    return ((x << r) | (x >> (32 - r))) & MASK32


def _gimli_round(state, rnd: int):
    """
    The Core Gimli Transformation:
    This function processes the 12-word (384-bit) state.
    """
    # 1. NON-LINEAR MIXING
    # We process each of the 4 columns of the matrix.
    for c in range(4):
        # Extract three words (x, y, z) from the current column 'c'.
        x = _rotl32(state[c], 24)
        y = _rotl32(state[4 + c], 9)
        z = state[8 + c]

        # Perform the "Gimli SP-box" logic.
        # This is a complex mix of bitwise operations.
        state[8 + c] = (x ^ (z << 1) ^ ((y & z) << 2)) & MASK32
        state[4 + c] = (y ^ x ^ ((x | z) << 1)) & MASK32
        state[c] = (z ^ y ^ ((x & y) << 3)) & MASK32

    # 2. COLUMN SWAPPING AND CONSTANTS
    # These steps happen only in specific rounds to ensure maximum mixing.
    
    # Every 4th round (0, 4, 8...):
    if rnd % 4 == 0:
        # Swap words in the first row.
        state[0], state[1] = state[1], state[0]
        state[2], state[3] = state[3], state[2]
        # Inject a round constant to word 0.
        state[0] ^= 0x9E377900 ^ rnd
        
    # Every 2nd round (2, 6, 10...):
    elif rnd % 4 == 2:
        # Swap words in a different pattern.
        state[0], state[2] = state[2], state[0]
        state[1], state[3] = state[3], state[1]


def gimli_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    MAIN GIMLI-STYLE ENCRYPTION FUNCTION:
    
    Arguments:
      plaintext: The 64-bit message to encrypt.
      key_words: Four 32-bit words forming the secret key.
      num_rounds: How many permutation rounds to run.
    """
    # 1. SETUP
    # Prepare the key and split the plaintext into two 32-bit halves.
    k = [int(v) & MASK32 for v in key_words]
    pt_hi = (int(plaintext) >> 32) & MASK32
    pt_lo = int(plaintext) & MASK32

    # Initialize the 12-word Gimli state.
    # We mix the plaintext and key into the starting state.
    state = [0] * 12
    state[0] = pt_hi ^ k[0]
    state[1] = pt_lo ^ k[1]
    state[2] = k[2]
    state[3] = k[3]
    state[4] = _rotl32(pt_hi, 7) ^ k[1]
    state[5] = _rotl32(pt_lo, 11) ^ k[2]
    state[6] = k[0] ^ 0x9E3779B9 # A famous constant (part of the Golden Ratio)
    state[7] = k[1] ^ 0x7F4A7C15
    state[8] = k[2] ^ pt_lo
    state[9] = k[3] ^ pt_hi
    state[10] = 0x243F6A88 ^ k[0] ^ k[2]
    state[11] = 0x85A308D3 ^ k[1] ^ k[3]

    # 2. ROUND EXECUTION
    rounds = max(1, int(num_rounds))
    for r in range(rounds):
        # Apply the Gimli permutation.
        _gimli_round(state, r)
        
        # EXTRA KEY INJECTION (For Research)
        # We periodically XOR parts of the key back into the state.
        inject_idx = r % 12
        mix_idx = (r + 5) % 12
        state[inject_idx] ^= (k[r % 4] + ((r + 1) * 0x9E3779B9)) & MASK32
        state[mix_idx] = _rotl32(state[mix_idx] ^ state[inject_idx], (r % 13) + 1)

    # 3. EXTRACTION
    # Combine specific words from the final state to create the 64-bit output.
    out_hi = (state[0] ^ state[4] ^ state[8]) & MASK32
    out_lo = (state[1] ^ state[5] ^ state[9]) & MASK32
    
    # Merge the two 32-bit halves into one 64-bit number.
    out = ((out_hi & MASK32) << 32) | (out_lo & MASK32)
    return out & MASK64
