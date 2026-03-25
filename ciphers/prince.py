"""
================================================================================
PRINCE - LOW-LATENCY BLOCK CIPHER
================================================================================

OVERVIEW:
PRINCE is a 64-bit block cipher designed in 2012 for "low-latency" applications. 
Latency refers to the time it takes to encrypt a single block. PRINCE is 
built so that the encryption can be done in a single clock cycle in hardware, 
making it incredibly fast for real-time systems.

HOW IT WORKS:
PRINCE uses a unique "α-reflection" structure. This means that the decryption 
process is almost identical to the encryption process, which saves space in 
hardware.

THE ROUND STEPS:
Each round involves:
1. KEY XOR:
   - A portion of the 128-bit key is mixed into the state.
2. S-BOX LAYER:
   - Uses a 4-bit S-box (similar to PRESENT) to provide non-linearity.
3. LINEAR LAYER (M-Layer):
   - Multiplies the state by a specific matrix to provide diffusion. 
     In this implementation, we use a simplified bit-rotation layer to 
     mimic this effect.

THIS MODULE:
This code provides a keyed mapping inspired by the PRINCE structure. It 
allows for variable 'num_rounds' to study how the security of the mapping 
strengthens as more layers are added.
================================================================================
"""

# Mask to keep numbers within 64 bits.
MASK64 = (1 << 64) - 1

# PRINCE S-BOX: A 4-bit lookup table.
SBOX4 = [
    0xB, 0xF, 0x3, 0x2,
    0xA, 0xC, 0x9, 0x1,
    0x6, 0x7, 0x8, 0x0,
    0xE, 0x5, 0xD, 0x4,
]


def _sbox_layer(x: int) -> int:
    """
    S-Box Layer:
    Processes the 64-bit state in 16 4-bit nibbles, replacing each 
    using the SBOX4 table.
    """
    y = 0
    for i in range(16):
        # Extract a nibble.
        n = (x >> (4 * i)) & 0xF
        # Replace and put back.
        y |= SBOX4[n] << (4 * i)
    return y


def _lin_layer(x: int) -> int:
    """
    Linear Layer:
    Provides diffusion by rotating the bits of the state and XORing 
    them back into itself.
    """
    # Rotate by 19 bits and XOR.
    x ^= ((x << 19) | (x >> (64 - 19))) & MASK64
    # Rotate by 28 bits and XOR.
    x ^= ((x << 28) | (x >> (64 - 28))) & MASK64
    return x & MASK64


def prince_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    MAIN PRINCE-STYLE ENCRYPTION FUNCTION:
    
    Arguments:
      plaintext: The 64-bit block to encrypt.
      key_words: Two 64-bit values forming the 128-bit key.
      num_rounds: Number of mixing rounds.
    """
    # 1. SETUP
    # Extract keys k0 and k1.
    k0 = int(key_words[0]) & MASK64
    k1 = int(key_words[1]) & MASK64

    # Initial XOR with key k0 (Pre-whitening).
    state = (plaintext ^ k0) & MASK64

    # 2. ROUND LOOP
    rounds = max(1, int(num_rounds))
    for r in range(rounds):
        # STEP A: KEY & CONSTANT ADDITION
        # Mix in key k1 and a mathematical constant based on the Golden Ratio 
        # to ensure each round is unique.
        state ^= (k1 ^ ((0x9E3779B97F4A7C15 * (r + 1)) & MASK64))
        
        # STEP B: SUBSTITUTION
        state = _sbox_layer(state)
        
        # STEP C: LINEAR MIXING
        state = _lin_layer(state)

    # 3. FINALIZATION
    # Final XOR with key k0 (Post-whitening).
    state ^= k0
    return state & MASK64
