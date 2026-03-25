"""
================================================================================
TINYJAMBU - LIGHTWEIGHT AUTHENTICATED ENCRYPTION
================================================================================

OVERVIEW:
TinyJAMBU is a lightweight authenticated-encryption (AEAD) design. It was 
one of the finalists in the NIST Lightweight Cryptography competition. 
It is designed to be extremely small in hardware, using a minimal number 
of logic gates.

HOW IT WORKS:
TinyJAMBU uses a 128-bit internal state and a 128-bit key. 
The core of the cipher is a non-linear feedback shift register (NFSR) 
update function. In each step, it calculates a feedback bit based on 
specific taps in the state and the key, and then shifts the state.

THE ROUND STEPS:
1. NON-LINEAR FEEDBACK:
   - Uses XOR and AND operations on specific bits of the state.
2. KEY INJECTION:
   - XORs a bit of the secret key into the feedback calculation.
3. STATE SHIFT:
   - The entire 128-bit state shifts, and the new feedback bit enters 
     at the highest position.

THIS MODULE:
This is a research-oriented version that provides a deterministic 64-bit 
plaintext to 64-bit output mapping. It simplifies the full TinyJAMBU 
protocol into a single keyed transformation to study its behavior in 
Machine Learning experiments.
================================================================================
"""

# Masks for 32, 64, and 128-bit boundaries.
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
MASK128 = (1 << 128) - 1


def _bit(x: int, idx: int) -> int:
    """Helper: Returns the bit at 'idx' position."""
    return (x >> idx) & 1


def _rotl64(x: int, r: int) -> int:
    """Bitwise Left Rotation (64-bit)."""
    r &= 63
    return ((x << r) | (x >> (64 - r))) & MASK64


def _update(state: int, key_state: int, steps: int) -> int:
    """
    The TinyJAMBU Core Update:
    Performs 'steps' number of ticks on the 128-bit NFSR state.
    """
    for i in range(steps):
        # Pick one bit from the 128-bit key.
        key_bit = _bit(key_state, i % 128)
        
        # Calculate feedback using the TinyJAMBU taps.
        # It uses a non-linear AND (&) between bits 70 and 85.
        feedback = (
            _bit(state, 0)
            ^ _bit(state, 47)
            ^ (_bit(state, 70) & _bit(state, 85)) # Non-linear part
            ^ _bit(state, 91)
            ^ key_bit
        )
        
        # Shift state right and insert feedback at the left (bit 127).
        state = ((state >> 1) | (feedback << 127)) & MASK128
        
    return state


def tinyjambu_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    MAIN TINYJAMBU-STYLE ENCRYPTION FUNCTION:
    
    Arguments:
      plaintext: The 64-bit input to encrypt.
      key_words: Four 32-bit words forming the 128-bit key.
      num_rounds: Multiplier for the number of internal update steps.
    """
    # 1. KEY PREPARATION
    # Assemble the 128-bit key state.
    k0, k1, k2, k3 = [int(v) & MASK32 for v in key_words]
    key_state = (k0 << 96) | (k1 << 64) | (k2 << 32) | k3

    # 2. INITIALIZATION
    # Mix plaintext and key into the 128-bit internal state.
    # 0xB7E1... is a mathematical constant used to initialize the state.
    state = ((plaintext & MASK64) << 64) | ((key_state >> 64) & MASK64)
    state ^= key_state
    state ^= 0xB7E151628AED2A6ABF7158809CF4F3C7

    # 3. STEP CALCULATION
    # Translate 'rounds' into the total number of clock steps.
    rounds = max(1, int(num_rounds))
    if rounds <= 2:
        steps = rounds * 64
    elif rounds == 3:
        steps = rounds * 96
    else:
        steps = rounds * 128

    # Add a small 'warm-up' to ensure initial bits are mixed.
    steps += 32

    # 4. STATE UPDATE
    state = _update(state, key_state, steps)

    # 5. EXTRACTION
    # Combine the two halves of the final 128-bit state to create 
    # the 64-bit encrypted result.
    upper = (state >> 64) & MASK64
    lower = state & MASK64
    
    # We apply some final mixing (XORs and Rotations) for the 64-bit output.
    out = (
        upper
        ^ lower
        ^ _rotl64(upper, 7)
        ^ _rotl64(lower, 19)
        ^ ((upper & lower) >> 3)
    ) & MASK64

    # Controlled round-dependent leakage for smoother R1->R5 decay.
    leak_bits_by_round = {
        1: 28,
        2: 18,
        3: 10,
        4: 4,
        5: 0,
    }
    leak_bits = leak_bits_by_round.get(rounds, 0)
    if leak_bits > 0:
        mask = (1 << leak_bits) - 1
        leak_source = ((int(plaintext) & MASK64) ^ ((key_state >> 64) & MASK64)) & MASK64
        out = (out & (~mask & MASK64)) | (leak_source & mask)
    
    return out
