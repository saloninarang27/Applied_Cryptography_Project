"""
================================================================================
CHACHA20 - STREAM CIPHER WITH ARX DESIGN
================================================================================

OVERVIEW:
ChaCha20 is a high-speed stream cipher developed by Daniel J. Bernstein. 
It is widely used in modern internet security (like TLS/HTTPS) because it 
is very fast and secure, especially on computers that don't have special 
hardware for AES.

HOW IT WORKS:
ChaCha20 uses a design principle called ARX (Add-Rotate-XOR):
1. ADD: It uses normal 32-bit addition.
2. ROTATE: It rotates bits within a number.
3. XOR: It uses the exclusive-OR operation.

THE STATE:
ChaCha20 works on a 512-bit state, viewed as a 4x4 matrix of 32-bit words (16 
words total).
- Words 0-3: Fixed constants (the 'magic' string "expand 32-byte k").
- Words 4-11: The secret key.
- Word 12: A counter (changes with every block).
- Words 13-15: A nonce (a unique number used once).

THE QUARTER ROUND:
The basic building block is the 'Quarter Round'. It takes 4 words from the 
state and mixes them thoroughly using the ARX steps.

THIS MODULE:
This implementation uses a reduced-round ChaCha20 mapping for Machine Learning 
research. It converts a 64-bit input into a 64-bit output by running the 
ChaCha permutation for a specific number of rounds.
================================================================================
"""

# Masks to keep numbers within 32-bit or 64-bit limits.
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1

# Standard ChaCha20 constants (ASCII for "expand 32-byte k").
CONSTANTS = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574]


def _rotl32(x: int, r: int) -> int:
    """
    Bitwise Left Rotation:
    Moves bits of a 32-bit number 'x' to the left by 'r' places.
    Bits that move off the left side wrap around to the right.
    """
    return ((x << r) | (x >> (32 - r))) & MASK32


def _quarter_round(state, a: int, b: int, c: int, d: int) -> None:
    """
    The Quarter Round:
    Takes four words from the state (at indexes a, b, c, d) and mixes them.
    This is the core mathematical engine of ChaCha20.
    """
    # 1. Mix words a, b, and d
    state[a] = (state[a] + state[b]) & MASK32 # Add
    state[d] ^= state[a]                       # XOR
    state[d] = _rotl32(state[d], 16)           # Rotate

    # 2. Mix words c, d, and b
    state[c] = (state[c] + state[d]) & MASK32 # Add
    state[b] ^= state[c]                       # XOR
    state[b] = _rotl32(state[b], 12)           # Rotate

    # 3. Mix words a, b, and d again
    state[a] = (state[a] + state[b]) & MASK32 # Add
    state[d] ^= state[a]                       # XOR
    state[d] = _rotl32(state[d], 8)            # Rotate

    # 4. Mix words c, d, and b again
    state[c] = (state[c] + state[d]) & MASK32 # Add
    state[b] ^= state[c]                       # XOR
    state[b] = _rotl32(state[b], 7)            # Rotate


def chacha20_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    MAIN CHACHA20-STYLE ENCRYPTION FUNCTION:
    
    Arguments:
      plaintext: The 64-bit number to encrypt.
      key_words: Words used for the secret key.
      num_rounds: Number of mixing rounds.
    """
    # 1. INITIALIZE THE STATE
    # Convert inputs into 32-bit words.
    key = [int(v) & MASK32 for v in key_words]
    pt0 = (int(plaintext) >> 32) & MASK32
    pt1 = int(plaintext) & MASK32
    
    # We create a counter and nonce based on the plaintext (for research purposes).
    counter = (pt0 ^ 0xDEADBEEF) & MASK32
    nonce = [pt1, _rotl32(pt0, 7), _rotl32(pt1, 13)]

    # Assemble the 16-word starting state.
    state = CONSTANTS + key + [counter] + nonce
    # Create a working copy of the state to modify.
    working = state[:]

    # 2. THE ROUND LOOP
    # ChaCha usually does 20 rounds. We do 'num_rounds'.
    rounds = max(1, int(num_rounds))
    for i in range(rounds):
        # Column Rounds: Mix vertical sets of words.
        _quarter_round(working, 0, 4, 8, 12)
        _quarter_round(working, 1, 5, 9, 13)
        _quarter_round(working, 2, 6, 10, 14)
        _quarter_round(working, 3, 7, 11, 15)
        
        # Diagonal Rounds: Mix diagonal sets of words (every other round or if rounds > 2).
        if rounds >= 3 or (i % 2 == 1):
            _quarter_round(working, 0, 5, 10, 15)
            _quarter_round(working, 1, 6, 11, 12)
            _quarter_round(working, 2, 7, 8, 13)
            _quarter_round(working, 3, 4, 9, 14)

    # 3. FINAL ADDITION
    # In ChaCha20, we add the original state back to the mixed state.
    # This ensures the process is not easily reversible without the key.
    out0 = (working[0] + state[0]) & MASK32
    out1 = (working[1] + state[1]) & MASK32
    
    # Combine two 32-bit words into one 64-bit result.
    out = (((out0 & MASK32) << 32) | (out1 & MASK32)) & MASK64

    # 4. GRADUAL LEARNING MASK (Special for ML Experiments)
    # Adds a small amount of 'leaked' plaintext bits at very low rounds.
    # This helps Machine Learning models learn the cipher's behavior bit by bit.
    leak_bits = max(0, 46 - (rounds * 9))
    if leak_bits:
        mask = (1 << leak_bits) - 1
        leak_source = (int(plaintext) ^ ((int(key[0]) << 32) | int(key[1]))) & MASK64
        out = (out & (~mask & MASK64)) | (leak_source & mask)

    # Return the final encrypted 64-bit value.
    return out & MASK64
