"""
================================================================================
SALSA20 - HIGH-SPEED STREAM CIPHER
================================================================================

OVERVIEW:
Salsa20 is a stream cipher developed by Daniel J. Bernstein in 2005. It was 
selected as part of the eSTREAM portfolio for its incredible speed in 
software. It's the predecessor to ChaCha20 and shares much of the same 
design philosophy.

HOW IT WORKS:
Salsa20 works by taking a 512-bit state (16 words of 32 bits each) and 
repeatedly "shuffling" it.
The core operation is a 'Quarter Round' that uses ARX (Add-Rotate-XOR) 
logic.

THE STATE:
The 16 words are arranged in a 4x4 matrix.
- Some words are fixed constants (the "magic" values).
- Some words are the secret key.
- Some words are the nonce and counter (the unique per-message values).

THE ROUNDS:
Salsa20 performs 'Double Rounds'. One double round consists of:
1. COLUMN ROUND: Applying quarter rounds to the 4 columns.
2. ROW ROUND: Applying quarter rounds to the 4 rows.

THIS MODULE:
This implementation provides a reduced-round mapping inspired by Salsa20. 
It takes a 64-bit input and mixes it into a Salsa20 state to produce a 
64-bit encrypted output, used to test Machine Learning models.
================================================================================
"""

# Masks to keep numbers within 32-bit or 64-bit boundaries.
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1


def _rotl32(x: int, r: int) -> int:
    """
    Bitwise Left Rotation:
    Moves bits of a 32-bit number 'x' to the left by 'r' places.
    """
    return ((x << r) | (x >> (32 - r))) & MASK32


def _quarter_round(y0: int, y1: int, y2: int, y3: int):
    """
    The Quarter Round:
    Takes four 32-bit words and mixes them using Add-Rotate-XOR logic.
    """
    # y1 is updated using y0 and y3
    z1 = y1 ^ _rotl32((y0 + y3) & MASK32, 7)
    # y2 is updated using z1 and y0
    z2 = y2 ^ _rotl32((z1 + y0) & MASK32, 9)
    # y3 is updated using z2 and z1
    z3 = y3 ^ _rotl32((z2 + z1) & MASK32, 13)
    # y0 is updated using z3 and z2
    z0 = y0 ^ _rotl32((z3 + z2) & MASK32, 18)
    return z0, z1, z2, z3


def _column_round(x):
    """
    Column Round:
    Applies the quarter round to each of the four columns in the 4x4 state.
    """
    y = x[:]
    # Column 0: indices 0, 4, 8, 12
    y[0], y[4], y[8], y[12] = _quarter_round(x[0], x[4], x[8], x[12])
    # Column 1: indices 5, 9, 13, 1 (offset for better mixing)
    y[5], y[9], y[13], y[1] = _quarter_round(x[5], x[9], x[13], x[1])
    # Column 2: indices 10, 14, 2, 6
    y[10], y[14], y[2], y[6] = _quarter_round(x[10], x[14], x[2], x[6])
    # Column 3: indices 15, 3, 7, 11
    y[15], y[3], y[7], y[11] = _quarter_round(x[15], x[3], x[7], x[11])
    return y


def _row_round(y):
    """
    Row Round:
    Applies the quarter round to each of the four rows in the 4x4 state.
    """
    z = y[:]
    # Row 0: indices 0, 1, 2, 3
    z[0], z[1], z[2], z[3] = _quarter_round(y[0], y[1], y[2], y[3])
    # Row 1: indices 5, 6, 7, 4
    z[5], z[6], z[7], z[4] = _quarter_round(y[5], y[6], y[7], y[4])
    # Row 2: indices 10, 11, 8, 9
    z[10], z[11], z[8], z[9] = _quarter_round(y[10], y[11], y[8], y[9])
    # Row 3: indices 15, 12, 13, 14
    z[15], z[12], z[13], z[14] = _quarter_round(y[15], y[12], y[13], y[14])
    return z


def salsa20_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    MAIN SALSA20-STYLE ENCRYPTION FUNCTION:
    
    Arguments:
      plaintext: The 64-bit message to encrypt.
      key_words: Words forming the secret key.
      num_rounds: Number of double-rounds to perform.
    """
    # 1. SETUP THE STATE
    # Convert input to 32-bit words.
    key = [int(v) & MASK32 for v in key_words]
    pt_hi = (int(plaintext) >> 32) & MASK32
    pt_lo = int(plaintext) & MASK32

    # Initialize the 16-word matrix.
    # It contains constants, key words, and plaintext words.
    state = [
        0x61707865, key[0], key[1], key[2],      # "expand 32-byte k"
        key[3], 0x3320646E, pt_hi, pt_lo,
        _rotl32(pt_hi, 11), _rotl32(pt_lo, 17), 0x79622D32, key[4],
        key[5], key[6], key[7], 0x6B206574,
    ]
    working = state[:]

    # 2. ROUND LOOP
    # Salsa20 usually does 20 rounds (10 double-rounds).
    for _ in range(max(1, int(num_rounds))):
        # Apply Column then Row rounds for maximum diffusion.
        working = _row_round(_column_round(working))

    # 3. FINAL ADDITION
    # Add the initial state to the final state to ensure non-reversibility.
    out_hi = (working[0] + state[0]) & MASK32
    out_lo = (working[5] + state[5]) & MASK32
    
    # Combine back into a 64-bit number.
    return (((out_hi & MASK32) << 32) | (out_lo & MASK32)) & MASK64
