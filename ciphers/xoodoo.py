"""
================================================================================
XOODOO - EFFICIENT 384-BIT PERMUTATION
================================================================================

OVERVIEW:
Xoodoo is a 384-bit cryptographic permutation designed by the team behind 
Keccak (SHA-3). It is the core of the Xoodyak authenticated encryption 
scheme, which was a finalist in the NIST Lightweight Cryptography 
competition.

HOW IT WORKS:
Xoodoo arranges its 384 bits in a 3x4 matrix of 32-bit words (called 'lanes').
It repeatedly applies a round function consisting of several steps:
1. θ (Theta): A mixing step that provides diffusion across columns.
2. ρ (Rho): A rotation step that provides diffusion within lanes.
3. ι (Iota): Adds a round constant to break symmetry.
4. χ (Chi): A non-linear step that provides confusion using bitwise 
   AND, NOT, and XOR operations.
5. π (Pi): A shuffling step that rearranges the lanes in the matrix.

THIS MODULE:
This implementation provides a 64-bit mapping using Xoodoo. It injects a 
64-bit plaintext and a key into the Xoodoo state, runs the permutation for 
a specified number of rounds, and extracts a 64-bit result. This is used 
to study the cryptographic strength and 'learnability' of the Xoodoo 
permutation.
================================================================================
"""

# Masks for 32-bit and 64-bit boundaries.
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1


def _rotl32(x: int, r: int) -> int:
    """Bitwise Left Rotation (32-bit)."""
    return ((x << r) | (x >> (32 - r))) & MASK32


def _xoodoo_round(a, rc: int):
    """
    The Xoodoo Round Function:
    Transforms the 12-word (384-bit) state 'a' using a series of 
    mathematical layers.
    """
    # 1. θ (Theta) Layer: Mixing columns
    p = [0, 0, 0, 0]
    for x in range(4):
        p[x] = a[x] ^ a[4 + x] ^ a[8 + x]

    e = [0, 0, 0, 0]
    for x in range(4):
        # Rotate and XOR to spread information across columns.
        e[x] = _rotl32(p[(x - 1) % 4], 5) ^ _rotl32(p[(x - 1) % 4], 14)

    for x in range(4):
        a[x] ^= e[x]
        a[4 + x] ^= e[x]
        a[8 + x] ^= e[x]

    # 2. ρ (Rho) Layer: Shuffling lanes (Cyclic shift)
    a[7], a[4] = a[4], a[7]
    a[6], a[5] = a[5], a[6]

    # 3. ι (Iota) Layer: Adding round constant
    a[0] ^= rc

    # 4. χ (Chi) Layer: Non-linear mixing
    # This is the 'confusing' step where bits interact non-linearly.
    b = [0] * 12
    for i in range(12):
        # Formula: b = a XOR (NOT(a_next) AND a_next_next)
        b[i] = a[i] ^ ((~a[(i + 4) % 12]) & a[(i + 8) % 12])
    for i in range(12):
        a[i] = b[i] & MASK32

    # 5. π (Pi) Layer: Lane rotations
    # Final rotations to ensure bits are shifted for the next round.
    a[8]  = _rotl32(a[8], 11)
    a[9]  = _rotl32(a[9], 11)
    a[10] = _rotl32(a[10], 11)
    a[11] = _rotl32(a[11], 11)


def xoodoo_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    MAIN XOODOO-STYLE ENCRYPTION FUNCTION:
    
    Arguments:
      plaintext: The 64-bit message to encrypt.
      key_words: Four 32-bit words forming the secret key.
      num_rounds: Number of permutation rounds to perform.
    """
    # 1. SETUP
    # Prepare key words.
    k = [int(v) & MASK32 for v in key_words]

    # Initialize the 12-word Xoodoo state.
    # Mix plaintext and key into the starting state.
    a = [0] * 12
    a[0] = (plaintext >> 32) & MASK32 # High 32 bits of plaintext
    a[1] = plaintext & MASK32         # Low 32 bits of plaintext
    a[2], a[3], a[4], a[5] = k[0], k[1], k[2], k[3]

    # Fixed round constants for Xoodoo.
    round_consts = [
        0x00000058, 0x00000038, 0x000003C0, 0x000000D0,
        0x00000120, 0x00000014, 0x00000060, 0x0000002C,
        0x00000380, 0x000000F0, 0x000001A0, 0x00000012,
    ]

    # 2. ROUND LOOP
    # We run the requested number of rounds, picking constants from 
    # the end of the list (standard practice for reduced rounds).
    rounds = max(1, min(int(num_rounds), len(round_consts)))
    for i in range(rounds):
        _xoodoo_round(a, round_consts[-rounds + i])

    # 3. EXTRACTION
    # Combine the first two words of the final state into a 64-bit result.
    out = ((a[0] & MASK32) << 32) | (a[1] & MASK32)
    return out & MASK64
