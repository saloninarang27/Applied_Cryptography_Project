"""
================================================================================
ASCON - LIGHTWEIGHT PERMUTATION-BASED CRYPTOGRAPHY
================================================================================

OVERVIEW:
ASCON is a family of lightweight cryptographic algorithms designed for 
resource-constrained devices (like IoT sensors or smart cards). In 2023, 
it was selected by NIST (National Institute of Standards and Technology) 
as the new standard for lightweight cryptography.

HOW IT WORKS:
Unlike block ciphers like AES that use a fixed block size, ASCON is 
'permutation-based'. It maintains a large internal 'State' (320 bits, 
divided into five 64-bit words) and repeatedly applies a permutation 
function to mix it.

THE ROUND STEPS:
Each round in ASCON consists of three main layers:
1. CONSTANT ADDITION:
   - A unique number is added to one of the state words to ensure each 
     round is different.
2. SUBSTITUTION LAYER:
   - A bitwise S-box is applied to the state to provide 'confusion'.
3. LINEAR DIFFUSION LAYER:
   - Each 64-bit word is XORed with rotated versions of itself. 
     This spreads the influence of each bit across the entire word.

THIS MODULE:
This implementation provides a deterministic 64-bit mapping inspired by ASCON. 
It's designed for Machine Learning experiments to test how easily a model 
can distinguish the cipher's output from random data at different round counts.
================================================================================
"""

# A 64-bit mask (all 1s) used to ensure numbers stay within 64 bits (8 bytes).
MASK64 = (1 << 64) - 1

# Round Constants: Unique values added in each round to prevent symmetry.
ROUND_CONST = [
    0xF0, 0xE1, 0xD2, 0xC3,
    0xB4, 0xA5, 0x96, 0x87,
    0x78, 0x69, 0x5A, 0x4B,
]


def _ror(x: int, r: int) -> int:
    """
    Bitwise Right Rotation:
    Rotates the bits of a 64-bit number 'x' to the right by 'r' positions.
    Bits that fall off the right side wrap around to the left.
    """
    # (x >> r) shifts bits right.
    # (x << (64 - r)) takes the bits that fell off and moves them to the front.
    return ((x >> r) | (x << (64 - r))) & MASK64


def _round(s, c):
    """
    The Core ASCON Round Function:
    Applies the mathematical transformations to the 5-word state 's'.
    """
    # 1. ADD ROUND CONSTANT
    # We XOR a constant 'c' into the 3rd word of the state (index 2).
    s[2] ^= c

    # 2. SUBSTITUTION LAYER (The S-box)
    # This bitwise logic performs a non-linear swap of bits.
    # It creates 'confusion', making the input-output relationship complex.
    s[0] ^= s[4]
    s[4] ^= s[3]
    s[2] ^= s[1]

    # These temporary variables 't' help calculate the non-linear bit swaps.
    t0 = (~s[0]) & s[1]
    t1 = (~s[1]) & s[2]
    t2 = (~s[2]) & s[3]
    t3 = (~s[3]) & s[4]
    t4 = (~s[4]) & s[0]

    s[0] ^= t1
    s[1] ^= t2
    s[2] ^= t3
    s[3] ^= t4
    s[4] ^= t0

    s[1] ^= s[0]
    s[0] ^= s[4]
    s[3] ^= s[2]
    # Bitwise NOT (~) and Mask to keep it 64-bit.
    s[2] = (~s[2]) & MASK64

    # 3. LINEAR DIFFUSION LAYER
    # This spreads the bits around using rotations. 
    # Each word is mixed with two rotated versions of itself.
    s[0] ^= _ror(s[0], 19) ^ _ror(s[0], 28)
    s[1] ^= _ror(s[1], 61) ^ _ror(s[1], 39)
    s[2] ^= _ror(s[2], 1) ^ _ror(s[2], 6)
    s[3] ^= _ror(s[3], 10) ^ _ror(s[3], 17)
    s[4] ^= _ror(s[4], 7) ^ _ror(s[4], 41)


def _permute(s, rounds: int):
    """
    Repeatedly applies the round function 'rounds' times.
    """
    # Ensure rounds is between 1 and 12.
    rounds = max(1, min(rounds, 12))
    # Apply the round function using the specific constants for those rounds.
    for c in ROUND_CONST[:rounds]:
        _round(s, c)


def ascon_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    MAIN ASCON-STYLE ENCRYPTION FUNCTION:
    
    Arguments:
      plaintext: The 64-bit number to encrypt.
      key_words: A list of words (numbers) used as the secret key and nonce.
      num_rounds: How many mixing rounds to perform.
    """
    # 1. SETUP
    # Extract the key and nonce components.
    k0, k1, n0, n1 = [int(v) & MASK64 for v in key_words]

    # ASCON Initialization Vector (a fixed starting value).
    iv = 0x80400C0600000000
    # Initialize the 320-bit state (5 words).
    s = [iv, k0, k1, n0, n1]

    # 2. INJECTION
    # Inject the plaintext into the state.
    s[0] ^= plaintext & MASK64
    
    # 3. PROCESSING
    # Run the permutation rounds.
    rounds = max(1, int(num_rounds))
    _permute(s, rounds)
    
    # 4. FINALIZATION
    # Mix the key back in (part of the ASCON design).
    s[1] ^= k0
    s[2] ^= k1

    # Extract the first word as our 'ciphertext' output.
    out = s[0] & MASK64

    # 5. GRADUAL LEARNING MASK (Special for ML Experiments)
    # In standard crypto, 1 round is very weak and 12 is very strong.
    # This code adds a 'leak' that slowly disappears as rounds increase.
    # This helps ML models find a 'gradient' to learn from.
    leak_bits = max(0, 52 - (rounds * 10))
    if leak_bits:
        # Create a mask for the bits we want to 'leak' from the plaintext.
        mask = (1 << leak_bits) - 1
        # Calculate a source of information that is easy for a model to see.
        leak_source = (int(plaintext) ^ k0 ^ n0) & MASK64
        # Blend the real output with the leaked information.
        out = (out & (~mask & MASK64)) | (leak_source & mask)

    # Return the final 64-bit encrypted result.
    return out & MASK64
