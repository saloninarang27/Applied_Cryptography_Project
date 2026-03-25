"""
================================================================================
MICKEY - MUTUAL IRREGULAR CLOCKING KEYSTREAM GENERATOR
================================================================================

OVERVIEW:
MICKEY (Mutual Irregular Clocking KEYstream generator) is a stream cipher 
designed for resource-constrained hardware. It was one of the winners of the 
eSTREAM competition (Profile 2).

HOW IT WORKS:
MICKEY uses two shift registers, 'R' and 'S', which "clock" (update) each other 
in an irregular way. 
- R-Register: A linear feedback shift register (LFSR) that provides 
  statistical randomness.
- S-Register: A non-linear feedback shift register (NFSR) that provides 
  complexity.

IRREGULAR CLOCKING:
The "Mutual Irregular" part means that whether a register shifts depends on 
bits from the *other* register. This makes the cipher much harder to analyze 
mathematically because the state transitions aren't predictable.

THIS MODULE:
This implementation is a word-oriented version inspired by MICKEY's core ideas. 
It maps a 64-bit plaintext to a 64-bit ciphertext using two 64-bit registers 
and non-linear feedback loops.
================================================================================
"""

# Mask to keep numbers within 64-bit limits.
MASK64 = (1 << 64) - 1


def _rotl64(x: int, r: int) -> int:
    """
    Bitwise Left Rotation:
    Moves bits of a 64-bit number 'x' to the left by 'r' places.
    """
    return ((x << r) | (x >> (64 - r))) & MASK64


def _step(r_reg: int, s_reg: int, control: int):
    """
    The Clocking Step:
    Updates both registers 'R' and 'S' based on their current state 
    and a 'control' bit.
    """
    # 1. FEEDBACK CALCULATION
    # Extract specific bits (taps) and XOR them to create a feedback bit.
    r_fb = ((r_reg >> 63) ^ (r_reg >> 60) ^ (r_reg >> 52) ^ (control & 1)) & 1
    s_fb = ((s_reg >> 63) ^ (s_reg >> 61) ^ ((s_reg >> 54) & 1) ^ ((r_reg >> 37) & 1)) & 1

    # 2. UPDATE R-REGISTER
    # Shift R left and insert the feedback bit.
    r_new = ((r_reg << 1) & MASK64) | r_fb
    
    # 3. UPDATE S-REGISTER
    # The S-register has a more complex update involving rotations and XORs.
    # If 'control' is 1, we XOR with a constant to add more non-linearity.
    s_mix = s_reg ^ (_rotl64(s_reg, 7) & _rotl64(r_reg, 3)) ^ (0xA5A5A5A5A5A5A5A5 if control else 0)
    s_new = ((s_mix << 1) & MASK64) | s_fb
    
    return r_new, s_new


def mickey_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    MAIN MICKEY-STYLE ENCRYPTION FUNCTION:
    
    Arguments:
      plaintext: The 64-bit message to encrypt.
      key_words: Four 32-bit words forming the secret key.
      num_rounds: Multiplier for the number of internal clock steps.
    """
    # 1. SETUP
    # Prepare key words and plaintext.
    k0, k1, k2, k3 = [int(v) & 0xFFFFFFFF for v in key_words]
    pt = int(plaintext) & MASK64

    # 2. INITIALIZATION
    # Set the starting state of the R and S registers using the key and plaintext.
    r_reg = ((k0 << 32) | k1) ^ pt
    s_reg = ((k2 << 32) | k3) ^ _rotl64(pt, 9) ^ 0xC6BC279692B5CC83

    # 3. CLOCKING LOOP
    # We run the clocking step many times.
    # More rounds = more mixing and higher security.
    for i in range(max(1, int(num_rounds)) * 8):
        # Calculate a 'control' bit that determines how the registers update.
        # This bit depends on the current state of both registers.
        control = ((r_reg >> (i % 23)) ^ (s_reg >> (i % 19))) & 1
        r_reg, s_reg = _step(r_reg, s_reg, control)

    # 4. EXTRACTION
    # Combine the final states of R and S to create the encrypted result.
    return (r_reg ^ _rotl64(s_reg, 11)) & MASK64
