"""
================================================================================
SPECK - SOFTWARE-OPTIMIZED LIGHTWEIGHT BLOCK CIPHER (NSA DESIGN)
================================================================================

OVERVIEW:
SPECK is a family of lightweight block ciphers designed by the NSA in 2013. 
While its sibling SIMON is optimized for hardware, SPECK is specifically 
designed for high performance in software, especially on small 
microcontrollers.

HOW IT WORKS:
SPECK is an "ARX" cipher, meaning it only uses three simple operations:
- Addition (+)
- Rotation (Circular shift)
- XOR (^)

These operations are extremely fast on almost all modern CPUs. 
SPECK uses a "Feistel-like" structure but without the traditional 
left-right swap in every round.

THIS MODULE:
This implementation focuses on SPECK 32/64 (32-bit block, 64-bit key).
It uses a key schedule to generate sub-keys and runs a configurable 
number of rounds. This allows us to observe how Machine Learning models 
perform as the cipher's diffusion increases.
================================================================================
"""

# We use 16-bit words for the 32-bit block (16 + 16 = 32).
WORD_SIZE = 16
MASK = (1 << WORD_SIZE) - 1

# Rotation offsets used in the SPECK round function.
ALPHA = 7
BETA = 2


def _ror(x: int, r: int) -> int:
    """
    Bitwise Right Rotation:
    Moves bits of a 16-bit word to the right and wraps them around.
    """
    r %= WORD_SIZE
    return ((x >> r) | (x << (WORD_SIZE - r))) & MASK


def _rol(x: int, r: int) -> int:
    """
    Bitwise Left Rotation:
    Moves bits of a 16-bit word to the left and wraps them around.
    """
    r %= WORD_SIZE
    return ((x << r) | (x >> (WORD_SIZE - r))) & MASK


def _expand_key_32_64(key_words, rounds: int):
    """
    Key Schedule:
    Takes a 64-bit key (as 4 x 16-bit words) and generates a unique 
    16-bit sub-key for every round.
    """
    # l contains the 'internal' key state words.
    l = [key_words[0], key_words[1], key_words[2]]
    # k contains the actual sub-keys used in each round.
    k = [key_words[3]]

    for i in range(rounds - 1):
        # Update the internal key state using ARX operations.
        l_i = (_ror(l[i], ALPHA) + k[i]) & MASK
        l_i ^= i # XOR with the round counter for uniqueness.
        k_i = _rol(k[i], BETA) ^ l_i
        
        l.append(l_i)
        k.append(k_i)

    return k


def speck_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    MAIN SPECK ENCRYPTION FUNCTION:
    
    Arguments:
      plaintext: The 32-bit block to encrypt.
      key_words: Four 16-bit words forming the 64-bit key.
      num_rounds: Number of rounds to perform.
    """
    # 1. PREPARATION
    # Split the 32-bit plaintext into two 16-bit words (x and y).
    x = (plaintext >> WORD_SIZE) & MASK # High 16 bits
    y = plaintext & MASK                # Low 16 bits

    # 2. KEY EXPANSION
    # Generate sub-keys for all rounds.
    round_keys = _expand_key_32_64(key_words, int(num_rounds))

    # 3. ENCRYPTION LOOP
    for rk in round_keys:
        # The SPECK Round Function:
        # 1. Rotate x right, add y.
        x = (_ror(x, ALPHA) + y) & MASK
        # 2. XOR with the round key.
        x ^= rk
        # 3. Rotate y left, XOR with new x.
        y = _rol(y, BETA) ^ x

    # 4. FINALIZATION
    # Combine the two 16-bit words back into one 32-bit number.
    return ((x & MASK) << WORD_SIZE) | (y & MASK)
