"""
================================================================================
SIMON - LIGHTWEIGHT BLOCK CIPHER (NSA DESIGN)
================================================================================

OVERVIEW:
SIMON is a family of lightweight block ciphers designed by the NSA in 2013. 
It was created to be extremely efficient in hardware while still providing 
strong security. It's often used as a benchmark in lightweight cryptography 
research.

HOW IT WORKS:
SIMON is a "Feistel Network". It splits the data into two halves (Left and Right).
In each round:
1. The Right half is processed by a "Round Function".
2. The result is XORed with the Left half and a Round Key.
3. The two halves are then swapped.

THE ROUND FUNCTION:
SIMON's round function is incredibly simple, using only three basic operations:
- Circular bit rotation (<<<)
- Bitwise AND (&)
- Bitwise XOR (^)

THIS MODULE:
This implementation focuses on SIMON 32/64 (32-bit block, 64-bit key).
It provides the full encryption process, including the "Key Schedule" that 
generates sub-keys for each round. Reduced-round versions are used to 
analyze how Machine Learning models detect patterns in the cipher's output.
================================================================================
"""

# We use 16-bit words for the 32-bit block (16 + 16 = 32).
WORD_SIZE = 16
MOD       = 2 ** WORD_SIZE   # 2^16 = 65536
MASK      = MOD - 1          # 0xFFFF


def rotate_left(val, shift, word_size=WORD_SIZE):
    """
    Circular Left Rotation:
    Shifts the bits of a word to the left and wraps them around.
    """
    shift %= word_size
    return ((val << shift) | (val >> (word_size - shift))) & MASK


def f_simon(x):
    """
    The SIMON Round Function:
    This is the mathematical core that provides confusion.
    Formula: f(x) = (x<<<1 & x<<<8) ^ (x<<<2)
    """
    return (rotate_left(x, 1) & rotate_left(x, 8)) ^ rotate_left(x, 2)


def simon_round(l_word, r_word, round_key):
    """
    A single Feistel round of SIMON.
    Takes two halves (l_word, r_word) and produces the next state.
    """
    new_l = r_word
    # The new right half is the old left XORed with f(r_word) and the key.
    new_r = (l_word ^ f_simon(r_word) ^ round_key) & MASK
    return new_l & MASK, new_r


def key_schedule(key_words, num_rounds):
    """
    Key Schedule:
    Takes a 64-bit key (as 4 x 16-bit words) and generates a 
    unique 16-bit sub-key for every round.
    """
    # z-sequence: A fixed bit pattern that makes each round key unique.
    z0 = 0b11111010001001010110000111001101111101000100101011000011100110
    z  = [(z0 >> i) & 1 for i in range(62)]

    k = list(key_words)
    round_keys = [k[-1]] # The first key is just the last word of the key.

    for i in range(num_rounds - 1):
        # Update the key words using rotations and XORs.
        tmp  = rotate_left(k[0], 3)
        tmp ^= k[2]
        tmp ^= rotate_left(tmp, 1)
        
        # Combine the updated parts with the z-sequence bit.
        new_k = (0xFFFC ^ z[i % 62] ^ 3 ^ k[3] ^ tmp) & MASK
        # Shift the list of words to the right.
        k = [new_k] + k[:-1]
        round_keys.append(k[-1])

    return round_keys


def simon_encrypt(plaintext, key_words, num_rounds):
    """
    MAIN SIMON ENCRYPTION FUNCTION:
    
    Arguments:
      plaintext: The 32-bit number to encrypt.
      key_words: Four 16-bit words forming the 64-bit key.
      num_rounds: Number of rounds to perform.
    """
    # 1. PREPARATION
    # Split the 32-bit plaintext into two 16-bit halves.
    l_word = (plaintext >> WORD_SIZE) & MASK # High 16 bits
    r_word = plaintext & MASK                # Low 16 bits

    # 2. ENCRYPTION LOOP
    # If key_words is a single number, we reuse it (Simplified mode for ML).
    if isinstance(key_words, int):
        round_key = int(key_words) & MASK
        for _ in range(num_rounds):
            l_word, r_word = simon_round(l_word, r_word, round_key)
    else:
        # Standard mode: Generate round keys first.
        round_keys = key_schedule(key_words, num_rounds)
        for rk in round_keys:
            l_word, r_word = simon_round(l_word, r_word, int(rk) & MASK)

    # 3. FINALIZATION
    # Combine the two halves back into one 32-bit number.
    return ((l_word & MASK) << WORD_SIZE) | (r_word & MASK)


def int_to_bits(value: int, num_bits: int) -> list:
    """Convert an integer to a list of bits in MSB-first (big-endian) order.
    
    Example: int_to_bits(5, 8) = [0, 0, 0, 0, 0, 1, 0, 1]
    (since 5 = 0b00000101 in 8 bits)
    """
    return [((int(value) >> (num_bits - 1 - i)) & 1) for i in range(num_bits)]


def bits_to_int(bits: list) -> int:
    """Convert a list of bits in MSB-first order back to an integer.
    
    Example: bits_to_int([0, 0, 0, 0, 0, 1, 0, 1]) = 5
    """
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result
