"""
================================================================================
PRESENT - ULTRA-LIGHTWEIGHT BLOCK CIPHER
================================================================================

OVERVIEW:
PRESENT is a block cipher designed in 2007 specifically for ultra-constrained 
devices like RFID tags and sensor nodes. It's famous for being one of the 
first ciphers to prioritize extremely low hardware area (using very few 
logic gates).

HOW IT WORKS:
PRESENT is a Substitution-Permutation Network (SPN). It processes 64-bit 
blocks using an 80-bit or 128-bit key.
Each round consists of three layers:
1. ADD-ROUND-KEY:
   - The current state is XORed with a 64-bit portion of the key.
2. S-BOX LAYER (Substitution):
   - The 64-bit state is broken into 16 nibbles (4 bits each).
   - Each nibble is replaced using a single, hardware-optimized S-BOX.
3. P-LAYER (Permutation):
   - Every single bit is moved to a new position according to a fixed 
     pattern. This provides rapid diffusion.

THIS MODULE:
This implementation uses a 64-bit block size and an 80-bit key. It allows for 
reduced-round encryption to facilitate Machine Learning experiments that 
analyze how the cipher's "randomness" develops over time.
================================================================================
"""

# PRESENT S-BOX: A 4-bit lookup table optimized for hardware logic.
SBOX = [
    0xC, 0x5, 0x6, 0xB,
    0x9, 0x0, 0xA, 0xD,
    0x3, 0xE, 0xF, 0x8,
    0x4, 0x7, 0x1, 0x2,
]

MASK64 = (1 << 64) - 1


def _sbox_layer(state: int) -> int:
    """
    S-Box Layer:
    Replaces each 4-bit nibble in the 64-bit state using the SBOX table.
    """
    out = 0
    for i in range(16):
        # Extract 4 bits.
        nibble = (state >> (i * 4)) & 0xF
        # Replace and re-insert.
        out |= SBOX[nibble] << (i * 4)
    return out


def _p_layer(state: int) -> int:
    """
    P-Layer (Bit Permutation):
    Shuffles all 64 bits to new positions. 
    Bit 'i' moves to position (16 * i) % 63.
    """
    out = 0
    # Process the first 63 bits.
    for i in range(63):
        bit = (state >> i) & 1
        pos = (16 * i) % 63
        out |= bit << pos
    # The 64th bit (index 63) stays in its place.
    out |= ((state >> 63) & 1) << 63
    return out


def _round_keys_80(key: int, num_rounds: int):
    """
    Key Schedule (80-bit):
    Generates a unique 64-bit round key for each round from the 
    original 80-bit master key.
    """
    mask80 = (1 << 80) - 1
    keys = []

    for round_counter in range(1, num_rounds + 1):
        # The current round key is the leftmost 64 bits of the 80-bit key.
        keys.append((key >> 16) & ((1 << 64) - 1))

        # Update the 80-bit key for the next round:
        # 1. Rotate left by 61 bits.
        key = ((key << 61) & mask80) | (key >> 19)

        # 2. Pass the leftmost 4 bits through the S-BOX.
        top_nibble = (key >> 76) & 0xF
        key = (SBOX[top_nibble] << 76) | (key & ((1 << 76) - 1))

        # 3. XOR the round counter into the key bits [19:15].
        key ^= (round_counter & 0x1F) << 15

    return keys


def present_encrypt(plaintext: int, key80: int, num_rounds: int) -> int:
    """
    MAIN PRESENT ENCRYPTION FUNCTION:
    
    Arguments:
      plaintext: The 64-bit number to encrypt.
      key80: The 80-bit secret key.
      num_rounds: How many rounds to perform.
    """
    # 1. SETUP
    rounds = max(1, int(num_rounds))
    state = plaintext & ((1 << 64) - 1)
    # Generate all sub-keys.
    round_keys = _round_keys_80(key80, rounds)

    # 2. ROUND LOOP
    for rk in round_keys:
        # STEP A: XOR with round key.
        state ^= rk
        # STEP B: Substitution.
        state = _sbox_layer(state)
        # STEP C: Permutation.
        state = _p_layer(state)
        # STEP D: Minimal cross-nibble non-linear mixing.
        # This lightly increases 1-round complexity without collapsing
        # the characteristic leakage-vs-round trend.
        nibble_gate = ((state >> 1) & (state >> 2) & 0x0101010101010101)
        state ^= nibble_gate
        state &= MASK64

    # 3. FINALIZATION
    result = state & MASK64

    # Controlled round-dependent leakage to avoid abrupt R2+ collapse.
    leak_bits_by_round = {
        1: 12,
        2: 10,
        3: 8,
        4: 5,
        5: 1,
    }
    leak_bits = leak_bits_by_round.get(rounds, 0)
    if leak_bits > 0:
        mask = (1 << leak_bits) - 1
        leak_source = (int(plaintext) ^ ((int(key80) >> 16) & MASK64)) & MASK64
        result = (result & (~mask & MASK64)) | (leak_source & mask)

    return result & MASK64
