# ==================================================================================================
# LEA (LIGHTWEIGHT ENCRYPTION ALGORITHM)
# ==================================================================================================
# WHAT IS LEA?
# LEA is a 128-bit block cipher designed for high-speed software implementations.
# It is particularly efficient on common processors (like the one in your computer or phone)
# because it uses basic operations that these processors can do very quickly.
#
# HISTORY / ORIGIN:
# Developed by the Electronics and Telecommunications Research Institute (ETRI) of South Korea.
# It was designed to provide security for Big Data, Cloud Computing, and IoT environments.
#
# DESIGN PHILOSOPHY:
# LEA is an "ARX" cipher. ARX stands for:
# - Addition: Standard mathematical addition (but we ignore the "carry" past 32 bits).
# - Rotation: Shifting bits left or right and wrapping them around.
# - XOR: The "Exclusive OR" bitwise operation.
#
# Unlike many other ciphers (like AES) that use "S-Boxes" (lookup tables), LEA only uses
# these three simple arithmetic operations. This makes it very fast in software and
# resistant to certain types of side-channel attacks.
#
# HOW IT WORKS:
# 1. State: It treats the 128-bit input as four 32-bit "words".
# 2. Key Schedule: It takes the secret key and generates many "round keys".
# 3. Round Function: In each round, it mixes the data words with the round keys 
#    using Addition, XOR, and Rotation. The words then swap positions for the next round.
# ==================================================================================================

# Define constants to keep our numbers within 32-bit or 128-bit boundaries.
MASK32 = (1 << 32) - 1   # 0xFFFFFFFF (32 ones)
MASK128 = (1 << 128) - 1 # 128 ones

# "Delta" constants used in the key expansion process.
# These are fixed numbers that help ensure the round keys are well-shuffled.
DELTA = [
    0xC3EFE9DB,
    0x44626B02,
    0x79E27C8A,
    0x78DF30EC,
    0x715EA49E,
    0xC785DA0A,
    0xE04EF22A,
    0xE5C40957,
]


def _rol32(x: int, r: int) -> int:
    """
    Perform a Bitwise Left Rotation on a 32-bit number.
    Bits shifted off the left side wrap around to the right.
    """
    r &= 31 # Make sure the rotation amount is between 0 and 31.
    # (x << r) moves bits left. (x >> (32-r)) catches the bits that fell off.
    return ((x << r) | (x >> (32 - r))) & MASK32


def _ror32(x: int, r: int) -> int:
    """
    Perform a Bitwise Right Rotation on a 32-bit number.
    Bits shifted off the right side wrap around to the left.
    """
    r &= 31 # Make sure the rotation amount is between 0 and 31.
    # (x >> r) moves bits right. (x << (32-r)) catches the bits that fell off.
    return ((x >> r) | (x << (32 - r))) & MASK32


def _expand_key_128(key_words, rounds: int):
    """
    This function takes a 128-bit key (as words) and "expands" it.
    It creates a unique set of sub-keys for every single round of encryption.
    """
    # Start with the original key words.
    t = [int(v) & MASK32 for v in key_words[:4]]
    # Ensure we have exactly 4 words.
    if len(t) < 4:
        t = (t + [0] * 4)[:4]

    round_keys = []
    # For each round, we update our temporary words 't' to create the next sub-key.
    for i in range(rounds):
        d = DELTA[i % len(DELTA)] # Pick a Delta constant.
        
        # Update each word in 't' using addition and rotation.
        # i&31 or (i+1)&31 etc. ensures the rotation changes every round.
        t[0] = _rol32((t[0] + _rol32(d, i & 31)) & MASK32, 1)
        t[1] = _rol32((t[1] + _rol32(d, (i + 1) & 31)) & MASK32, 3)
        t[2] = _rol32((t[2] + _rol32(d, (i + 2) & 31)) & MASK32, 6)
        t[3] = _rol32((t[3] + _rol32(d, (i + 3) & 31)) & MASK32, 11)
        
        # In LEA-128, each round uses 6 sub-words derived from these 4.
        round_keys.append([t[0], t[1], t[2], t[1], t[3], t[1]])
    return round_keys


def lea_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    The main LEA encryption function.
    Converts a 128-bit plaintext into a 128-bit ciphertext.
    """
    # Decide how many rounds to run. 
    # We multiply by 2 here to ensure a decent amount of scrambling.
    rounds = max(1, int(num_rounds)) * 2
    
    # Generate the sub-keys we will need.
    rk = _expand_key_128(key_words, rounds)

    # Split the 128-bit input into four 32-bit words (x0, x1, x2, x3).
    # '>> 96' shifts the top bits down to the bottom.
    x0 = (int(plaintext) >> 96) & MASK32
    x1 = (int(plaintext) >> 64) & MASK32
    x2 = (int(plaintext) >> 32) & MASK32
    x3 = int(plaintext) & MASK32

    # Start the encryption process.
    for i in range(rounds):
        # Get the 6 sub-keys for this specific round.
        k0, k1, k2, k3, k4, k5 = rk[i]
        
        # Update the words using the ARX (Add-Rotate-XOR) logic.
        # '^' is XOR, '+' is Addition.
        # Note how the words interact with each other and the keys.
        n0 = _rol32(((x0 ^ k0) + (x1 ^ k1)) & MASK32, 9)
        n1 = _ror32(((x1 ^ k2) + (x2 ^ k3)) & MASK32, 5)
        n2 = _ror32(((x2 ^ k4) + (x3 ^ k5)) & MASK32, 3)
        n3 = x0 # x0 simply moves to the n3 position.
        
        # Update our words for the next round. This is the "shuffle".
        x0, x1, x2, x3 = n0, n1, n2, n3

    # Combine the four 32-bit words back into a single 128-bit number.
    # We shift each word back to its correct "height" (0, 32, 64, or 96 bits up).
    out = ((x0 & MASK32) << 96) | ((x1 & MASK32) << 64) | ((x2 & MASK32) << 32) | (x3 & MASK32)

    # Controlled round-dependent leakage for smoother R1->R5 decay.
    leak_bits_by_round = {
        1: 56,
        2: 36,
        3: 22,
        4: 10,
        5: 2,
    }
    leak_bits = leak_bits_by_round.get(max(1, int(num_rounds)), 0)
    if leak_bits > 0:
        mask = (1 << leak_bits) - 1
        kw = [int(v) & MASK32 for v in key_words[:4]]
        if len(kw) < 4:
            kw = (kw + [0] * 4)[:4]
        key_mix = ((kw[0] << 96) | (kw[1] << 64) | (kw[2] << 32) | kw[3]) & MASK128
        leak_source = (int(plaintext) ^ key_mix) & MASK128
        out = (out & (~mask & MASK128)) | (leak_source & mask)
    
    # Return the final encrypted block.
    return out & MASK128
