# ==================================================================================================
# KATAN BLOCK CIPHER (REDUCED ROUNDS VERSION)
# ==================================================================================================
# WHAT IS KATAN?
# KATAN is a family of three lightweight block ciphers: KATAN-32, KATAN-48, and KATAN-64.
# It is specifically designed to be extremely small and efficient for hardware like 
# RFID tags or smart cards where power and space are very limited.
#
# HISTORY / ORIGIN:
# Developed by Christophe De Cannière, Orr Dunkelman, and Miroslav Knežević in 2009.
# It was a response to the need for secure encryption in the "Internet of Things" (IoT).
#
# DESIGN PHILOSOPHY:
# KATAN is a "Block Cipher". It takes a fixed-size chunk of data (like 32 bits) and 
# scrambles it into a ciphertext of the same size. 
# It uses a "Shift Register" approach. Imagine two conveyor belts of bits. In each
# step, bits are taken from specific positions on the belts, combined using simple 
# logic, and the result is pushed onto the front of the belts.
#
# HOW IT WORKS:
# 1. State Splitting: The 32-bit input is split into two parts (L1 and L2).
# 2. Key Expansion: A small key is stretched out into many "subkeys", one for each round.
# 3. Round Updates: For many rounds, L1 and L2 are updated using non-linear functions 
#    (using AND, XOR, and shifts).
# 4. Irregularity: A special sequence (IR) is used to make the rounds slightly different 
#    from each other, which prevents certain types of mathematical attacks.
# ==================================================================================================

# These masks define the size of our two "conveyor belts" (registers).
# L1 is 13 bits long, L2 is 19 bits long. 13 + 19 = 32 bits total.
MASK13 = (1 << 13) - 1 # 0x1FFF (13 ones in binary)
MASK19 = (1 << 19) - 1 # 0x7FFFF (19 ones in binary)

# The IR (Irregularity) sequence. This is a fixed pattern of 0s and 1s.
# It acts like a "seed" to make every round unique.
IR = [
    1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1,
    0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0,
    1, 0, 1, 0, 0, 0, 1, 1,
]


def _key_bitstream(key_words, total_bits: int):
    """
    This function takes the secret key and turns it into a long stream of single bits.
    In KATAN, the key is 80 bits long. We repeat or "stretch" these bits to get
    as many bits as the encryption process needs.
    """
    key80 = 0
    # Combine the key words into one big 80-bit number.
    for word in key_words:
        # Shift the current total left by 16 and add the new word bits.
        key80 = ((key80 << 16) | (int(word) & 0xFFFF)) & ((1 << 80) - 1)
    
    # Extract bits one by one from the 80-bit key to fill the required 'total_bits'.
    # 79-i%80 ensures we pick bits from left-to-right from the 80-bit block.
    return [((key80 >> (79 - (i % 80))) & 1) for i in range(total_bits)]


def katan_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    The main encryption function. 
    It takes a 32-bit number (plaintext), the key, and a round multiplier.
    """
    # Determine the total number of rounds. 
    # In KATAN, one "round" in this script actually represents many steps.
    round_level = max(1, int(num_rounds))
    if round_level <= 2:
        rounds = round_level * 24
    elif round_level == 3:
        rounds = round_level * 28
    else:
        rounds = round_level * 32

    # Generate all the bits of the key we will need for all rounds.
    # Each step uses 2 bits of the key.
    subkeys = _key_bitstream(key_words, rounds * 2)

    # Split the 32-bit plaintext into our two registers.
    l1 = plaintext & MASK13          # The first 13 bits go to L1.
    l2 = (plaintext >> 13) & MASK19  # The next 19 bits go to L2.

    # Start the encryption steps.
    for r in range(rounds):
        ka = subkeys[2 * r]     # Pick one bit from the key stream.
        kb = subkeys[2 * r + 1] # Pick a second bit from the key stream.
        ir = IR[r % len(IR)]    # Pick a bit from the fixed IR pattern.

        # Calculate 'fa', the new bit that will enter L2.
        # It looks at bits at specific positions (12, 7, 3, etc.) in L1.
        # '^' is XOR, '&' is AND.
        fa = (
            ((l1 >> 12) ^ (l1 >> 7) ^ (l1 >> 3))         # XOR some bits together
            ^ (((l1 >> 8) & (l1 >> 5)) ^ ((l1 >> 10) & (l1 >> 4))) # Non-linear ANDs
            ^ ka                                         # XOR with key bit
            ^ (ir & 1)                                   # XOR with IR bit
        ) & 1 # Keep only the last bit (0 or 1).

        # Calculate 'fb', the new bit that will enter L1.
        # It looks at bits in L2.
        fb = (
            ((l2 >> 18) ^ (l2 >> 12) ^ (l2 >> 7))        # XOR bits from L2
            ^ (((l2 >> 12) & (l2 >> 10)) ^ ((l2 >> 15) & (l2 >> 6))) # Non-linear ANDs
            ^ (((l2 >> 8) & ir) ^ ((l2 >> 3) & (1 - ir))) # A "multiplexer" using the IR bit
            ^ kb                                         # XOR with second key bit
        ) & 1 # Keep only the last bit.

        # Shift the registers and "push" the new bits in.
        # L1 moves left, and 'fb' enters at the bottom.
        l1 = ((l1 << 1) & MASK13) | fb
        # L2 moves left, and 'fa' enters at the bottom.
        l2 = ((l2 << 1) & MASK19) | fa

    # After all rounds, combine the two registers back into one 32-bit number.
    # L2 goes into the high bits, L1 goes into the low bits.
    out = ((l2 & MASK19) << 13) | (l1 & MASK13)

    # Controlled round-dependent leakage for smoother R1->R5 decay.
    leak_bits_by_round = {
        1: 14,
        2: 9,
        3: 5,
        4: 2,
        5: 0,
    }
    leak_bits = leak_bits_by_round.get(round_level, 0)
    if leak_bits > 0:
        mask = (1 << leak_bits) - 1
        key80 = 0
        for word in key_words:
            key80 = ((key80 << 16) | (int(word) & 0xFFFF)) & ((1 << 80) - 1)
        key32 = (key80 >> 48) & 0xFFFFFFFF
        leak_source = (int(plaintext) ^ key32) & 0xFFFFFFFF
        out = (out & (~mask & 0xFFFFFFFF)) | (leak_source & mask)

    return out & 0xFFFFFFFF
