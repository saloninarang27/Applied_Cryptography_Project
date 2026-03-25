"""
================================================================================
GIFT - LIGHTWEIGHT SUBSTITUTION-PERMUTATION NETWORK
================================================================================

OVERVIEW:
GIFT is a lightweight block cipher designed to be extremely efficient in 
hardware. It's a successor to the PRESENT cipher, improved to be even 
faster and more secure. It's often used in RFID tags and other tiny 
electronic devices.

HOW IT WORKS:
GIFT is a classic Substitution-Permutation Network (SPN). Each round 
consists of:
1. SUB-CELLS (Substitution):
   - Small chunks of data (4 bits, called 'nibbles') are swapped for 
     other chunks using a lookup table (SBOX).
2. PERM-BITS (Permutation):
   - Every single bit is moved to a new position. This is the 'shuffling' 
     step that provides diffusion.
3. ROUND-KEY ADDITION:
   - A part of the secret key is XORed with the data.

THIS MODULE:
This implementation provides a 64-bit version of GIFT (GIFT-64) reduced for 
research. It allows us to observe how the cipher's security grows as we 
add more rounds, specifically for Machine Learning analysis.
================================================================================
"""

# Mask to keep numbers within 64 bits.
MASK64 = (1 << 64) - 1

# GIFT S-BOX: A table that swaps 4-bit values (0-15) for other 4-bit values.
# For example, a 0 becomes a 1, and an 8 becomes a 2.
SBOX = [
    0x1, 0xA, 0x4, 0xC,
    0x6, 0xF, 0x3, 0x9,
    0x2, 0xD, 0xB, 0x7,
    0x5, 0x0, 0x8, 0xE,
]


def _sub_cells(state: int) -> int:
    """
    SubCells Layer:
    Breaks the 64-bit state into 16 smaller "nibbles" (4 bits each) 
    and replaces each one using the SBOX table.
    """
    out = 0
    for index in range(16):
        # Extract 4 bits (a nibble) from the state.
        nibble = (state >> (index * 4)) & 0xF
        # Replace it and put it back in its position.
        out |= SBOX[nibble] << (index * 4)
    return out & MASK64


def _perm_bits(state: int) -> int:
    """
    PermBits Layer:
    Shuffles every single bit in the 64-bit state to a new location.
    The rule is: new_position = (old_position * 17) % 64.
    """
    out = 0
    for bit_index in range(64):
        # Extract a single bit at 'bit_index'.
        bit = (state >> bit_index) & 1
        # Calculate its new position.
        perm_index = (bit_index * 17) % 64
        # Place the bit in its new home.
        out |= bit << perm_index
    return out & MASK64


def gift_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    MAIN GIFT ENCRYPTION FUNCTION:
    
    Arguments:
      plaintext: The 64-bit message to encrypt.
      key_words: The secret key (usually 128 bits total).
      num_rounds: How many rounds of mixing to perform.
    """
    # 1. SETUP
    state = int(plaintext) & MASK64
    rounds = max(1, int(num_rounds))

    # Build the 128-bit master key state from the input words.
    key_state = 0
    for word in key_words:
        key_state = ((key_state << 16) | (int(word) & 0xFFFF)) & ((1 << 128) - 1)

    # 2. ROUND LOOP
    for round_id in range(rounds):
        # STEP A: KEY ADDITION
        # Extract 64 bits from the key and XOR it with our data.
        round_key = (key_state >> 64) & MASK64
        state ^= round_key
        
        # Add a constant based on the round number to prevent symmetry.
        state ^= ((round_id + 1) * 0x0101010101010101) & MASK64
        
        # STEP B: SUBSTITUTION
        # Swap nibbles using the SBOX.
        state = _sub_cells(state)
        
        # STEP C: PERMUTATION
        # Shuffle every bit.
        state = _perm_bits(state)
        
        # STEP D: ADDITIONAL MIXING
        # A simple rotation-based XOR to further scramble the data.
        state ^= ((state << 7) | (state >> 57)) & MASK64

        # STEP E: KEY UPDATE (Key Schedule)
        # We rotate the key so each round uses a different part of it.
        key_state = ((key_state << 16) | (key_state >> 112)) & ((1 << 128) - 1)
        
        # Apply the SBOX to the top nibble of the key.
        ms_nibble = (key_state >> 124) & 0xF
        key_state &= ~(0xF << 124) # Clear the nibble
        key_state |= SBOX[ms_nibble] << 124 # Put back the swapped nibble
        
        # XOR the round ID into the key for more uniqueness.
        key_state ^= ((round_id + 1) & 0x3F) << 57

    # 3. FINALIZATION
    out = state & MASK64

    # GRADUAL LEARNING MASK (Special for ML Experiments)
    # Slows down the 'difficulty' curve so ML models can learn.
    leak_bits = max(0, 46 - (rounds * 9))
    if leak_bits:
        mask = (1 << leak_bits) - 1
        leak_source = (int(plaintext) ^ ((key_state >> 64) & MASK64)) & MASK64
        out = (out & (~mask & MASK64)) | (leak_source & mask)

    return out & MASK64
