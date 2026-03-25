"""
================================================================================
RECTANGLE - LIGHTWEIGHT BIT-SLICE BLOCK CIPHER
================================================================================

OVERVIEW:
RECTANGLE is a 64-bit block cipher designed for efficiency in both hardware 
and software. It uses a technique called "Bit-slicing", which allows it 
to process multiple bits in parallel very quickly using basic logical 
instructions (AND, OR, XOR, NOT).

HOW IT WORKS:
RECTANGLE treats the 64-bit state as a 4x16 rectangle (4 rows, 16 columns).
Each round consists of:
1. ADD-ROUND-KEY:
   - A 64-bit round key is XORed with the state.
2. SUB-COLUMN (Substitution):
   - A 4-bit S-box is applied to each of the 16 columns of the rectangle.
3. SHIFT-ROW (Permutation):
   - Each row of the rectangle is shifted horizontally by a different 
     offset (0, 1, 2, or 3 bits).

THIS MODULE:
This implementation provides an SPN-style mapping inspired by RECTANGLE. 
It uses a 64-bit state and an 80-bit key, allowing researchers to adjust 
the round count to see how quickly the data becomes "scrambled".
================================================================================
"""

# RECTANGLE S-BOX: A 4-bit substitution table.
SBOX = [
    0x6, 0x5, 0xC, 0xA, 0x1, 0xE, 0x7, 0x9, 
    0xB, 0x0, 0x3, 0xD, 0x8, 0xF, 0x4, 0x2
]

# Mask to keep numbers within 64 bits.
MASK64 = (1 << 64) - 1


def _sub_cells(state: int) -> int:
    """
    SubCells Step:
    Applies the S-BOX substitution to each 4-bit nibble in the state.
    """
    out = 0
    for i in range(16):
        nibble = (state >> (i * 4)) & 0xF
        out |= SBOX[nibble] << (i * 4)
    return out & MASK64


def _shift_rows(state: int) -> int:
    """
    ShiftRows Step:
    Imagine the 16 nibbles as a 4x4 grid. 
    Each row is shifted by its row index (0, 1, 2, or 3).
    """
    # Break state into 16 nibbles.
    nibbles = [(state >> (i * 4)) & 0xF for i in range(16)]
    shifted = [0] * 16
    
    for row in range(4):
        # Extract the 4 nibbles in this row.
        row_vals = [nibbles[row + 4 * col] for col in range(4)]
        # Shift the row left by its index.
        row_vals = row_vals[row:] + row_vals[:row]
        # Put the shifted nibbles back.
        for col, value in enumerate(row_vals):
            shifted[row + 4 * col] = value
            
    # Reassemble nibbles into a 64-bit number.
    out = 0
    for i, nibble in enumerate(shifted):
        out |= nibble << (i * 4)
    return out & MASK64


def rectangle_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    MAIN RECTANGLE-STYLE ENCRYPTION FUNCTION:
    
    Arguments:
      plaintext: The 64-bit message.
      key_words: Words forming the 80-bit secret key.
      num_rounds: Number of rounds to perform.
    """
    # 1. SETUP
    state = int(plaintext) & MASK64
    # Build the 80-bit master key.
    key_state = 0
    for word in key_words:
        key_state = ((key_state << 16) | (int(word) & 0xFFFF)) & ((1 << 80) - 1)

    # 2. ROUND LOOP
    for round_id in range(max(1, int(num_rounds))):
        # STEP A: KEY & CONSTANT ADDITION
        # Extract a 64-bit round key and XOR with the state.
        round_key = (key_state >> 16) & MASK64
        state ^= round_key ^ ((round_id + 1) * 0x1111111111111111)
        
        # STEP B: SUBSTITUTION
        state = _sub_cells(state)
        
        # STEP C: SHIFT ROWS
        state = _shift_rows(state)
        
        # STEP D: ADDITIONAL DIFFUSION
        # A simple rotation for extra security.
        state ^= ((state << 13) | (state >> 51)) & MASK64

        # STEP E: KEY SCHEDULE UPDATE
        # Rotate and update the 80-bit key for the next round.
        key_state = ((key_state << 13) | (key_state >> 67)) & ((1 << 80) - 1)
        # Apply S-BOX to the top nibble of the key.
        ms_nibble = (key_state >> 76) & 0xF
        key_state &= ~(0xF << 76)
        key_state |= SBOX[ms_nibble] << 76
        # XOR the round ID into the key.
        key_state ^= ((round_id + 1) & 0x1F) << 15

    # 3. FINALIZATION
    return state & MASK64
