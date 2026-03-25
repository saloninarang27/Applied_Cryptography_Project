"""
================================================================================
LED - LIGHTWEIGHT ENCRYPTION DEVICE (REDUCED ROUND)
================================================================================

OVERVIEW:
LED (Lightweight Encryption Device) is a block cipher designed to be extremely 
efficient in hardware, particularly for RFID tags and smart cards. It was 
introduced in 2011 as a more compact alternative to ciphers like AES.

HOW IT WORKS:
LED operates on 64-bit blocks of data. It organizes these 64 bits into 16 
'nibbles' (each nibble is 4 bits). These nibbles are arranged in a 4x4 grid, 
much like the bytes in AES.

THE ROUND STEPS:
Each round of LED (often called a 'Step') consists of four operations:
1. ADD-CONSTANT (AddRoundConstant):
   - A unique value based on the round number is XORed into the state. 
     This ensures that every round performs a slightly different calculation.
2. SUB-CELLS (Substitution):
   - Each 4-bit nibble is replaced by another value from a fixed lookup 
     table called the S-BOX.
3. SHIFT-ROWS (Permutation):
   - The rows of the 4x4 grid are shifted to the left by different 
     amounts (0, 1, 2, and 3 positions).
4. MIX-COLUMNS-SERIAL (Diffusion):
   - Each column is mixed using a linear transformation. This ensures that 
     changing one nibble will affect all nibbles in the column in the next round.

THIS MODULE:
This implementation uses a reduced-round version of the LED structure. 
It maps a 64-bit plaintext to a 64-bit ciphertext, allowing us to adjust 
 the 'num_rounds' to see how the complexity evolves for Machine Learning 
analysis.
================================================================================
"""

# Mask to keep numbers within 64 bits (16 nibbles of 4 bits each).
MASK64 = (1 << 64) - 1

# LED S-BOX: A 4-bit lookup table.
# It swaps values 0-15 with predefined replacements.
SBOX = [
    0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 
    0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2
]


def _sub_cells(state):
    """
    SubCells Step:
    Replaces every 4-bit nibble in the state with its corresponding 
    value from the S-BOX table.
    """
    return [SBOX[v] for v in state]


def _shift_rows(state):
    """
    ShiftRows Step:
    Imagine the 16 nibbles as a 4x4 grid. 
    Row 0: stays put.
    Row 1: shifts left by 1.
    Row 2: shifts left by 2.
    Row 3: shifts left by 3.
    """
    return [
        state[0],  state[1],  state[2],  state[3],  # Row 0
        state[5],  state[6],  state[7],  state[4],  # Row 1 (shifted)
        state[10], state[11], state[8],  state[9],  # Row 2 (shifted)
        state[15], state[12], state[13], state[14], # Row 3 (shifted)
    ]


def _mix_columns(state):
    """
    MixColumns Step:
    Mixes the nibbles within each column using XOR operations. 
    This provides 'diffusion', spreading information across the block.
    """
    mixed = state[:]
    for c in range(4):
        # Extract the 4 nibbles in the current column 'c'.
        a0, a1, a2, a3 = state[c], state[4 + c], state[8 + c], state[12 + c]
        
        # Combine them using XOR (^) to create new nibbles.
        mixed[c]      = a0 ^ a1 ^ a2
        mixed[4 + c]  = a1 ^ a2 ^ a3
        mixed[8 + c]  = a0 ^ a2 ^ a3
        mixed[12 + c] = a0 ^ a1 ^ a3
        
    # Ensure each result is still just a 4-bit nibble.
    return [v & 0xF for v in mixed]


def _to_nibbles(x: int):
    """
    Helper function:
    Takes a 64-bit number and breaks it into a list of 16 4-bit nibbles.
    """
    return [((x >> (4 * (15 - i))) & 0xF) for i in range(16)]


def _from_nibbles(state):
    """
    Helper function:
    Takes a list of 16 nibbles and packs them back into one 64-bit number.
    """
    out = 0
    for value in state:
        out = (out << 4) | (value & 0xF)
    return out & MASK64


def led_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    MAIN LED ENCRYPTION FUNCTION:
    
    Arguments:
      plaintext: The 64-bit message to encrypt.
      key_words: Words forming the secret key.
      num_rounds: How many rounds of mixing to perform.
    """
    # 1. KEY PREPARATION
    # Assemble the secret key from the input words.
    key = 0
    for word in key_words:
        key = ((key << 16) | (int(word) & 0xFFFF)) & MASK64

    # 2. INITIALIZATION
    # XOR the plaintext with the key and convert it to the nibble grid.
    state = _to_nibbles((plaintext ^ key) & MASK64)
    rounds = max(1, int(num_rounds))

    # 3. ENCRYPTION LOOP
    for r in range(rounds):
        # STEP A: ADD CONSTANT
        # XOR a value based on the round 'r' to word 0 and word 4.
        state[0] ^= (r & 0xF)
        state[4] ^= ((r >> 1) & 0xF)
        
        # STEP B: SUB CELLS
        # Substitution for confusion.
        state = _sub_cells(state)
        
        # STEP C: SHIFT ROWS
        # Shuffling for diffusion.
        state = _shift_rows(state)
        
        # STEP D: MIX COLUMNS
        # Mathematical mixing within columns.
        state = _mix_columns(state)
        
        # STEP E: ADD ROUND KEY
        # We derive a round key by rotating our main key and XORing it.
        key_rotation = (4 * (r % 16))
        rotated_key = ((key << key_rotation) | (key >> (64 - key_rotation))) & MASK64
        round_key_nibbles = _to_nibbles(rotated_key)
        
        # XOR each nibble of the state with the round key.
        state = [(a ^ b) & 0xF for a, b in zip(state, round_key_nibbles)]

    # 4. FINALIZATION
    # Convert the nibble grid back into a 64-bit number.
    return _from_nibbles(state)
