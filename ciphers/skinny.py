"""
================================================================================
SKINNY - TWEAKABLE BLOCK CIPHER
================================================================================

OVERVIEW:
SKINNY is a family of lightweight block ciphers designed in 2016. It's meant 
to compete with NSA's SIMON cipher. One of its key features is that it's 
"Tweakable", meaning it can take an extra piece of information (the Tweak) 
along with the key to change the encryption output.

HOW IT WORKS:
SKINNY follows the Substitution-Permutation Network (SPN) model, similar to 
AES, but optimized for extremely low power consumption.
It organizes its state into a 4x4 grid of 4-bit cells (for SKINNY-64).

THE ROUND STEPS:
1. SUB-CELLS (Substitution):
   - Each 4-bit cell is replaced using an S-BOX table.
2. ADD-CONSTANTS:
   - A round-dependent constant is XORed into the state.
3. ADD-ROUND-TWEAKEY:
   - Parts of the key and tweak are XORed into the first two rows.
4. SHIFT-ROWS (Permutation):
   - Cells in each row are shifted to the left by different amounts.
5. MIX-COLUMNS (Diffusion):
   - Columns are mixed using a simple linear operation.

THIS MODULE:
This implementation uses a 64-bit block size. It allows for reduced-round 
encryption to help Machine Learning researchers understand how the security 
of the cipher builds up with each round.
================================================================================
"""

# Mask to keep numbers within 64 bits (16 cells of 4 bits).
MASK64 = (1 << 64) - 1

# SKINNY S-BOX: A 4-bit substitution table.
SBOX = [
    0xC, 0x6, 0x9, 0x0, 0x1, 0xA, 0x2, 0xB, 
    0x3, 0x8, 0x5, 0xD, 0x4, 0xE, 0x7, 0xF
]


def _to_cells(x: int):
    """
    Helper function:
    Breaks a 64-bit number into a list of 16 4-bit cells.
    """
    return [((x >> (4 * (15 - i))) & 0xF) for i in range(16)]


def _from_cells(cells):
    """
    Helper function:
    Packs a list of 16 4-bit cells into one 64-bit number.
    """
    out = 0
    for cell in cells:
        out = (out << 4) | (cell & 0xF)
    return out & MASK64


def _shift_rows(state):
    """
    ShiftRows Step:
    Row 0: no shift.
    Row 1: shift left by 1.
    Row 2: shift left by 2.
    Row 3: shift left by 3.
    """
    rows = [state[i:i + 4] for i in range(0, 16, 4)]
    for idx, row in enumerate(rows):
        # Perform the rotation for this row.
        rows[idx] = row[idx:] + row[:idx]
    # Flatten back to a single list.
    return rows[0] + rows[1] + rows[2] + rows[3]


def _mix_columns(state):
    """
    MixColumns Step:
    Mixes the four cells in each column using a simple XOR-based formula.
    """
    mixed = state[:]
    for col in range(4):
        # Extract the column.
        a0, a1, a2, a3 = state[col], state[4 + col], state[8 + col], state[12 + col]
        
        # Apply the SKINNY mixing formula.
        mixed[col]      = a0 ^ a2 ^ a3
        mixed[4 + col]  = a0
        mixed[8 + col]  = a1 ^ a2
        mixed[12 + col] = a0 ^ a2
        
    return [v & 0xF for v in mixed]


def skinny_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    MAIN SKINNY ENCRYPTION FUNCTION:
    
    Arguments:
      plaintext: The 64-bit block to encrypt.
      key_words: Words forming the secret "Tweakey".
      num_rounds: Number of rounds to perform.
    """
    # 1. SETUP
    # Combine key words into a 64-bit Tweakey.
    tweakey = 0
    for word in key_words:
        tweakey = ((tweakey << 16) | (int(word) & 0xFFFF)) & MASK64

    # Initial XOR (Pre-whitening).
    state = _to_cells((plaintext ^ tweakey) & MASK64)
    rounds = max(1, int(num_rounds))

    # 2. ROUND LOOP
    for r in range(rounds):
        # STEP A: SUBSTITUTION
        state = [SBOX[v] for v in state]
        
        # STEP B: ADD CONSTANTS
        # XOR unique round-dependent values into the state.
        state[0] ^= r & 0xF
        state[4] ^= (r >> 4) & 0xF
        
        # STEP C: ADD ROUND TWEAKEY
        # Generate a round-specific tweakey by rotating the master tweakey.
        tk_cells = _to_cells(((tweakey >> (4 * (r % 16))) | (tweakey << (64 - 4 * (r % 16)))) & MASK64)
        # Only the first 8 cells (top 2 rows) get the tweakey XORed in.
        for idx in range(8):
            state[idx] ^= tk_cells[idx]
            
        # STEP D: PERMUTATION (Shift Rows)
        state = _shift_rows(state)
        
        # STEP E: DIFFUSION (Mix Columns)
        state = _mix_columns(state)

    # 3. FINALIZATION
    return _from_cells(state)
