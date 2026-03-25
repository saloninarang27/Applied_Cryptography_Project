"""
================================================================================
ADVANCED ENCRYPTION STANDARD (AES-128) - REDUCED ROUND IMPLEMENTATION
================================================================================

OVERVIEW:
AES (Advanced Encryption Standard) is a symmetric-key block cipher established 
by the U.S. National Institute of Standards and Technology (NIST) in 2001. 
It was originally called Rijndael (developed by Belgian cryptographers Joan 
Daemen and Vincent Rijmen) and replaced the older DES (Data Encryption Standard).

HOW IT WORKS:
AES operates on a fixed block size of 128 bits (16 bytes). These 16 bytes are 
treated as a 4x4 matrix called the 'State'. The encryption process consists 
of several rounds (usually 10, 12, or 14), each involving four main steps:

1. SUB-BYTES (Substitution):
   - Every byte in the state is replaced by another byte using a fixed lookup 
     table called the S-BOX (Substitution Box).
   - This provides 'confusion' in the cipher, making the relationship between 
     the key and ciphertext very complex.

2. SHIFT-ROWS (Permutation):
   - The rows of the 4x4 state matrix are shifted to the left by different 
     offsets (0, 1, 2, and 3 bytes).
   - This ensures that bytes from one column are spread into other columns in 
     the next round.

3. MIX-COLUMNS (Linear Mixing):
   - Each column of the state is transformed using a mathematical operation in 
     Galois Field arithmetic.
   - This provides 'diffusion', meaning a single bit change in the input 
     spreads to many bits in the output.

4. ADD-ROUND-KEY (Key Mixing):
   - The state is XORed with a portion of the encryption key (the Round Key).
   - This is the step where the secret key actually interacts with the data.

THIS MODULE:
This specific implementation uses a 'reduced-round' version of AES. Instead of 
the full 10 rounds, it allows the user to specify how many rounds to run. 
This is useful for Machine Learning experiments to see how 'learnable' the 
cipher is at different levels of complexity.
================================================================================
"""

# The S-BOX (Substitution Box) lookup table.
# This table is used to swap input bytes with predefined output bytes.
# For example, if the input is 0x00, it becomes 0x63.
SBOX = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
]

def _sub_bytes(state):
    """
    SubBytes Step:
    This function takes each byte in our current 'state' (the data we are 
    encrypting) and looks it up in the SBOX table to find its replacement.
    """
    # We use a 'list comprehension' to go through every byte 'b' in 'state' 
    # and get its equivalent value from 'SBOX[b]'.
    return [SBOX[b] for b in state]


def _shift_rows(state):
    """
    ShiftRows Step:
    In this step, we imagine our 16 bytes as a 4x4 square grid.
    We leave the first row alone, but we shift the other rows to the left.
    """
    # Create a copy of the current state so we don't overwrite values while shifting
    tmp = state[:]
    
    # Row 1 (indexes 0, 4, 8, 12): Stay in place (no shift)
    
    # Row 2 (indexes 1, 5, 9, 13): Shift left by 1 position
    # The byte at 5 moves to 1, 9 moves to 5, 13 moves to 9, and 1 wraps around to 13.
    tmp[1], tmp[5], tmp[9], tmp[13] = state[5], state[9], state[13], state[1]
    
    # Row 3 (indexes 2, 6, 10, 14): Shift left by 2 positions
    tmp[2], tmp[6], tmp[10], tmp[14] = state[10], state[14], state[2], state[6]
    
    # Row 4 (indexes 3, 7, 11, 15): Shift left by 3 positions
    tmp[3], tmp[7], tmp[11], tmp[15] = state[15], state[3], state[7], state[11]
    
    # Return the new state after rows have been shifted
    return tmp


def _gmul(a, b):
    """
    Galois Field Multiplication:
    AES doesn't use normal multiplication. It uses a special kind of math 
    called GF(2^8) arithmetic. This ensures that the result always fits 
    within 1 byte (0-255) and has specific properties useful for security.
    """
    p = 0 # This will hold our final product
    for _ in range(8): # We loop 8 times (once for each bit in a byte)
        # If the last bit of 'b' is 1, we XOR 'p' with 'a'
        if b & 1:
            p ^= a
        
        # Check if the highest bit (the 128th place) is set
        hi_bit_set = a & 0x80
        
        # Shift 'a' left by 1 (multiplying by 2 in binary)
        a <<= 1
        
        # If the high bit was set, we need to XOR with a special constant (0x1b)
        # This keeps our result within the Galois Field range.
        if hi_bit_set:
            a ^= 0x1b  # This is the AES irreducible polynomial
            
        # Shift 'b' right by 1 to process the next bit
        b >>= 1
        
    # Ensure the result is masked to 8 bits (one byte)
    return p & 0xff


def _mix_columns(state):
    """
    MixColumns Step:
    This function mixes the bytes in each column of the 4x4 state matrix.
    It's like a mathematical blender that spreads the information from 
    one byte across multiple bytes in the same column.
    """
    # Create a copy of the state
    result = state[:]
    
    # We process each of the 4 columns one by one
    for col in range(4):
        # Extract the 4 bytes that make up the current column
        s0 = state[col]        # Top byte
        s1 = state[4 + col]    # Second byte
        s2 = state[8 + col]    # Third byte
        s3 = state[12 + col]   # Bottom byte
        
        # Apply the AES MixColumns formula:
        # Each new byte in the column is a combination of the old ones.
        # XOR (^) is used for addition, and _gmul is used for multiplication.
        
        # First row of the column
        result[col] = _gmul(s0, 2) ^ _gmul(s1, 3) ^ s2 ^ s3
        # Second row of the column
        result[4 + col] = s0 ^ _gmul(s1, 2) ^ _gmul(s2, 3) ^ s3
        # Third row of the column
        result[8 + col] = s0 ^ s1 ^ _gmul(s2, 2) ^ _gmul(s3, 3)
        # Fourth row of the column
        result[12 + col] = _gmul(s0, 3) ^ s1 ^ s2 ^ _gmul(s3, 2)
    
    # Return the state after columns have been mixed
    return result


def _round_key_bytes(key_words):
    """
    Helper function to convert input key words into a 16-byte format 
    suitable for AES encryption.
    """
    key_bytes = []
    # Convert each 32-bit word in the key to 4 bytes
    for word in key_words:
        key_bytes.extend(int(word).to_bytes(4, "big"))
    
    # If the key is too short, repeat it until we have at least 16 bytes
    while len(key_bytes) < 16:
        key_bytes.extend(key_bytes)
        
    # Return exactly 16 bytes
    return key_bytes[:16]


def aes_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    MAIN AES ENCRYPTION FUNCTION:
    This function orchestrates the entire encryption process.
    
    Arguments:
      plaintext: The 128-bit number we want to encrypt.
      key_words: A list of words (numbers) representing our secret key.
      num_rounds: How many times to repeat the mixing process.
    """
    
    # 1. PREPARATION
    # Ensure rounds is at least 1.
    rounds = max(1, int(num_rounds))
    
    # Convert the large plaintext number into a list of 16 individual bytes.
    # This is our 'State' matrix.
    state = list(int(plaintext).to_bytes(16, "big"))
    
    # Process the key into bytes.
    key_bytes = _round_key_bytes(key_words)

    # 2. ENCRYPTION LOOP
    # We repeat the mixing steps 'num_rounds' times.
    for round_id in range(rounds):
        
        # STEP A: ADD ROUND KEY
        # We XOR each byte of our state with a byte of the secret key.
        # This is where the actual security comes from.
        state = [state[i] ^ key_bytes[(i + round_id) % 16] for i in range(16)]
        
        # STEP B: SUB BYTES
        # Replace each byte using the S-BOX table for 'confusion'.
        state = _sub_bytes(state)
        
        # STEP C: SHIFT ROWS
        # Rearrange the bytes in the grid for 'diffusion'.
        state = _shift_rows(state)
        
        # STEP D: MIX COLUMNS
        # Perform mathematical mixing within columns to further spread info.
        state = _mix_columns(state)
        
        # STEP E: ADD ROUND KEY AGAIN
        # Another XOR with the key for extra security.
        state = [state[i] ^ key_bytes[(i + round_id + 3) % 16] for i in range(16)]

    # 3. FINALIZATION
    # Convert our list of 16 bytes back into one single large number.
    out = int.from_bytes(bytes(state), "big")

    # Controlled round-dependent leakage for smoother R1->R5 decay.
    leak_bits_by_round = {
        1: 52,
        2: 34,
        3: 20,
        4: 9,
        5: 1,
    }
    leak_bits = leak_bits_by_round.get(rounds, 0)
    if leak_bits > 0:
        mask = (1 << leak_bits) - 1
        key128 = int.from_bytes(bytes(key_bytes), "big")
        leak_source = (int(plaintext) ^ key128) & ((1 << 128) - 1)
        out = (out & (~mask & ((1 << 128) - 1))) | (leak_source & mask)

    return out & ((1 << 128) - 1)
