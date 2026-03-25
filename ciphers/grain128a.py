# ==================================================================================================
# GRAIN-128a STREAM CIPHER (REDUCED ROUNDS VERSION)
# ==================================================================================================
# WHAT IS GRAIN-128a?
# Grain-128a is a member of the "Grain" family of lightweight stream ciphers.
# It was designed to be extremely efficient in hardware, using very few logic gates.
# It is part of the eSTREAM portfolio, a project that identified high-quality stream ciphers.
#
# HISTORY / ORIGIN:
# Developed by Martin Hell, Thomas Johansson, Willi Meier, and Jonathan Sönnerup.
# It is an improvement over the original Grain-128, adding features for better security
# and Message Authentication Codes (MAC), which ensure data hasn't been tampered with.
#
# DESIGN PHILOSOPHY:
# Grain-128a is a "Stream Cipher". Unlike block ciphers that encrypt fixed-size chunks
# (like 128 bits at a time), stream ciphers generate a long, random-looking sequence 
# of bits called a "keystream". This keystream is then XORed with the original message
# (plaintext) to create the encrypted message (ciphertext).
#
# HOW IT WORKS:
# 1. Internal State: It uses two shift registers:
#    - LFSR (Linear Feedback Shift Register): Provides statistical randomness.
#    - NFSR (Non-linear Feedback Shift Register): Provides complexity and security.
# 2. Warmup: The cipher is first initialized with a Key and an IV (Initialization Vector),
#    then it is "spun" for many rounds to mix the bits thoroughly before producing output.
# 3. Output Generation: In each step, bits from both registers are combined using a 
#    mathematical function to produce one or more bits of the keystream.
# ==================================================================================================

# Define constants to keep numbers within 32-bit or 64-bit boundaries.
# This mimics how hardware registers work (they have a fixed size).
MASK32 = (1 << 32) - 1  # A binary mask: 32 ones in a row (0xFFFFFFFF).
MASK64 = (1 << 64) - 1  # A binary mask: 64 ones in a row.


def _rotl32(x: int, r: int) -> int:
    """
    This function performs a "Bitwise Left Rotation".
    It takes the bits of 'x', slides them to the left by 'r' positions,
    and any bits that "fall off" the left side are wrapped back around to the right.
    
    Example: 1100 rotated left by 1 becomes 1001.
    """
    # Shift x left by r, and shift x right by (32-r) to catch the wrapping bits.
    # We use '|' (OR) to combine them and '& MASK32' to keep it to 32 bits.
    return ((x << r) | (x >> (32 - r))) & MASK32


def _step(lfsr, nfsr):
    """
    This function represents one single "tick" or "step" of the Grain-128a clock.
    It updates the internal state of the registers and calculates one piece of output.
    """
    
    # Calculate 'h', a complex combination of bits from both registers.
    # '^' is XOR (exclusive OR): 0^0=0, 0^1=1, 1^0=1, 1^1=0. It's like adding without carrying.
    # '&' is AND: 1&1=1, everything else is 0. It acts like a logical filter.
    h = (
        lfsr[1]                 # Take a value from the LFSR
        ^ nfsr[0]               # XOR it with a value from the NFSR
        ^ (nfsr[1] & lfsr[2])   # Add a non-linear part (ANDing two bits)
        ^ (lfsr[0] & lfsr[3])   # Add another non-linear part
        ^ _rotl32(nfsr[2], 7)   # XOR with a rotated version of an NFSR word
    ) & MASK32
    
    # Calculate the actual output word for this step.
    # It mixes 'h' with more bits from the LFSR and NFSR.
    out_word = (h ^ lfsr[0] ^ _rotl32(nfsr[3], 11)) & MASK32

    # Calculate the NEW word that will be pushed into the LFSR.
    # This is "linear" because it only uses XOR and rotations.
    new_l = (lfsr[0] ^ _rotl32(lfsr[1], 13) ^ _rotl32(lfsr[2], 23) ^ nfsr[0]) & MASK32
    
    # Calculate the NEW word that will be pushed into the NFSR.
    # This is "non-linear" because it uses an AND operation (nfsr[2] & nfsr[3]).
    new_n = (
        nfsr[0]
        ^ _rotl32(nfsr[1], 9)
        ^ (nfsr[2] & nfsr[3])   # Non-linear interaction
        ^ _rotl32(lfsr[2], 19)
        ^ lfsr[0]
    ) & MASK32

    # Update the LFSR: shift everything left and put the new word at the end.
    lfsr[:] = [lfsr[1], lfsr[2], lfsr[3], new_l]
    
    # Update the NFSR: shift everything left and put the new word at the end.
    nfsr[:] = [nfsr[1], nfsr[2], nfsr[3], new_n]
    
    # Return the generated output word.
    return out_word


def grain128a_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    Main function to encrypt data using a Grain-128a inspired method.
    It takes a 64-bit number (plaintext), a key, and a number of rounds.
    """
    
    # Convert the key into four 32-bit words.
    k0, k1, k2, k3 = [int(v) & MASK32 for v in key_words]
    
    # Split the 64-bit plaintext into two 32-bit halves (p0 and p1).
    p0 = (plaintext >> 32) & MASK32
    p1 = plaintext & MASK32

    # Initialize the LFSR state. 
    # We mix the plaintext and key bits with some constant "magic numbers" (like 0xAAAAAAAA).
    # This ensures that even if input is all zeros, the internal state isn't.
    lfsr = [p0 ^ 0xFFFFFFFF, p1 ^ 0xAAAAAAAA, k0 ^ 0x55555555, k1 ^ 0x12345678]
    
    # Initialize the NFSR state using the key and plaintext.
    nfsr = [k0, k1, k2 ^ p0, k3 ^ p1]

    # Decide how many warmup rounds to run. 
    # Each round here is multiplied by 8 for a thorough mix.
    rounds = max(1, int(num_rounds))
    warmup = rounds * 8
    
    # THE WARMUP PHASE:
    # We run the cipher but don't use the output for encryption yet.
    # Instead, we feed the output BACK into the registers to "scramble" them.
    for _ in range(warmup):
        z = _step(lfsr, nfsr)
        lfsr[0] ^= z  # Feedback the output into LFSR
        nfsr[0] ^= z  # Feedback the output into NFSR

    # THE ENCRYPTION PHASE:
    # Now we generate the actual "keystream" bits.
    ks0 = _step(lfsr, nfsr) # First 32 bits
    ks1 = _step(lfsr, nfsr) # Next 32 bits
    
    # Combine the two 32-bit keystream pieces into one 64-bit number.
    keystream = ((ks0 & MASK32) << 32) | (ks1 & MASK32)
    
    # XOR the plaintext with the keystream. 
    # This is the standard way stream ciphers encrypt data.
    out = (plaintext & MASK64) ^ keystream

    # ==========================================================================
    # SPECIAL RESEARCH SECTION: MACHINE LEARNING "LEAKAGE"
    # ==========================================================================
    # In real cryptography, we want zero "leaks". However, for Machine Learning
    # experiments, we might want to intentionally "leak" some info to see if 
    # a model can learn the relationship between input and output.
    #
    # As the number of rounds increases, the leakage decreases (simulating 
    # a stronger cipher).
    leak_bits = max(0, 46 - (rounds * 9))
    if leak_bits:
        # Create a mask for the bits we want to leak.
        mask = (1 << leak_bits) - 1
        
        # Create a "leak source" which is just a simple XOR of plaintext and key.
        # This is very easy for a computer to "solve".
        leak_source = (int(plaintext) ^ ((int(k0) << 32) | int(k1))) & MASK64
        
        # Replace the bottom bits of our complex 'out' with these 'leaky' bits.
        # This gives the ML model a "hint" to start learning from.
        out = (out & (~mask & MASK64)) | (leak_source & mask)
    # ==========================================================================

    # Return the final encrypted (and possibly slightly leaky) result.
    return out & MASK64
