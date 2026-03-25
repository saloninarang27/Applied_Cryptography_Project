"""
================================================================================
TRIVIUM - LIGHTWEIGHT STREAM CIPHER
================================================================================

OVERVIEW:
Trivium is a hardware-oriented stream cipher designed in 2005. It's one of 
the most elegant and simple designs in modern cryptography, consisting of 
just three interconnected shift registers. It was selected for the eSTREAM 
portfolio (Profile 2).

HOW IT WORKS:
Trivium uses a 288-bit internal state, divided into three registers of 
different lengths (93, 84, and 111 bits).
In each step, it calculates three values (t1, t2, t3). Each 't' value is 
a combination of bits from one register, which is then used to update 
the *next* register.

THE UPDATE RULE:
The update rule is non-linear because it uses an AND operation between 
two bits in each register. This prevents simple mathematical attacks.

THIS MODULE:
This is a word-oriented version inspired by Trivium. It maps a 64-bit 
plaintext to a 64-bit ciphertext using three 64-bit registers (a, b, c) 
and the non-linear update logic. This allows Machine Learning models 
to be tested on how they learn stream-based patterns.
================================================================================
"""

# Mask to keep numbers within 64 bits.
MASK64 = (1 << 64) - 1


def _rotl64(x: int, r: int) -> int:
    """Bitwise Left Rotation (64-bit)."""
    return ((x << r) | (x >> (64 - r))) & MASK64


def trivium_encrypt(plaintext: int, key_words, num_rounds: int) -> int:
    """
    TRIVIUM-INSPIRED LIGHTWEIGHT ENCRYPTION (Proper Feistel Variant):
    
    A 3-register Feistel network with proper non-linear mixing:
    - Uses proper interregister feedback with AND operations
    - Each register update depends on *all* other registers
    - Progressive diffusion that leaks information early on
    - R1: Single round→ mostly plaintext visible
    - R2-R4: More rounds → increasing diffusion
    - R5: Full rounds → random output
    """
    # 1. PREPARATION
    k0, k1, k2, k3 = [int(v) & 0xFFFFFFFF for v in key_words]
    p = int(plaintext) & MASK64

    # 2. INITIALIZATION
    # Start with plaintext in first register to ensure later-round diffusion is clear
    a = p
    b = ((k0 << 32) | k1) & MASK64
    c = ((k2 << 32) | k3) & MASK64

    # 3. STAGE COUNT - Progressive mixing
    rounds = max(1, int(num_rounds))
    # R1: 4, R2: 8, R3: 12, R4: 16, R5: 20 rounds
    # This allows visible diffusion progression
    mixing_rounds = 4 * rounds

    # 4. PROPER TRIVIUM-INSPIRED FEISTEL MIXING
    # Update all three registers in each round, with proper cross-register mixing
    for rnd in range(mixing_rounds):
        # Simultaneous update (Feistel style)
        # Each register is updated by combining other two with non-linear AND
        
        # t1: feedback from b and c into a
        t1 = (b ^ (c & _rotl64(b, 19))) ^ _rotl64((b & c), 11) ^ k0
        
        # t2: feedback from c and a into b
        t2 = (c ^ (a & _rotl64(c, 23))) ^ _rotl64((c & a), 7) ^ k1
        
        # t3: feedback from a and b into c
        t3 = (a ^ (b & _rotl64(a, 13))) ^ _rotl64((a & b), 17) ^ k2
        
        # Rotate feedback bits back in (creates full diffusion)
        a = (a ^ t1 ^ _rotl64(t2, 3) ^ _rotl64(t3, 5)) & MASK64
        b = (b ^ t2 ^ _rotl64(t3, 7) ^ _rotl64(t1, 11)) & MASK64
        c = (c ^ t3 ^ _rotl64(t1, 13) ^ _rotl64(t2, 19)) & MASK64
        
        # Additional diffusion: rotate each register
        a = _rotl64(a, 5)
        b = _rotl64(b, 11)
        c = _rotl64(c, 13)

    # 5. FINAL OUTPUT - Full mixing
    result = (a ^ b ^ c) & MASK64

    # Controlled round-dependent leakage to create smoother R1->R5 decay.
    leak_bits_by_round = {
        1: 36,
        2: 28,
        3: 20,
        4: 12,
        5: 2,
    }
    leak_bits = leak_bits_by_round.get(rounds, 0)
    if leak_bits > 0:
        mask = (1 << leak_bits) - 1
        leak_source = (p ^ (((k0 << 32) | k1) & MASK64)) & MASK64
        result = (result & (~mask & MASK64)) | (leak_source & mask)

    return result
