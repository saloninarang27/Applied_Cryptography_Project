"""Cipher/permutation mapping package for dataset generation.

This package intentionally avoids eager imports to keep unrelated cipher runs
isolated from syntax issues in other modules. Import required ciphers directly,
for example: `from ciphers.simon import simon_encrypt`.
"""

__all__ = []

