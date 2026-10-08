"""Port of Showdown's ``sim/prng.ts``.

Supports the Gen 5 LCG seeds (``"gen5,xxxxxxxxxxxxxxxx"`` or ``"a,b,c,d"``) and
``sodium`` (ChaCha20) seeds, so battles are reproducible bit-for-bit against the
reference implementation given the same seed and choices.
"""
from __future__ import annotations

import math
import os
import struct

_MASK64 = 0xFFFFFFFFFFFFFFFF
_GEN5_A = 0x5D588B656C078965
_GEN5_C = 0x00269EC3


class Gen5RNG:
    __slots__ = ('seed',)

    def __init__(self, seed: list[int] | None = None):
        if seed is None:
            seed = [int.from_bytes(os.urandom(2), 'big') for _ in range(4)]
        self.seed = (seed[0] << 48) | (seed[1] << 32) | (seed[2] << 16) | seed[3]

    def get_seed(self) -> str:
        s = self.seed
        return f"{(s >> 48) & 0xFFFF},{(s >> 32) & 0xFFFF},{(s >> 16) & 0xFFFF},{s & 0xFFFF}"

    def next(self) -> int:
        self.seed = (self.seed * _GEN5_A + _GEN5_C) & _MASK64
        return self.seed >> 32


def _rotl(v: int, c: int) -> int:
    return ((v << c) & 0xFFFFFFFF) | (v >> (32 - c))


def _chacha20_block(key: bytes, counter: int, nonce: bytes) -> bytes:
    const = (0x61707865, 0x3320646e, 0x79622d32, 0x6b206574)
    k = struct.unpack('<8I', key)
    n = struct.unpack('<3I', nonce)
    state = [*const, *k, counter & 0xFFFFFFFF, *n]
    x = list(state)

    def qr(a, b, c, d):
        x[a] = (x[a] + x[b]) & 0xFFFFFFFF; x[d] = _rotl(x[d] ^ x[a], 16)
        x[c] = (x[c] + x[d]) & 0xFFFFFFFF; x[b] = _rotl(x[b] ^ x[c], 12)
        x[a] = (x[a] + x[b]) & 0xFFFFFFFF; x[d] = _rotl(x[d] ^ x[a], 8)
        x[c] = (x[c] + x[d]) & 0xFFFFFFFF; x[b] = _rotl(x[b] ^ x[c], 7)

    for _ in range(10):
        qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15)
        qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14)
    return struct.pack('<16I', *[(x[i] + state[i]) & 0xFFFFFFFF for i in range(16)])


class SodiumRNG:
    """Drop-in for libsodium's randombytes_buf_deterministic (ChaCha20, IETF)."""

    NONCE = b'LibsodiumDRG'
    __slots__ = ('seed',)

    def __init__(self, seed_hex: str):
        self.seed = bytes.fromhex(seed_hex.ljust(64, '0')[:64])

    def get_seed(self) -> str:
        return 'sodium,' + self.seed.hex()

    def next(self) -> int:
        buf = _chacha20_block(self.seed, 0, self.NONCE)
        self.seed = buf[:32]
        return int.from_bytes(buf[32:36], 'big')


class PRNG:
    """High-level PRNG API identical to Showdown's ``PRNG``."""

    __slots__ = ('starting_seed', 'rng')

    def __init__(self, seed: str | list | None = None, initial_seed: str | None = None):
        if not seed:
            seed = PRNG.generate_seed()
        if isinstance(seed, (list, tuple)):
            seed = ','.join(str(s) for s in seed)
        self.starting_seed = initial_seed if initial_seed is not None else seed
        self.set_seed(seed)

    def set_seed(self, seed: str) -> None:
        if seed.startswith('sodium,'):
            self.rng = SodiumRNG(seed.split(',')[1])
        elif seed.startswith('gen5,'):
            parts = [seed[5:9], seed[9:13], seed[13:17], seed[17:21]]
            self.rng = Gen5RNG([int(p, 16) for p in parts])
        elif seed[:1].isdigit():
            self.rng = Gen5RNG([int(float(p)) for p in seed.split(',')])
        else:
            raise ValueError(f"Unrecognized RNG seed {seed}")

    def get_seed(self) -> str:
        return self.rng.get_seed()

    def clone(self) -> 'PRNG':
        return PRNG(self.rng.get_seed(), self.starting_seed)

    def random(self, from_=None, to=None):
        result = self.rng.next()
        if from_:
            from_ = math.floor(from_)
        if to:
            to = math.floor(to)
        if from_ is None:
            return result / 4294967296
        elif not to:
            return (result * from_) // 4294967296
        else:
            return (result * (to - from_)) // 4294967296 + from_

    def random_chance(self, numerator, denominator) -> bool:
        return self.random(denominator) < numerator

    def sample(self, items):
        if not len(items):
            raise IndexError('Cannot sample an empty array')
        return items[self.random(len(items))]

    def shuffle(self, items: list, start: int = 0, end: int | None = None) -> None:
        if end is None:
            end = len(items)
        while start < end - 1:
            next_index = self.random(start, end)
            if start != next_index:
                items[start], items[next_index] = items[next_index], items[start]
            start += 1

    @staticmethod
    def generate_seed() -> str:
        return 'sodium,' + os.urandom(16).hex()

    # Showdown-style aliases
    randomChance = random_chance
    getSeed = get_seed
