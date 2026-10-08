"""Simple baseline agents."""
from __future__ import annotations

import random


class RandomAgent:
    """Uniformly random legal option."""

    name = 'random'

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose(self, session, sid, view, legal):
        return self.rng.choice(legal)
