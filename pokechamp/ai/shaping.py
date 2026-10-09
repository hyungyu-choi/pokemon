"""Torch-free helpers shared by the neural-network agent, training and the advisor.

Kept separate from :mod:`pokechamp.ai.nn_agent` (which imports torch) so the advisor and the
battle assistant UI still work with the heuristic AI when PyTorch is not installed.
"""
from __future__ import annotations

from ..env.view import BattleView


def potential(view: BattleView) -> float:
    """Material balance used for reward shaping: (our HP - their HP) / Pokemon brought, in [-1, 1]."""
    def side_hp(side):
        hp = sum(p.hp for p in side.pokemon if p.revealed and not p.fainted)
        revealed = sum(1 for p in side.pokemon if p.revealed)
        hp += max(0, side.team_size - revealed)  # unseen Pokemon are at full HP
        return hp / max(1, side.team_size)
    return side_hp(view.my_side) - side_hp(view.foe_side)
