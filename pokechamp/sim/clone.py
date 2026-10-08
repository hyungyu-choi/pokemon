"""Cloning of running battles (for search / rollouts).

``copy.deepcopy`` copies the whole battle state (sides, Pokemon, queue, PRNG, effect
states) while sharing the immutable dex data (species, moves, items, abilities,
conditions, formats), so identity comparisons against dex effects keep working.
"""
from __future__ import annotations

import copy

from .js import NULL


def _dex_memo(dex) -> dict:
    memo = {id(dex): dex, id(NULL): NULL}
    for table in (dex.moves, dex.abilities, dex.items, dex.species, dex.natures):
        memo[id(table)] = table
        memo[id(table.empty)] = table.empty
        for v in table.data.values():
            memo[id(v)] = v
        for v in getattr(table, 'cache', {}).values():
            memo[id(v)] = v
    for v in dex.conditions.cache.values():
        memo[id(v)] = v
    memo[id(dex.conditions.empty)] = dex.conditions.empty
    for v in dex.rulesets.values():
        memo[id(v)] = v
    formats = dex.formats
    memo[id(formats)] = formats
    for v in getattr(formats, 'cache', {}).values():
        memo[id(v)] = v
    return memo


def clone_battle(battle):
    """Independent copy of ``battle`` (same RNG state; continue it with different choices)."""
    memo = _dex_memo(battle.dex)
    memo[id(battle.send)] = battle.send
    memo[id(battle.format)] = battle.format
    memo[id(battle.ruleTable)] = battle.ruleTable
    return copy.deepcopy(battle, memo)
