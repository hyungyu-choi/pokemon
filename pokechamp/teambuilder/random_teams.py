"""Random (but legal) Pokemon Champions team generation.

Used both for differential testing of the engine and as the starting point of
the evolutionary team optimiser: every gene (species, ability, item, moves,
nature, Stat Points) starts out uniformly random.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..sim.dex import ModdedDex, get_dex
from ..sim.js import Obj

STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
SP_TOTAL = 66
SP_MAX = 32
LEVEL = 50


@dataclass
class LegalPool:
    species: list[str]
    abilities: dict[str, list[str]]
    moves: dict[str, list[str]]
    items: list[str]
    mega_stones: dict[str, list[str]]  # species name -> mega stone names
    natures: list[str]
    meta: dict = field(default_factory=dict)


_POOLS: dict[str, LegalPool] = {}


def legal_pool(dex: ModdedDex | None = None, formatid: str = 'gen9championsbssregmc') -> LegalPool:
    """Everything a team of ``formatid`` may contain (from Showdown's TeamValidator, see legality.json)."""
    from .validator import format_legality
    if formatid in _POOLS and dex is None:
        return _POOLS[formatid]
    dex = dex or get_dex()
    leg = format_legality(formatid)
    species = sorted(leg.species)
    abilities = {s: list(leg.species[s]['abilities']) for s in species}
    moves = {s: sorted(leg.species[s]['moves']) for s in species}
    items = []
    mega_stones: dict[str, list[str]] = {}
    for name in leg.items:
        it = dex.items.get(name)
        if it.megaStone:
            for base in it.megaStone:
                mega_stones.setdefault(base, []).append(it.name)
            continue
        items.append(it.name)
    natures = sorted(n.name for n in dex.natures.all())
    nums = {s: leg.species[s]['num'] for s in species}
    all_items = set(leg.items)
    excluded = {s: all_items - set(leg.species[s]['items']) for s in species}
    pool = LegalPool(species, abilities, moves, sorted(items), mega_stones, natures,
                     meta={'num': nums, 'excluded_items': excluded})
    if dex is get_dex():
        _POOLS[formatid] = pool
    return pool


def random_sp(rng: random.Random) -> Obj:
    """Random Stat Point spread (<= 32 per stat, <= 66 in total)."""
    evs = {s: 0 for s in STATS}
    budget = SP_TOTAL
    order = list(STATS)
    rng.shuffle(order)
    for s in order:
        if budget <= 0:
            break
        amount = rng.randint(0, min(SP_MAX, budget))
        evs[s] = amount
        budget -= amount
    # spend remaining points randomly
    while budget > 0:
        candidates = [s for s in STATS if evs[s] < SP_MAX]
        if not candidates:
            break
        s = rng.choice(candidates)
        add = min(budget, SP_MAX - evs[s], rng.randint(1, 8))
        evs[s] += add
        budget -= add
    return Obj(evs)


class RandomTeamGenerator:
    def __init__(self, seed: int | None = None, dex: ModdedDex | None = None, mega_chance: float = 0.35,
                 formatid: str = 'gen9championsbssregmc'):
        self.rng = random.Random(seed)
        self.pool = legal_pool(dex, formatid)
        self.mega_chance = mega_chance

    def random_set(self, species: str | None = None, used_items: set | None = None) -> Obj:
        rng = self.rng
        pool = self.pool
        species = species or rng.choice(pool.species)
        movepool = pool.moves[species]
        moves = rng.sample(movepool, min(4, len(movepool)))
        used_items = used_items if used_items is not None else set()
        banned = pool.meta['excluded_items'].get(species, set())
        item = ''
        base = species
        stones = [st for st in pool.mega_stones.get(base, []) if st not in used_items and st not in banned]
        if stones and rng.random() < self.mega_chance:
            item = rng.choice(stones)
        else:
            choices = [i for i in pool.items if i not in used_items and i not in banned]
            if choices:
                item = rng.choice(choices)
        if item:
            used_items.add(item)
        return Obj(
            species=species,
            name=species,
            item=item,
            ability=rng.choice(pool.abilities[species]),
            moves=moves,
            nature=rng.choice(pool.natures),
            evs=random_sp(rng),
            level=LEVEL,
        )

    def random_team(self, size: int = 6) -> list[Obj]:
        # Species Clause: at most one Pokemon per National Dex number
        nums = self.pool.meta['num']
        species = []
        used_nums = set()
        for s in self.rng.sample(self.pool.species, len(self.pool.species)):
            if nums[s] in used_nums:
                continue
            species.append(s)
            used_nums.add(nums[s])
            if len(species) == size:
                break
        used_items: set = set()
        return [self.random_set(s, used_items) for s in species]
