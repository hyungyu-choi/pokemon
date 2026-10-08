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


_POOL: LegalPool | None = None


def legal_pool(dex: ModdedDex | None = None) -> LegalPool:
    """Everything a Champions team may contain (species, abilities, learnable moves, items)."""
    global _POOL
    if _POOL is not None and dex is None:
        return _POOL
    dex = dex or get_dex()
    species = []
    abilities = {}
    moves = {}
    for s in dex.species.all():
        if not s.legal or s.battleOnly or s.isMega or s.tier == 'Illegal' or s.isNonstandard:
            continue
        if s.requiredItem or s.requiredItems:
            continue
        learn = [m for m in dex.learnset(s.id) if dex.moves.get(m).exists and not dex.moves.get(m).isNonstandard]
        if len(learn) < 1:
            continue
        species.append(s.name)
        abilities[s.name] = sorted(set(a for a in s.abilities.values() if dex.abilities.get(a).exists))
        moves[s.name] = sorted(set(dex.moves.get(m).name for m in learn))
    items = []
    mega_stones: dict[str, list[str]] = {}
    for it in dex.items.all():
        if it.isNonstandard:
            continue
        if it.megaStone:
            for base, mega in it.megaStone.items():
                mega_stones.setdefault(base, []).append(it.name)
            continue
        items.append(it.name)
    natures = sorted(n.name for n in dex.natures.all())
    pool = LegalPool(sorted(species), abilities, moves, sorted(items), mega_stones, natures)
    if dex is get_dex():
        _POOL = pool
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
    def __init__(self, seed: int | None = None, dex: ModdedDex | None = None, mega_chance: float = 0.35):
        self.rng = random.Random(seed)
        self.pool = legal_pool(dex)
        self.mega_chance = mega_chance

    def random_set(self, species: str | None = None, used_items: set | None = None) -> Obj:
        rng = self.rng
        pool = self.pool
        species = species or rng.choice(pool.species)
        movepool = pool.moves[species]
        moves = rng.sample(movepool, min(4, len(movepool)))
        used_items = used_items if used_items is not None else set()
        item = ''
        base = species
        stones = [st for st in pool.mega_stones.get(base, []) if st not in used_items]
        if stones and rng.random() < self.mega_chance:
            item = rng.choice(stones)
        else:
            choices = [i for i in pool.items if i not in used_items]
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
        species = self.rng.sample(self.pool.species, size)
        used_items: set = set()
        return [self.random_set(s, used_items) for s in species]
