"""Pokemon Champions team validation.

The legal species / abilities / moves / items per format come from
``pokechamp/data/legality.json``, which the exporter computes with Showdown's
own ``TeamValidator`` (so e.g. Greninja's Battle Bond or Mythical / Restricted
Legendary Pokemon are rejected exactly like on Showdown).  On top of that the
team-level clauses of the Champions "Flat Rules" are checked here:

* 6 Pokemon per team (bring 6, pick 3 in singles / 4 in doubles)
* Species Clause (one of each National Dex number), Item Clause (one of each item),
  Nickname Clause
* Stat Points: at most 32 per stat and 66 in total
* at most 4 distinct moves
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from ..sim.dex import get_dex
from ..sim.js import to_id

STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
SP_MAX = 32
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


@lru_cache(maxsize=None)
def load_legality() -> dict:
    with open(os.path.join(DATA_DIR, 'legality.json'), encoding='utf-8') as f:
        return json.load(f)


class FormatLegality:
    """Legal content of one format (species -> abilities / moves / items)."""

    def __init__(self, formatid: str):
        table = load_legality()
        if formatid not in table:
            raise ValueError(f"no legality data for format {formatid!r} (have: {', '.join(table)})")
        data = table[formatid]
        self.formatid = formatid
        self.ev_limit = data.get('evLimit') or 66
        self.picked_team_size = data.get('pickedTeamSize')
        self.min_team_size = data.get('minTeamSize') or 1
        self.max_team_size = data.get('maxTeamSize') or 6
        self.items = list(data['items'])
        self.species = {}
        for name, sp in data['species'].items():
            excluded = set(sp.get('itemsExcluded') or ())
            self.species[name] = {
                'abilities': list(sp['abilities']),
                'moves': list(sp['moves']),
                'items': [i for i in self.items if i not in excluded],
                'num': sp['num'],
            }
        self._move_ids = {name: {to_id(m) for m in sp['moves']} for name, sp in self.species.items()}
        self._ability_ids = {name: {to_id(a) for a in sp['abilities']} for name, sp in self.species.items()}
        self._item_ids = {name: {to_id(i) for i in sp['items']} for name, sp in self.species.items()}

    def species_name(self, species: str) -> str | None:
        s = get_dex().species.get(species)
        if s.exists and s.name in self.species:
            return s.name
        return None

    def can_learn(self, species: str, move: str) -> bool:
        name = self.species_name(species)
        return bool(name) and to_id(move) in self._move_ids[name]

    def can_have_ability(self, species: str, ability: str) -> bool:
        name = self.species_name(species)
        return bool(name) and to_id(ability) in self._ability_ids[name]

    def can_hold(self, species: str, item: str) -> bool:
        if not item:
            return True
        name = self.species_name(species)
        return bool(name) and to_id(item) in self._item_ids[name]


@lru_cache(maxsize=None)
def format_legality(formatid: str = 'gen9championsbssregmc') -> FormatLegality:
    return FormatLegality(formatid)


class TeamValidator:
    """``TeamValidator('gen9championsbssregmc').validate_team(team)`` -> list of problems (empty = legal)."""

    def __init__(self, formatid: str = 'gen9championsbssregmc'):
        self.formatid = formatid
        self.legality = format_legality(formatid)
        self.dex = get_dex()

    def validate_set(self, set_) -> list[str]:
        problems = []
        dex = self.dex
        leg = self.legality
        species_name = set_.get('species') or set_.get('name') or ''
        label = set_.get('name') or species_name or '(no species)'
        species = dex.species.get(species_name)
        if not species.exists:
            return [f"{label}: unknown species {species_name!r}"]
        name = leg.species_name(species_name)
        if name is None:
            return [f"{label}: {species.name} is not allowed in {self.formatid}"]

        ability = set_.get('ability') or ''
        if not dex.abilities.get(ability).exists:
            problems.append(f"{label}: unknown ability {ability!r}")
        elif not leg.can_have_ability(name, ability):
            problems.append(f"{label}: {name} can't have {dex.abilities.get(ability).name}")

        item = set_.get('item') or ''
        if item:
            if not dex.items.get(item).exists:
                problems.append(f"{label}: unknown item {item!r}")
            elif not leg.can_hold(name, item):
                problems.append(f"{label}: {dex.items.get(item).name} is not allowed")

        moves = [m for m in (set_.get('moves') or []) if m]
        if not moves:
            problems.append(f"{label}: has no moves")
        if len(moves) > 4:
            problems.append(f"{label}: has more than 4 moves")
        seen = set()
        for m in moves:
            mid = to_id(m)
            if mid.startswith('hiddenpower'):
                mid = 'hiddenpower'
            if not dex.moves.get(mid).exists:
                problems.append(f"{label}: unknown move {m!r}")
                continue
            if mid in seen:
                problems.append(f"{label}: has {dex.moves.get(mid).name} more than once")
            seen.add(mid)
            if not leg.can_learn(name, mid):
                problems.append(f"{label}: {name} can't learn {dex.moves.get(mid).name}")

        nature = set_.get('nature') or ''
        if nature and not dex.natures.get(nature).exists:
            problems.append(f"{label}: unknown nature {nature!r}")

        evs = set_.get('evs') or {}
        total = 0
        for stat in STATS:
            v = evs.get(stat) or 0
            if not isinstance(v, int) or v < 0:
                problems.append(f"{label}: invalid Stat Points for {stat}: {v!r}")
                continue
            if v > SP_MAX:
                problems.append(f"{label}: {v} Stat Points in {stat} (max {SP_MAX})")
            total += v
        if total > leg.ev_limit:
            problems.append(f"{label}: {total} Stat Points in total (max {leg.ev_limit})")
        return problems

    def validate_team(self, team) -> list[str]:
        problems = []
        leg = self.legality
        if len(team) < leg.min_team_size:
            problems.append(f"team has {len(team)} Pokemon (need at least {leg.min_team_size})")
        if len(team) > leg.max_team_size:
            problems.append(f"team has {len(team)} Pokemon (max {leg.max_team_size})")
        nums = {}
        items = {}
        names = {}
        for set_ in team:
            problems.extend(self.validate_set(set_))
            species = self.dex.species.get(set_.get('species') or set_.get('name') or '')
            if species.exists:
                if species.num in nums:
                    problems.append(f"Species Clause: {nums[species.num]} and {species.name}")
                nums[species.num] = species.name
            item = to_id(set_.get('item') or '')
            if item:
                if item in items:
                    problems.append(f"Item Clause: more than one {self.dex.items.get(item).name}")
                items[item] = True
            nick = set_.get('name') or species.baseSpecies
            if nick:
                if nick in names:
                    problems.append(f"Nickname Clause: more than one Pokemon named {nick!r}")
                names[nick] = True
        return problems

    def is_legal(self, team) -> bool:
        return not self.validate_team(team)
