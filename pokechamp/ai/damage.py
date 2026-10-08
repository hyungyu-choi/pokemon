"""Fast damage / speed estimation from a player's point of view.

The exact damage formula lives in the battle engine, but an AI only knows what
it has seen.  This module estimates damage with the Gen 9 formula (Champions
stats: level 50, Stat Points) using known information where available and
reasonable assumptions otherwise (opponent Stat Points spread, unknown item /
ability ignored unless the species can only have one ability).

It is used by the heuristic agents and to build features for the neural network.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..env.view import BattleView, PokemonView
from ..sim.dex import get_dex
from ..sim.js import to_id

STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
BOOST_TABLE = {i: (max(2, 2 + i) / max(2, 2 - i)) for i in range(-6, 7)}
# assumed Stat Points of an opponent whose spread is unknown (66 in total)
ASSUMED_SP = {'hp': 16, 'atk': 0, 'def': 5, 'spa': 0, 'spd': 5, 'spe': 16}
ASSUMED_SP_ATTACK = 24

TYPE_BOOST_ITEMS = {
    'Black Belt': 'Fighting', 'Black Glasses': 'Dark', 'Charcoal': 'Fire', 'Dragon Fang': 'Dragon',
    'Fairy Feather': 'Fairy', 'Hard Stone': 'Rock', 'Magnet': 'Electric', 'Metal Coat': 'Steel',
    'Miracle Seed': 'Grass', 'Mystic Water': 'Water', 'Never-Melt Ice': 'Ice', 'Poison Barb': 'Poison',
    'Sharp Beak': 'Flying', 'Silk Scarf': 'Normal', 'Silver Powder': 'Bug', 'Soft Sand': 'Ground',
    'Spell Tag': 'Ghost', 'Twisted Spoon': 'Psychic',
}
RESIST_BERRIES = {
    'Occa Berry': 'Fire', 'Passho Berry': 'Water', 'Wacan Berry': 'Electric', 'Rindo Berry': 'Grass',
    'Yache Berry': 'Ice', 'Chople Berry': 'Fighting', 'Kebia Berry': 'Poison', 'Shuca Berry': 'Ground',
    'Coba Berry': 'Flying', 'Payapa Berry': 'Psychic', 'Tanga Berry': 'Bug', 'Charti Berry': 'Rock',
    'Kasib Berry': 'Ghost', 'Haban Berry': 'Dragon', 'Colbur Berry': 'Dark', 'Babiri Berry': 'Steel',
    'Roseli Berry': 'Fairy', 'Chilan Berry': 'Normal',
}
ATE_ABILITIES = {'Aerilate': 'Flying', 'Pixilate': 'Fairy', 'Refrigerate': 'Ice', 'Galvanize': 'Electric',
                 'Dragonize': 'Dragon'}
IMMUNITY_ABILITIES = {
    'Levitate': 'Ground', 'Flash Fire': 'Fire', 'Well-Baked Body': 'Fire', 'Water Absorb': 'Water',
    'Storm Drain': 'Water', 'Dry Skin': 'Water', 'Volt Absorb': 'Electric', 'Lightning Rod': 'Electric',
    'Motor Drive': 'Electric', 'Sap Sipper': 'Grass', 'Earth Eater': 'Ground',
}


@lru_cache(maxsize=None)
def type_multiplier(move_type: str, defender_types: tuple) -> float:
    dex = get_dex()
    if not move_type or move_type == '???':
        return 1.0
    if not dex.getImmunity(move_type, list(defender_types)):
        return 0.0
    return 2.0 ** dex.getEffectiveness(move_type, list(defender_types))


@lru_cache(maxsize=None)
def species_info(species: str):
    s = get_dex().species.get(species)
    if not s.exists:
        return None
    abilities = tuple(sorted(set(a for a in s.abilities.values() if a)))
    return s.baseStats, tuple(s.types), abilities, float(s.weightkg or 1.0), bool(s.nfe)


@lru_cache(maxsize=None)
def move_info(name: str):
    m = get_dex().moves.get(name)
    if not m.exists:
        return None
    return m


def nature_mult(nature: str | None, stat: str) -> float:
    if not nature:
        return 1.0
    n = get_dex().natures.get(nature)
    if n.plus == stat:
        return 1.1
    if n.minus == stat:
        return 0.9
    return 1.0


def calc_stat(base: int, sp: int, stat: str, nature: str | None = None) -> int:
    if stat == 'hp':
        return base + sp + 75
    return int((base + sp + 20) * nature_mult(nature, stat))


def estimated_stats(mon: PokemonView) -> dict:
    """Exact stats for our own Pokemon; an assumed spread for the opponent's."""
    info = species_info(mon.species)
    if mon.stats:
        out = dict(mon.stats)
        out['hp'] = mon.maxhp or (calc_stat(info[0]['hp'], 16, 'hp') if info else 150)
        return out
    if mon.belief is not None:
        # posterior mean over the hidden Stat Points / nature (see ai.inference)
        return mon.belief.mean_stats(mon.species)
    if info is None:
        return {s: 100 for s in STATS}
    base = info[0]
    sp = dict(ASSUMED_SP)
    if base['atk'] >= base['spa']:
        sp['atk'] = ASSUMED_SP_ATTACK
    else:
        sp['spa'] = ASSUMED_SP_ATTACK
    return {s: calc_stat(base[s], sp[s], s) for s in STATS}


def known_ability(mon: PokemonView) -> str | None:
    if mon.ability:
        return mon.ability
    info = species_info(mon.species)
    if info and len(info[2]) == 1:
        return info[2][0]
    return None


def is_grounded(mon: PokemonView, view: BattleView | None = None) -> bool:
    if view is not None and 'gravity' in view.field.pseudo:
        return True
    if mon.item == 'Iron Ball':
        return True
    if 'Flying' in mon.types or known_ability(mon) == 'Levitate' or mon.item == 'Air Balloon':
        return False
    if 'magnetrise' in mon.volatiles or 'telekinesis' in mon.volatiles:
        return False
    return True


def speed_multiplier(mon: PokemonView, view: BattleView, side_id: str, include_scarf: bool = True) -> float:
    """Everything that multiplies the Speed stat (boosts, items, abilities, paralysis, Tailwind)."""
    m = BOOST_TABLE[max(-6, min(6, mon.boosts.get('spe', 0)))]
    ability = known_ability(mon)
    weather = view.field.weather
    if mon.item == 'Choice Scarf' and include_scarf:
        m *= 1.5
    if mon.item == 'Iron Ball':
        m *= 0.5
    if ability == 'Swift Swim' and weather in ('raindance', 'primordialsea'):
        m *= 2
    elif ability == 'Chlorophyll' and weather in ('sunnyday', 'desolateland'):
        m *= 2
    elif ability == 'Sand Rush' and weather == 'sandstorm':
        m *= 2
    elif ability == 'Slush Rush' and weather in ('snowscape', 'snow', 'hail'):
        m *= 2
    elif ability == 'Surge Surfer' and view.field.terrain == 'electricterrain':
        m *= 2
    elif ability == 'Unburden' and mon.item == '' and mon.last_item:
        m *= 2
    elif ability == 'Quick Feet' and mon.status:
        m *= 1.5
    if mon.status == 'par' and ability != 'Quick Feet':
        m *= 0.5
    side = view.sides.get(side_id)
    if side is not None and 'tailwind' in side.conditions:
        m *= 2
    return m


def effective_speed(mon: PokemonView, view: BattleView, side_id: str) -> float:
    if mon.belief is not None and not mon.stats:
        # expected speed under the posterior (incl. a possible Choice Scarf)
        mult = speed_multiplier(mon, view, side_id, include_scarf=False)
        p = mon.belief.p_scarf() if mon.item in (None, 'Choice Scarf') else 0.0
        return estimated_stats(mon)['spe'] * mult * (1 + 0.5 * p)
    return estimated_stats(mon)['spe'] * speed_multiplier(mon, view, side_id)


def outspeeds(a: PokemonView, a_side: str, b: PokemonView, b_side: str, view: BattleView) -> float:
    """Probability that ``a`` moves before ``b`` with equal priority (0..1)."""
    tr = 'trickroom' in view.field.pseudo
    if b.belief is not None and not b.stats and a.belief is None:
        mine = effective_speed(a, view, a_side)
        return 1.0 - b.belief.p_faster(b.species, speed_multiplier(b, view, b_side, include_scarf=False), mine, tr,
                                       scarf_active=b.item in (None, 'Choice Scarf'))
    if a.belief is not None and not a.stats and b.belief is None:
        theirs = effective_speed(b, view, b_side)
        return a.belief.p_faster(a.species, speed_multiplier(a, view, a_side, include_scarf=False), theirs, tr,
                                 scarf_active=a.item in (None, 'Choice Scarf'))
    sa = effective_speed(a, view, a_side)
    sb = effective_speed(b, view, b_side)
    if tr:
        sa, sb = -sa, -sb
    if sa > sb:
        return 1.0
    if sa < sb:
        return 0.0
    return 0.5


def move_type_of(move, attacker: PokemonView, view: BattleView) -> str:
    mtype = move.type
    ability = known_ability(attacker)
    mid = move.id
    if mid == 'weatherball':
        mtype = {'sunnyday': 'Fire', 'desolateland': 'Fire', 'raindance': 'Water', 'primordialsea': 'Water',
                 'sandstorm': 'Rock', 'snowscape': 'Ice', 'snow': 'Ice', 'hail': 'Ice'}.get(view.field.weather, 'Normal')
    elif mid == 'terrainpulse' and is_grounded(attacker, view):
        mtype = {'electricterrain': 'Electric', 'grassyterrain': 'Grass', 'mistyterrain': 'Fairy',
                 'psychicterrain': 'Psychic'}.get(view.field.terrain, 'Normal')
    elif mid == 'ragingbull' and attacker.species.startswith('Tauros-Paldea'):
        mtype = attacker.types[-1] if attacker.types else mtype
    elif mid == 'ivycudgel' and '-' in attacker.species:
        mtype = attacker.types[-1] if attacker.types else mtype
    if mtype == 'Normal' and ability in ATE_ABILITIES and move.category != 'Status':
        mtype = ATE_ABILITIES[ability]
    if ability == 'Normalize' and move.category != 'Status':
        mtype = 'Normal'
    return mtype


def base_power(move, attacker: PokemonView, defender: PokemonView, view: BattleView,
               a_stats: dict, d_stats: dict) -> float:
    bp = move.basePower or 0
    mid = move.id
    if mid in ('lowkick', 'grassknot'):
        w = species_info(defender.species)[3] if species_info(defender.species) else 50
        bp = 20 if w < 10 else 40 if w < 25 else 60 if w < 50 else 80 if w < 100 else 100 if w < 200 else 120
    elif mid in ('heavyslam', 'heatcrash'):
        wa = species_info(attacker.species)[3] if species_info(attacker.species) else 50
        wd = species_info(defender.species)[3] if species_info(defender.species) else 50
        r = wa / max(wd, 0.1)
        bp = 120 if r >= 5 else 100 if r >= 4 else 80 if r >= 3 else 60 if r >= 2 else 40
    elif mid == 'acrobatics':
        bp = 110 if attacker.item == '' else 55
    elif mid == 'facade' and attacker.status in ('brn', 'par', 'psn', 'tox'):
        bp = 140
    elif mid == 'hex' and defender.status:
        bp = 130
    elif mid == 'venoshock' and defender.status in ('psn', 'tox'):
        bp = 130
    elif mid == 'knockoff' and defender.item:
        bp = 97.5
    elif mid in ('eruption', 'waterspout', 'dragonenergy'):
        bp = max(1, 150 * attacker.hp)
    elif mid in ('reversal', 'flail'):
        r = attacker.hp * 48
        bp = 200 if r < 2 else 150 if r < 5 else 100 if r < 10 else 80 if r < 17 else 40 if r < 33 else 20
    elif mid == 'gyroball':
        bp = min(150, 25 * max(1, d_stats['spe']) / max(1, a_stats['spe']) + 1)
    elif mid == 'electroball':
        r = a_stats['spe'] / max(1, d_stats['spe'])
        bp = 150 if r >= 4 else 120 if r >= 3 else 80 if r >= 2 else 60 if r >= 1 else 40
    elif mid in ('storedpower', 'powertrip'):
        bp = 20 + 20 * sum(max(0, v) for v in attacker.boosts.values())
    elif mid == 'weatherball' and view.field.weather:
        bp = 100
    elif mid == 'terrainpulse' and view.field.terrain and is_grounded(attacker, view):
        bp = 100
    elif mid == 'brine' and defender.hp <= 0.5:
        bp = 130
    elif mid == 'risingvoltage' and view.field.terrain == 'electricterrain' and is_grounded(defender, view):
        bp = 140
    elif mid == 'expandingforce' and view.field.terrain == 'psychicterrain' and is_grounded(attacker, view):
        bp = 120
    elif mid == 'payback' or mid == 'avalanche' or mid == 'revenge':
        bp = bp * 1.5  # often doubled; expected value
    elif mid in ('boltbeak', 'fishiousrend'):
        bp = bp * 1.5
    elif mid == 'lastrespects':
        side = view.sides.get(_side_of(attacker, view))
        fainted = sum(1 for p in side.pokemon if p.fainted) if side else 0
        bp = 50 + 50 * fainted
    elif mid == 'ragefist':
        bp = min(350, 50 + 50 * attacker.times_attacked)
    elif mid in ('crushgrip', 'wringout', 'hardpress'):
        bp = max(1, 120 * defender.hp) if mid != 'hardpress' else max(1, 100 * defender.hp)
    elif mid == 'beatup':
        bp = 40
    if not bp and move.category != 'Status':
        bp = 60
    return bp


def _side_of(mon: PokemonView, view: BattleView) -> str:
    for sid, side in view.sides.items():
        if any(p is mon for p in side.pokemon):
            return sid
    return view.me


def expected_hits(move, attacker: PokemonView) -> float:
    mh = move.multihit
    if not mh:
        return 1.0
    if isinstance(mh, (list, tuple)):
        if known_ability(attacker) == 'Skill Link':
            return float(mh[1])
        if mh[0] == 2 and mh[1] == 5:
            return 4.5 if attacker.item == 'Loaded Dice' else 3.1
        return (mh[0] + mh[1]) / 2
    return float(mh)


def damage_parts(attacker: PokemonView, defender: PokemonView, move_name: str, view: BattleView,
                 spread: bool = False, a_stats: dict | None = None, d_stats: dict | None = None):
    """Everything in the damage formula except the raw attacking / defending stats.

    ``damage = floor(floor(22 * bp * (A * atk_mult) / (D * def_mult)) / 50) + 2`` times ``mod`` and the
    random factor (0.85..1), where A / D are the raw stats named ``atk_key`` / ``def_key``.  Keeping the
    stats separate lets :mod:`pokechamp.ai.inference` evaluate many hypothetical opponent spreads at once.
    Returns ``None`` when the move does no regular damage, or a :class:`DamageParts`.
    """
    move = move_info(move_name)
    if move is None or move.category == 'Status':
        return None
    a_side = _side_of(attacker, view)
    d_side = _side_of(defender, view)
    a_stats = a_stats or estimated_stats(attacker)
    d_stats = d_stats or estimated_stats(defender)
    a_ability = known_ability(attacker)
    d_ability = known_ability(defender)
    mold_breaker = a_ability in ('Mold Breaker', 'Teravolt', 'Turboblaze')
    if mold_breaker:
        d_ability = None

    mtype = move_type_of(move, attacker, view)
    # fixed-damage moves
    if move.damage == 'level' or move.id in ('seismictoss', 'nightshade'):
        if type_multiplier(mtype, tuple(defender.types)) == 0:
            return DamageParts(kind='immune')
        return DamageParts(kind='fixed', fixed=attacker.level)
    if move.id in ('superfang', 'ruination', 'naturesmadness'):
        if type_multiplier(mtype, tuple(defender.types)) == 0:
            return DamageParts(kind='immune')
        return DamageParts(kind='half')
    if move.id == 'finalgambit':
        hp = attacker.hp_exact if attacker.hp_exact is not None else attacker.hp * a_stats['hp']
        return DamageParts(kind='fixed', fixed=hp)

    eff = type_multiplier(mtype, tuple(defender.types))
    if move.id == 'freezedry' and 'Water' in defender.types:
        eff *= 4 if eff else 0
    if move.id == 'thousandarrows' and 'Flying' in defender.types:
        eff = type_multiplier('Ground', tuple(t for t in defender.types if t != 'Flying')) or 1.0
    if a_ability == 'Scrappy' and 'Ghost' in defender.types and mtype in ('Normal', 'Fighting'):
        eff = type_multiplier(mtype, tuple(t for t in defender.types if t != 'Ghost')) or 1.0
    if mtype == 'Ground' and not is_grounded(defender, view) and move.id != 'thousandarrows':
        eff = 0.0
    if d_ability and IMMUNITY_ABILITIES.get(d_ability) == mtype:
        eff = 0.0
    if d_ability == 'Wonder Guard' and eff <= 1:
        eff = 0.0
    if d_ability == 'Bulletproof' and move.flags.get('bullet'):
        eff = 0.0
    if d_ability == 'Soundproof' and move.flags.get('sound'):
        eff = 0.0
    if d_ability in ('Good as Gold',) and move.category == 'Status':
        eff = 0.0
    if eff == 0:
        return DamageParts(kind='immune')

    physical = move.category == 'Physical'
    # attacking stat
    atk_from_target = move.overrideOffensivePokemon == 'target'
    atk_mon = defender if atk_from_target else attacker
    atk_key = move.overrideOffensiveStat or ('atk' if physical else 'spa')
    atk = BOOST_TABLE[max(-6, min(6, atk_mon.boosts.get(atk_key, 0)))]
    def_key = move.overrideDefensiveStat or ('def' if physical else 'spd')
    if 'wonderroom' in view.field.pseudo:
        def_key = 'spd' if def_key == 'def' else 'def'
    dfn = 1.0
    if not (move.ignoreDefensive or a_ability == 'Unaware' or move.id in ('chipaway', 'sacredsword', 'darkestlariat')):
        dfn *= BOOST_TABLE[max(-6, min(6, defender.boosts.get(def_key, 0)))]
    if d_ability == 'Unaware':
        atk = 1.0

    # attacker stat modifiers
    if a_ability in ('Huge Power', 'Pure Power') and atk_key == 'atk':
        atk *= 2
    if a_ability == 'Guts' and attacker.status and physical:
        atk *= 1.5
    if a_ability == 'Hustle' and physical:
        atk *= 1.5
    if a_ability == 'Solar Power' and not physical and view.field.weather in ('sunnyday', 'desolateland'):
        atk *= 1.5
    if a_ability == 'Gorilla Tactics' and physical:
        atk *= 1.5
    if attacker.item == 'Choice Band' and physical:
        atk *= 1.5
    if attacker.item == 'Choice Specs' and not physical:
        atk *= 1.5
    # defender stat modifiers
    if defender.item == 'Assault Vest' and not physical:
        dfn *= 1.5
    if defender.item == 'Eviolite' and species_info(defender.species) and species_info(defender.species)[4]:
        dfn *= 1.5
    if d_ability == 'Fur Coat' and physical:
        dfn *= 2
    if view.field.weather == 'sandstorm' and 'Rock' in defender.types and not physical:
        dfn *= 1.5
    if view.field.weather in ('snowscape', 'snow') and 'Ice' in defender.types and physical:
        dfn *= 1.5

    bp = base_power(move, attacker, defender, view, a_stats, d_stats)
    # base power modifiers
    if a_ability == 'Technician' and bp <= 60:
        bp *= 1.5
    if a_ability in ATE_ABILITIES and move.type == 'Normal':
        bp *= 1.2
    if a_ability == 'Tough Claws' and move.flags.get('contact'):
        bp *= 1.3
    if a_ability == 'Strong Jaw' and move.flags.get('bite'):
        bp *= 1.5
    if a_ability == 'Iron Fist' and move.flags.get('punch'):
        bp *= 1.2
    if a_ability == 'Mega Launcher' and move.flags.get('pulse'):
        bp *= 1.5
    if a_ability == 'Sharpness' and move.flags.get('slicing'):
        bp *= 1.5
    if a_ability == 'Punk Rock' and move.flags.get('sound'):
        bp *= 1.3
    if a_ability == 'Reckless' and (move.recoil or move.hasCrashDamage):
        bp *= 1.2
    if a_ability == 'Sheer Force' and move.secondaries:
        bp *= 1.3
    if a_ability == 'Sand Force' and view.field.weather == 'sandstorm' and mtype in ('Rock', 'Ground', 'Steel'):
        bp *= 1.3
    if a_ability in ('Steelworker', 'Steely Spirit') and mtype == 'Steel':
        bp *= 1.5
    if a_ability == 'Transistor' and mtype == 'Electric':
        bp *= 1.3
    if a_ability == "Dragon's Maw" and mtype == 'Dragon':
        bp *= 1.5
    if a_ability == 'Rocky Payload' and mtype == 'Rock':
        bp *= 1.5
    if a_ability == 'Water Bubble' and mtype == 'Water':
        bp *= 2
    if a_ability in ('Overgrow', 'Blaze', 'Torrent', 'Swarm') and attacker.hp <= 1 / 3:
        if {'Overgrow': 'Grass', 'Blaze': 'Fire', 'Torrent': 'Water', 'Swarm': 'Bug'}[a_ability] == mtype:
            bp *= 1.5
    if TYPE_BOOST_ITEMS.get(attacker.item or '') == mtype:
        bp *= 1.2
    if attacker.item == 'Muscle Band' and physical:
        bp *= 1.1
    if attacker.item == 'Wise Glasses' and not physical:
        bp *= 1.1
    if d_ability == 'Thick Fat' and mtype in ('Fire', 'Ice'):
        bp *= 0.5
    if d_ability in ('Heatproof', 'Water Bubble') and mtype == 'Fire':
        bp *= 0.5
    if d_ability == 'Dry Skin' and mtype == 'Fire':
        bp *= 1.25
    terrain = view.field.terrain
    if terrain and is_grounded(attacker, view):
        if (terrain, mtype) in (('electricterrain', 'Electric'), ('grassyterrain', 'Grass'),
                                ('psychicterrain', 'Psychic')):
            bp *= 1.3
    if terrain == 'mistyterrain' and mtype == 'Dragon' and is_grounded(defender, view):
        bp *= 0.5
    if terrain == 'grassyterrain' and move.id in ('earthquake', 'bulldoze', 'magnitude'):
        bp *= 0.5
    if 'charge' in attacker.volatiles and mtype == 'Electric':
        bp *= 2

    mod = 1.0
    if spread:
        mod *= 0.75
    weather = view.field.weather
    if weather in ('sunnyday', 'desolateland'):
        if mtype == 'Fire':
            mod *= 1.5
        elif mtype == 'Water':
            mod *= 0.5 if move.id != 'hydrosteam' else 1.5
    elif weather in ('raindance', 'primordialsea'):
        if mtype == 'Water':
            mod *= 1.5
        elif mtype == 'Fire':
            mod *= 0.5
    # STAB
    stab_types = attacker.types
    if mtype in stab_types or a_ability in ('Protean', 'Libero'):
        mod *= 2.0 if a_ability == 'Adaptability' else 1.5
    mod *= eff
    if eff > 1 and d_ability in ('Filter', 'Solid Rock', 'Prism Armor'):
        mod *= 0.75
    if eff > 1 and attacker.item == 'Expert Belt':
        mod *= 1.2
    if eff < 1 and a_ability == 'Tinted Lens':
        mod *= 2
    if eff > 1 and a_ability == 'Neuroforce':
        mod *= 1.25
    if attacker.status == 'brn' and physical and a_ability != 'Guts' and move.id != 'facade':
        mod *= 0.5
    # screens
    d_conds = view.sides[d_side].conditions
    if move.id not in ('brickbreak', 'psychicfangs', 'ragingbull') and a_ability != 'Infiltrator':
        screen = ('reflect' in d_conds and physical) or ('lightscreen' in d_conds and not physical) or \
                 'auroraveil' in d_conds
        if screen:
            mod *= 0.5 if view.gametype == 'singles' else 2732 / 4096
    if d_ability in ('Multiscale', 'Shadow Shield') and defender.hp >= 0.999:
        mod *= 0.5
    if d_ability == 'Ice Scales' and not physical:
        mod *= 0.5
    if d_ability == 'Fluffy' and move.flags.get('contact') and mtype != 'Fire':
        mod *= 0.5
    if d_ability == 'Fluffy' and mtype == 'Fire':
        mod *= 2
    if d_ability == 'Punk Rock' and move.flags.get('sound'):
        mod *= 0.5
    if attacker.item == 'Life Orb':
        mod *= 1.3
    if eff > 1 and RESIST_BERRIES.get(defender.item or '') == mtype:
        mod *= 0.5
    if a_ability == 'Parental Bond' and not move.multihit:
        mod *= 1.25
    if a_ability == 'Sniper':
        mod *= 1.0 + 0.5 / 24
    hits = expected_hits(move, attacker)
    acc = move.accuracy
    if acc is True or a_ability == 'No Guard' or d_ability == 'No Guard':
        acc_p = 1.0
    else:
        acc_p = (acc / 100.0) * BOOST_TABLE[max(-6, min(6, attacker.boosts.get('accuracy', 0) -
                                                        defender.boosts.get('evasion', 0)))]
        if a_ability == 'Compound Eyes':
            acc_p *= 1.3
        if 'gravity' in view.field.pseudo:
            acc_p *= 5 / 3
        acc_p = min(1.0, acc_p)
    return DamageParts(kind='ohko' if move.ohko else 'normal', bp=bp, atk_key=atk_key, atk_mult=atk,
                       atk_from_target=atk_from_target, def_key=def_key, def_mult=dfn, mod=mod, hits=hits,
                       acc=acc_p, will_crit=bool(move.willCrit),
                       sturdy=(d_ability == 'Sturdy' or defender.item == 'Focus Sash'))


@dataclass
class DamageParts:
    kind: str                     # 'normal' | 'immune' | 'fixed' | 'half' | 'ohko'
    fixed: float = 0.0
    bp: float = 0.0
    atk_key: str = 'atk'
    atk_mult: float = 1.0
    atk_from_target: bool = False
    def_key: str = 'def'
    def_mult: float = 1.0
    mod: float = 1.0
    hits: float = 1.0
    acc: float = 1.0
    will_crit: bool = False
    sturdy: bool = False

    def base(self, atk_stat, def_stat):
        """Base damage for raw stats (scalars or numpy arrays)."""
        a = atk_stat * self.atk_mult
        d = def_stat * self.def_mult
        if isinstance(a, np.ndarray) or isinstance(d, np.ndarray):
            d = np.maximum(1, d)
            return np.floor(np.floor(22 * self.bp * a / d) / 50) + 2
        return math.floor(math.floor(22 * self.bp * a / max(1, d)) / 50) + 2

    def roll_range(self, atk_stat, def_stat, crit: bool = False):
        """(min, max) damage over the random roll, in HP points."""
        base = self.base(atk_stat, def_stat)
        m = self.mod * self.hits * (1.5 if (crit or self.will_crit) else 1.0)
        return base * m * 0.85, base * m


def estimate_damage(attacker: PokemonView, defender: PokemonView, move_name: str, view: BattleView,
                    spread: bool = False) -> tuple[float, float]:
    """Expected damage as a fraction of the defender's max HP, and the chance to KO (0..1).

    Accuracy is folded into the expected damage but not into the KO chance.
    """
    a_stats = estimated_stats(attacker)
    d_stats = estimated_stats(defender)
    parts = damage_parts(attacker, defender, move_name, view, spread, a_stats, d_stats)
    if parts is None or parts.kind == 'immune':
        return 0.0, 0.0
    d_maxhp = max(1, d_stats['hp'])
    d_hp = defender.hp_exact if defender.hp_exact is not None else defender.hp * d_maxhp
    if parts.kind == 'fixed':
        return min(1.5, parts.fixed / d_maxhp), 1.0 if parts.fixed >= d_hp else 0.0
    if parts.kind == 'half':
        return defender.hp / 2, 0.0
    if parts.kind == 'ohko':
        return parts.acc * d_hp / d_maxhp, 0.3
    atk_stats = d_stats if parts.atk_from_target else a_stats
    lo, hi = parts.roll_range(atk_stats[parts.atk_key], d_stats[parts.def_key])
    crit_chance = 1 / 24 if not parts.will_crit else 1.0
    mean = (lo + hi) / 2 * (1 + 0.5 * crit_chance if not parts.will_crit else 1.0)
    if d_hp <= lo:
        ko = 1.0
    elif d_hp > hi:
        ko = 0.0
    else:
        ko = (hi - d_hp) / max(1e-9, hi - lo)
    if defender.hp >= 0.999 and parts.sturdy and parts.hits <= 1:
        ko = 0.0
    return parts.acc * mean / d_maxhp, ko * parts.acc


def likely_moves(mon: PokemonView) -> list[str]:
    """Moves to assume for an opponent: revealed moves, otherwise generic STAB attacks."""
    if mon.moves:
        return list(mon.moves)
    out = []
    info = species_info(mon.species)
    if info is None:
        return out
    base = info[0]
    physical = base['atk'] >= base['spa']
    for t in mon.types[:2]:
        out.append(STAB_PLACEHOLDER[(t, physical)])
    return out


# a typical STAB attack per (type, physical?) used for unrevealed movesets
STAB_PLACEHOLDER = {
    ('Normal', True): 'Body Slam', ('Normal', False): 'Hyper Voice',
    ('Fire', True): 'Flare Blitz', ('Fire', False): 'Flamethrower',
    ('Water', True): 'Liquidation', ('Water', False): 'Surf',
    ('Electric', True): 'Wild Charge', ('Electric', False): 'Thunderbolt',
    ('Grass', True): 'Leaf Blade', ('Grass', False): 'Energy Ball',
    ('Ice', True): 'Icicle Crash', ('Ice', False): 'Ice Beam',
    ('Fighting', True): 'Close Combat', ('Fighting', False): 'Aura Sphere',
    ('Poison', True): 'Poison Jab', ('Poison', False): 'Sludge Bomb',
    ('Ground', True): 'Earthquake', ('Ground', False): 'Earth Power',
    ('Flying', True): 'Brave Bird', ('Flying', False): 'Air Slash',
    ('Psychic', True): 'Zen Headbutt', ('Psychic', False): 'Psychic',
    ('Bug', True): 'X-Scissor', ('Bug', False): 'Bug Buzz',
    ('Rock', True): 'Stone Edge', ('Rock', False): 'Power Gem',
    ('Ghost', True): 'Shadow Claw', ('Ghost', False): 'Shadow Ball',
    ('Dragon', True): 'Dragon Claw', ('Dragon', False): 'Dragon Pulse',
    ('Dark', True): 'Crunch', ('Dark', False): 'Dark Pulse',
    ('Steel', True): 'Iron Head', ('Steel', False): 'Flash Cannon',
    ('Fairy', True): 'Play Rough', ('Fairy', False): 'Moonblast',
    ('Stellar', True): 'Body Slam', ('Stellar', False): 'Hyper Voice',
    ('???', True): 'Body Slam', ('???', False): 'Hyper Voice',
}


def best_damage(attacker: PokemonView, defender: PokemonView, view: BattleView,
                moves: list[str] | None = None) -> tuple[float, float, str | None]:
    """Highest expected damage (fraction), its KO chance and the move name."""
    best = (0.0, 0.0, None)
    for m in (moves if moves is not None else (list(attacker.moves) or likely_moves(attacker))):
        if m in attacker.disabled_moves:
            continue
        d, ko = estimate_damage(attacker, defender, m, view)
        if (d, ko) > best[:2]:
            best = (d, ko, m)
    return best


def move_priority(move_name: str, attacker: PokemonView, view: BattleView) -> int:
    move = move_info(move_name)
    if move is None:
        return 0
    pr = move.priority or 0
    ability = known_ability(attacker)
    if ability == 'Prankster' and move.category == 'Status':
        pr += 1
    if ability == 'Gale Wings' and move.type == 'Flying' and attacker.hp >= 0.999:
        pr += 1
    if ability == 'Triage' and move.flags.get('heal'):
        pr += 3
    if move.id == 'grassyglide' and view.field.terrain == 'grassyterrain':
        pr += 1
    return pr


def to_move_id(name: str) -> str:
    return to_id(name)
