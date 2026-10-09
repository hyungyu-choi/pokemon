"""Encode a :class:`BattleView` (+ request) into fixed-size numpy arrays for the neural network.

Layout (``N = 12`` Pokemon tokens: our 6 in request order, then the opponent's 6):

* ``ids``      int   [N, 3]            species / item / ability vocabulary ids
* ``mon``      float [N, MON_F]        numeric Pokemon features
* ``move_ids`` int   [N, 4]            move vocabulary ids (unknown opponent moves = UNK)
* ``move``     float [N, 4, MOVE_F]    numeric move features (incl. estimated damage vs. foes)
* ``glob``     float [GLOB_F]          field / side features
* ``mask``     bool  [2, SLOT_ACTIONS] legal actions per active slot (move decisions)

Vocabularies are built from the exported data in a fixed (sorted) order and saved
with every model checkpoint.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..env.actions import N_MOVES, SLOT_ACTIONS, SWITCH_BASE, decode
from ..env.view import BOOST_STATS, BattleView, PokemonView
from ..sim.dex import get_dex
from ..sim.js import to_id
from .damage import estimate_damage, estimated_stats, likely_moves, move_info, outspeeds, species_info

TYPES = ('Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', 'Poison', 'Ground', 'Flying',
         'Psychic', 'Bug', 'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel', 'Fairy')
TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}
STATUS_LIST = ('brn', 'par', 'slp', 'frz', 'psn', 'tox')
VOLATILES = ('substitute', 'confusion', 'leechseed', 'taunt', 'encore', 'torment', 'disable', 'yawn',
             'perish1', 'perish2', 'perish3', 'focusenergy', 'curse', 'attract', 'partiallytrapped',
             'saltcure', 'charge', 'flashfire', 'magnetrise', 'aquaring', 'ingrain', 'healblock',
             'transform', 'typechange', 'abilitychanged', 'destinybond', 'protosynthesis',
             'quarkdrive', 'stockpile', 'lockedmove', 'mustrecharge', 'twoturnmove', 'choicelock',
             'glaiverush', 'saltcure', 'syrupbomb', 'throatchop', 'gastroacid', 'imprison')
VOL_INDEX = {v: i for i, v in enumerate(dict.fromkeys(VOLATILES))}
WEATHERS = ('sunnyday', 'raindance', 'sandstorm', 'snowscape')
TERRAINS = ('electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain')
PSEUDO = ('trickroom', 'gravity', 'magicroom', 'wonderroom')
SIDE_CONDS = ('reflect', 'lightscreen', 'auroraveil', 'tailwind', 'safeguard', 'mist', 'stealthrock',
              'spikes', 'toxicspikes', 'stickyweb')
CATEGORIES = ('Physical', 'Special', 'Status')

N_MONS = 12
MON_F = (1 + 1 + 1 + 1 + 1 + 1 + len(STATUS_LIST) + len(BOOST_STATS) + 6 + 6 + len(TYPES) + 4 + 2 +
         len(VOL_INDEX) + 2 + 2 + 1 + 4 + 3)
MOVE_F = 1 + 1 + 1 + len(CATEGORIES) + len(TYPES) + 1 + 1 + 1 + 1 + 2 + 2 + 1 + 1 + 6
GLOB_F = (len(WEATHERS) + 1 + len(TERRAINS) + 1 + len(PSEUDO) + 1 + 2 * len(SIDE_CONDS) + 2 + 2 + 1 + 1 +
          4 + 4 + 4 + 1)

UNK = 1  # id 0 = padding / none, 1 = unknown

# Version of the observation layout produced by :func:`encode`.  It is stored in every model checkpoint
# (``config['feature_version']``; checkpoints without it are version 1) and checked when loading, so a model
# is never fed features it was not trained on.  Bump it whenever the meaning or size of any array changes.
FEATURE_VERSION = 1


@lru_cache(maxsize=None)
def vocab() -> dict:
    dex = get_dex()
    species = sorted(dex.species.data.keys())
    moves = sorted(dex.move_data.keys())
    items = sorted(dex.item_data.keys())
    abilities = sorted(dex.ability_data.keys())

    def idx(names):
        return {n: i + 2 for i, n in enumerate(names)}
    return {'species': idx(species), 'moves': idx(moves), 'items': idx(items), 'abilities': idx(abilities)}


def vocab_sizes() -> dict:
    v = vocab()
    return {k: len(m) + 2 for k, m in v.items()}


def _vid(table: str, name: str | None) -> int:
    if name is None:
        return UNK
    if name == '':
        return 0
    return _vid_cached(table, name)


@lru_cache(maxsize=None)
def _vid_cached(table: str, name: str) -> int:
    return vocab()[table].get(to_id(name), UNK)


def _stat_vec(mon: PokemonView):
    stats = estimated_stats(mon)
    return [stats[s] / 300.0 for s in ('hp', 'atk', 'def', 'spa', 'spd', 'spe')]


def _base_vec(mon: PokemonView):
    info = species_info(mon.species)
    if info is None:
        return [0.4] * 6
    b = info[0]
    return [b[s] / 200.0 for s in ('hp', 'atk', 'def', 'spa', 'spd', 'spe')]


def encode_mon(mon: PokemonView | None, mine: bool, view: BattleView, out_ids, out_num, i: int):
    if mon is None:
        return
    out_ids[i, 0] = _vid('species', mon.species)
    out_ids[i, 1] = _vid('items', mon.item)
    out_ids[i, 2] = _vid('abilities', mon.ability)
    f = []
    f.append(mon.hp)
    f.append(1.0 if mon.fainted else 0.0)
    f.append(1.0 if mon.active else 0.0)
    f.append(1.0 if mon.revealed else 0.0)
    f.append(1.0 if mine else 0.0)
    f.append(1.0 if mon.brought is not False else 0.0)
    f.extend(1.0 if mon.status == s else 0.0 for s in STATUS_LIST)
    f.extend(mon.boosts.get(s, 0) / 6.0 for s in BOOST_STATS)
    f.extend(_base_vec(mon))
    f.extend(_stat_vec(mon))
    tv = [0.0] * len(TYPES)
    for t in mon.types:
        if t in TYPE_INDEX:
            tv[TYPE_INDEX[t]] = 1.0
    f.extend(tv)
    f.append(1.0 if mon.item is not None else 0.0)
    f.append(1.0 if mon.item == '' else 0.0)
    f.append(1.0 if mon.ability is not None else 0.0)
    f.append(1.0 if mon.last_item else 0.0)
    f.append(1.0 if mon.mega else 0.0)
    f.append(1.0 if mon.can_mega else 0.0)
    vv = [0.0] * len(VOL_INDEX)
    for v in mon.volatiles:
        j = VOL_INDEX.get(v)
        if j is not None:
            vv[j] = 1.0
    f.extend(vv)
    f.append(min(mon.sleep_turns, 3) / 3.0)
    f.append(min(mon.toxic_turns, 8) / 8.0)
    f.append(1.0 if mon.slot == 0 else 0.0)
    f.append(1.0 if mon.slot == 1 else 0.0)
    f.append(min(mon.times_attacked, 6) / 6.0)
    f.append(1.0 if mon.trapped else 0.0)
    f.append(len(mon.moves) / 4.0)
    f.append(1.0 if mon.disabled_moves else 0.0)
    f.append(1.0 if mon.last_move else 0.0)
    # what has been inferred about hidden Stat Points / nature / Choice Scarf (opponent only)
    b = mon.belief
    if b is not None and not mine:
        lo, hi = b.speed_quantiles(mon.species)
        f.extend([b.certainty(), b.p_scarf() if mon.item in (None, 'Choice Scarf') else 0.0, (hi - lo) / 200.0])
    else:
        f.extend([1.0 if mine else 0.0, 0.0, 0.0])
    out_num[i, :len(f)] = f


def encode_moves(mon: PokemonView | None, mine: bool, view: BattleView, targets: list, out_ids, out_num, i: int):
    if mon is None:
        return
    # for our active Pokemon, slot k must match "move k+1" of the request (locked moves shrink the list)
    if mine and mon.active and mon.move_pp:
        names = list(mon.move_pp)[:N_MOVES]
    else:
        names = list(mon.moves)[:N_MOVES]
    for k in range(N_MOVES):
        if k >= len(names):
            out_ids[i, k] = 0 if mine else (UNK if not mon.fainted else 0)
            continue
        name = names[k]
        static = _move_static(name)
        out_ids[i, k] = _vid('moves', name)
        if static is None:
            continue
        head, spread, tail = static
        pp = mon.move_pp.get(name)
        dmg = [0.0, 0.0]
        ko = [0.0, 0.0]
        for t_i, tgt in enumerate(targets[:2]):
            if tgt is None or not tgt.alive or mon.fainted:
                continue
            d, k_ = estimate_damage(mon, tgt, name, view)
            dmg[t_i] = min(d, 1.5)
            ko[t_i] = k_
        f = [*head, pp[0] / max(1, pp[1]) if pp else 1.0, 1.0 if name in mon.disabled_moves else 0.0,
             1.0,  # known
             spread, *dmg, *ko, *tail]
        out_num[i, k, :len(f)] = f


@lru_cache(maxsize=None)
def _move_static(name: str):
    """The parts of a move's features that depend only on the move (cached per name): ``(head, spread,
    tail)`` = (base power, accuracy, priority, category and type one-hots), the spread-target flag and the
    effect flags; ``None`` for an unknown move.  :func:`encode_moves` puts the per-Pokemon values (PP,
    disabled, known, damage and KO chance against each target) between them."""
    move = move_info(name)
    if move is None:
        return None
    head = [min(move.basePower or 0, 250) / 150.0]
    acc = move.accuracy
    head.append(1.0 if acc is True else acc / 100.0)
    head.append((move.priority or 0) / 5.0)
    head.extend(1.0 if move.category == c else 0.0 for c in CATEGORIES)
    tv = [0.0] * len(TYPES)
    if move.type in TYPE_INDEX:
        tv[TYPE_INDEX[move.type]] = 1.0
    head.extend(tv)
    spread = 1.0 if move.target in ('allAdjacentFoes', 'allAdjacent', 'all') else 0.0
    tail = (1.0 if move.flags.get('contact') else 0.0,
            1.0 if (move.self or {}).get('boosts') or move.boosts else 0.0,
            1.0 if move.heal or move.flags.get('heal') else 0.0,
            1.0 if move.status or move.volatileStatus else 0.0,
            1.0 if move.sideCondition else 0.0,
            1.0 if move.selfSwitch else 0.0,
            1.0 if move.drain else 0.0,
            1.0 if move.recoil else 0.0)
    return tuple(head), spread, tail


def encode_global(view: BattleView, out):
    f = []
    fld = view.field
    w = fld.weather
    f.extend(1.0 if w == x or (x == 'snowscape' and w in ('snow', 'hail')) else 0.0 for x in WEATHERS)
    f.append(min(view.turn - fld.weather_turn, 8) / 8.0 if w else 0.0)
    f.extend(1.0 if fld.terrain == x else 0.0 for x in TERRAINS)
    f.append(min(view.turn - fld.terrain_turn, 8) / 8.0 if fld.terrain else 0.0)
    f.extend(1.0 if x in fld.pseudo else 0.0 for x in PSEUDO)
    f.append(min(view.turn - fld.pseudo.get('trickroom', view.turn), 5) / 5.0)
    for side in (view.my_side, view.foe_side):
        for c in SIDE_CONDS:
            layers = side.conditions.get(c, 0)
            f.append(min(layers, 3) / (3.0 if c == 'spikes' else 2.0 if c == 'toxicspikes' else 1.0))
    f.append(1.0 if view.my_side.mega_used else 0.0)
    f.append(1.0 if view.foe_side.mega_used else 0.0)
    f.append(view.my_side.alive_count() / 6.0)
    f.append(view.foe_side.alive_count() / 6.0)
    f.append(min(view.turn, 40) / 40.0)
    f.append(1.0 if view.gametype == 'doubles' else 0.0)
    my_act = [view.my_side.active_at(s) for s in range(2)]
    foe_act = [view.foe_side.active_at(s) for s in range(2)]
    # speed order and foe threat against our actives
    for a in my_act:
        for b in foe_act:
            if a is not None and b is not None and a.alive and b.alive:
                f.append(outspeeds(a, view.me, b, view.foe, view))
            else:
                f.append(0.5)
    for a in foe_act:
        for b in my_act:
            if a is not None and b is not None and a.alive and b.alive:
                best = 0.0
                for m in likely_moves(a):
                    best = max(best, estimate_damage(a, b, m, view)[0])
                f.append(min(best, 1.5))
            else:
                f.append(0.0)
    for a in my_act + foe_act:
        f.append(1.0 if a is not None and a.alive else 0.0)
    f.append(view.my_side.team_size / 6.0)
    out[:len(f)] = f


def team_order(view: BattleView, request: dict | None) -> list:
    """Our Pokemon in request order (switch positions refer to this order)."""
    if request and request.get('side'):
        out = []
        for entry in request['side']['pokemon']:
            name = entry['ident'].split(': ', 1)[1]
            out.append(view.my_side.find(name))
        return out
    return list(view.my_side.pokemon)


def foe_order(view: BattleView) -> list:
    return list(view.foe_side.pokemon)[:6]


def encode(view: BattleView, request: dict | None, legal: list | None = None) -> dict:
    ids = np.zeros((N_MONS, 3), dtype=np.int64)
    mon = np.zeros((N_MONS, MON_F), dtype=np.float32)
    move_ids = np.zeros((N_MONS, N_MOVES), dtype=np.int64)
    move = np.zeros((N_MONS, N_MOVES, MOVE_F), dtype=np.float32)
    glob = np.zeros((GLOB_F,), dtype=np.float32)
    mine = team_order(view, request)[:6]
    foes = foe_order(view)
    my_act = [view.my_side.active_at(s) for s in range(2)]
    foe_act = [view.foe_side.active_at(s) for s in range(2)]
    for i, m in enumerate(mine):
        encode_mon(m, True, view, ids, mon, i)
        encode_moves(m, True, view, foe_act, move_ids, move, i)
    for j, m in enumerate(foes):
        encode_mon(m, False, view, ids, mon, 6 + j)
        encode_moves(m, False, view, my_act, move_ids, move, 6 + j)
    encode_global(view, glob)
    obs = {'ids': ids, 'mon': mon, 'move_ids': move_ids, 'move': move, 'glob': glob}
    obs['active_idx'] = np.array([_index_of(mine, a) for a in my_act], dtype=np.int64)
    return obs


def _index_of(lst, item) -> int:
    for i, x in enumerate(lst):
        if x is item and item is not None:
            return i
    return -1


def legal_mask(legal: list, n_active: int) -> np.ndarray:
    mask = np.zeros((2, SLOT_ACTIONS), dtype=bool)
    for combo in legal:
        for s, a in enumerate(combo):
            if a >= 0:
                mask[s, a] = True
    return mask


def action_move_index(a: int) -> int:
    return decode(a)[1] if a < SWITCH_BASE else -1
