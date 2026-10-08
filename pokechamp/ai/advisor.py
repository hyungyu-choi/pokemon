"""Live battle advisor: given the current situation of a real battle, rank the possible actions.

Input is a JSON description of what the player can see (their own team in full,
what the opponent has revealed, HP, status, boosts and the field; see
``examples/advisor_state.json``).  The advisor

1. builds the player's view and asks the policy network for a prior and a value,
2. samples the opponent's hidden information (unrevealed moves, item, ability,
   Stat Points, which Pokemon they brought) from a prior - by default from the
   evolved team population if one is available,
3. recreates the battle in the simulator for each sample ("determinization"),
4. for every legal action, simulates the turn with the opponent's likely reply and
   plays the battle forward a few turns with the policy, scoring the result with the
   value network (or the outcome if the battle ends),

and reports the actions sorted by estimated win probability.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field

from ..env.actions import PASS, decode, legal_joint_actions, team_choice
from ..env.runner import gen5_seed
from ..env.state import RolloutSession, view_from_battle
from ..env.view import BOOST_STATS, BattleView, PokemonView
from ..sim.battle import Battle
from ..sim.clone import clone_battle
from ..sim.dex import get_dex
from ..sim.js import Obj, to_id
from ..sim.teams import import_team
from ..teambuilder.random_teams import STATS, legal_pool
from .damage import species_info
from .nn_agent import potential

WEATHER_ALIASES = {
    'sun': 'sunnyday', 'sunny': 'sunnyday', 'harshsunlight': 'sunnyday', 'rain': 'raindance', 'sand': 'sandstorm',
    'sandstorm': 'sandstorm', 'snow': 'snowscape', 'hail': 'snowscape', 'snowscape': 'snowscape',
    'sunnyday': 'sunnyday', 'raindance': 'raindance',
}
TERRAIN_ALIASES = {
    'electric': 'electricterrain', 'grassy': 'grassyterrain', 'misty': 'mistyterrain', 'psychic': 'psychicterrain',
}
SUPPORTED_VOLATILES = ('substitute', 'confusion', 'leechseed', 'taunt', 'encore', 'yawn', 'focusenergy',
                       'aquaring', 'ingrain', 'magnetrise', 'perishsong', 'curse', 'saltcure', 'healblock')
GOOD_STATUS = {
    'protect', 'detect', 'swordsdance', 'nastyplot', 'calmmind', 'dragondance', 'quiverdance', 'bulkup',
    'recover', 'roost', 'slackoff', 'shoreup', 'moonlight', 'synthesis', 'willowisp', 'thunderwave', 'toxic',
    'spore', 'sleeppowder', 'yawn', 'stealthrock', 'spikes', 'taunt', 'trickroom', 'tailwind', 'encore',
    'substitute', 'leechseed', 'reflect', 'lightscreen', 'auroraveil', 'haze', 'partingshot', 'followme',
    'ragepowder', 'helpinghand', 'wideguard', 'irondefense', 'shellsmash', 'bellydrum', 'agility',
    'kingsshield', 'spikyshield', 'banefulbunker', 'stickyweb', 'defog', 'rapidspin', 'trick', 'switcheroo',
    'painsplit', 'wish', 'healbell', 'aromatherapy', 'strengthsap', 'glare', 'coil', 'growth',
}
COMMON_ITEMS = {
    'Choice Scarf': 4, 'Choice Band': 3, 'Choice Specs': 3, 'Life Orb': 3, 'Focus Sash': 4, 'Leftovers': 4,
    'Sitrus Berry': 4, 'Assault Vest': 2, 'Lum Berry': 2, 'Expert Belt': 1, 'Rocky Helmet': 2,
    'White Herb': 1, 'Mental Herb': 1, 'Black Glasses': 1, 'Mystic Water': 1, 'Charcoal': 1, 'Light Clay': 1,
    'Shuca Berry': 1, 'Occa Berry': 1, 'Yache Berry': 1, 'Chople Berry': 1, 'Haban Berry': 1,
}


# ---------------------------------------------------------------------------
# Input parsing

@dataclass
class MonState:
    species: str
    hp: float | None = None          # fraction (0..1)
    hp_exact: int | None = None
    status: str = ''
    boosts: dict = field(default_factory=dict)
    item: str | None = None          # None unknown, '' none / consumed
    ability: str | None = None
    moves: list = field(default_factory=list)
    mega: bool = False
    fainted: bool = False
    volatiles: list = field(default_factory=list)
    pp: dict = field(default_factory=dict)
    locked_move: str = ''
    sleep_turns: int = 0
    toxic_turns: int = 0
    fresh: bool = False              # switched in this turn (Fake Out etc. still work)
    substitute_hp: float | None = None
    belief: object = None            # opponent: StatBelief over hidden Stat Points / nature / Scarf
    choices: list = field(default_factory=list)       # opponent: (move used, our Pokemon it faced, view)
    view: object = None              # opponent: its PokemonView (for behaviour-based move inference)
    faster_than: list = field(default_factory=list)   # opponent: our Pokemon it moved before (same priority)
    slower_than: list = field(default_factory=list)   # opponent: our Pokemon that moved before it


@dataclass
class SideState:
    team: list                        # own: full sets; foe: species names (team preview)
    brought: list                     # names of brought Pokemon (own: known; foe: seen so far)
    active: list
    mons: dict                        # species name -> MonState
    conditions: dict = field(default_factory=dict)   # condition id -> turns left (or layers for hazards)
    mega_used: bool = False


@dataclass
class AdvisorInput:
    formatid: str
    turn: int
    me: SideState
    foe: SideState
    weather: str = ''
    weather_turns: int = 0
    terrain: str = ''
    terrain_turns: int = 0
    pseudo: dict = field(default_factory=dict)  # trickroom/gravity/... -> turns left


def _species_name(name: str) -> str:
    sp = get_dex().species.get(name)
    if not sp.exists:
        raise ValueError(f"unknown Pokemon {name!r}")
    return sp.name


def _parse_hp(v, maxhp=None):
    """'64%' / 0.64 / 120 (exact, needs maxhp) / '120/170' -> (fraction, exact or None)."""
    if v is None:
        return None, None
    if isinstance(v, str):
        v = v.strip()
        if v.endswith('%'):
            return max(0.0, min(1.0, float(v[:-1]) / 100)), None
        if '/' in v:
            a, b = v.split('/')
            return int(a) / int(b), int(a)
        v = float(v)
    if isinstance(v, float) and v <= 1.0:
        return v, None
    if maxhp:
        return int(v) / maxhp, int(v)
    return min(1.0, float(v) / 100), None


def _norm_cond(name: str) -> str:
    return to_id(name)


def parse_input(data: dict) -> AdvisorInput:
    dex = get_dex()
    fmt = data.get('format', 'gen9championsbssregmc')
    me_d = data['me']
    foe_d = data['foe']
    team = me_d['team']
    if isinstance(team, str):
        team = import_team(team)
    else:
        team = [Obj({**s, 'evs': Obj({k: (s.get('evs') or {}).get(k, 0) for k in STATS})}) for s in team]
    for s in team:
        s.species = _species_name(s.species)
        s.name = s.get('name') or s.species
        s.level = 50
    my_names = [s.species for s in team]

    def side_mons(d, names, own):
        out = {}
        for raw_name, md in (d.get('pokemon') or {}).items():
            name = _species_name(raw_name)
            base = get_dex().species.get(name)
            # allow describing a Mega by its base name or the Mega forme
            if name not in names and base.baseSpecies in [get_dex().species.get(n).baseSpecies for n in names]:
                name = next(n for n in names if get_dex().species.get(n).baseSpecies == base.baseSpecies)
            ms = MonState(species=name)
            hp, exact = _parse_hp(md.get('hp'))
            ms.hp, ms.hp_exact = hp, exact
            ms.status = to_id(md.get('status') or '')
            ms.boosts = {k: int(v) for k, v in (md.get('boosts') or {}).items() if k in BOOST_STATS}
            if md.get('item') is not None:  # missing / null = unknown
                it = md['item']
                ms.item = '' if (it in ('', 'none', 'consumed', 'knocked off')) else dex.items.get(it).name
            if md.get('ability'):
                ms.ability = dex.abilities.get(md['ability']).name
            ms.moves = [dex.moves.get(m).name for m in md.get('moves') or []]
            ms.mega = bool(md.get('mega')) or '-Mega' in raw_name
            ms.fainted = bool(md.get('fainted')) or (ms.hp == 0)
            ms.volatiles = [to_id(v) for v in md.get('volatiles') or []]
            ms.pp = {dex.moves.get(k).name: int(v) for k, v in (md.get('pp') or {}).items()}
            ms.locked_move = dex.moves.get(md['locked_move']).name if md.get('locked_move') else ''
            ms.sleep_turns = int(md.get('sleep_turns') or 0)
            ms.toxic_turns = int(md.get('toxic_turns') or 0)
            ms.fresh = bool(md.get('fresh'))
            if md.get('substitute_hp') is not None:
                ms.substitute_hp = float(md['substitute_hp'])
            ms.faster_than = [_species_name(n) for n in md.get('faster_than') or []]
            ms.slower_than = [_species_name(n) for n in md.get('slower_than') or []]
            out[name] = ms
        return out

    me_mons = side_mons(me_d, my_names, True)
    foe_team = [_species_name(n) for n in foe_d['team']]
    foe_mons = side_mons(foe_d, foe_team, False)

    def resolve(names, pool):
        out = []
        for n in names:
            n = _species_name(n)
            if n not in pool:
                base = get_dex().species.get(n).baseSpecies
                n = next((p for p in pool if get_dex().species.get(p).baseSpecies == base), n)
            out.append(n)
        return out

    me_active = resolve(me_d['active'], my_names)
    foe_active = resolve(foe_d['active'], foe_team)
    me_brought = resolve(me_d.get('brought') or [], my_names)
    foe_seen = resolve(foe_d.get('brought') or [], foe_team)
    for n in foe_active + list(foe_mons):
        if n not in foe_seen and (n in foe_active or (foe_mons.get(n) and (foe_mons[n].hp is not None or
                                                                           foe_mons[n].fainted))):
            foe_seen.append(n)
    for n in me_active:
        if n not in me_brought:
            me_brought.insert(0, n)

    def conds(d):
        out = {}
        for k, v in (d.get('side') or {}).items():
            out[_norm_cond(k)] = int(v) if v is not True else 1
        return out

    fld = data.get('field') or {}
    weather = WEATHER_ALIASES.get(to_id(fld.get('weather') or ''), to_id(fld.get('weather') or ''))
    terrain = to_id(fld.get('terrain') or '')
    terrain = TERRAIN_ALIASES.get(terrain, terrain)
    if terrain and not terrain.endswith('terrain'):
        terrain += 'terrain'
    pseudo = {}
    for k in ('trickroom', 'gravity', 'magicroom', 'wonderroom'):
        for key in (k, {'trickroom': 'trick_room', 'magicroom': 'magic_room', 'wonderroom': 'wonder_room'}.get(k, k)):
            if fld.get(key):
                pseudo[k] = int(fld[key])
    _speed_hint_beliefs(team, foe_mons, me_mons)
    return AdvisorInput(
        formatid=fmt, turn=int(data.get('turn') or 1),
        me=SideState(team=team, brought=me_brought, active=me_active, mons=me_mons, conditions=conds(me_d),
                     mega_used=bool(me_d.get('mega_used'))),
        foe=SideState(team=foe_team, brought=foe_seen, active=foe_active, mons=foe_mons, conditions=conds(foe_d),
                      mega_used=bool(foe_d.get('mega_used'))),
        weather=weather, weather_turns=int(fld.get('weather_turns') or 0),
        terrain=terrain, terrain_turns=int(fld.get('terrain_turns') or 0), pseudo=pseudo,
    )


def _speed_hint_beliefs(my_team, foe_mons: dict, my_mons: dict | None = None):
    """Turn "it moved before / after my X" hints into a posterior over the opponent's Speed.

    Our Speed is taken in our current forme (Mega Evolved or not) and with a Choice Scarf only while we
    still hold it; boosts/paralysis are assumed neutral at the time of the observation.
    """
    from ..sim.teams import calc_stats
    from .inference import StatBelief
    my_mons = my_mons or {}
    by_name = {s.species: s for s in my_team}

    def my_speed_of(mine):
        ms = my_mons.get(mine.species)
        species = None
        item = to_id(mine.get('item') or '')
        if ms is not None and ms.mega:
            stone = get_dex().items.get(mine.get('item') or '')
            if stone.megaStone and stone.megaStone.get(mine.species):
                species = get_dex().species.get(stone.megaStone[mine.species])
        if ms is not None and ms.item == '':
            item = ''
        spe = calc_stats(mine, species)['spe'] if species is not None else calc_stats(mine)['spe']
        return spe * (1.5 if item == 'choicescarf' else 1.0)

    for ms in foe_mons.values():
        if not (ms.faster_than or ms.slower_than):
            continue
        belief = ms.belief or StatBelief()
        if ms.item is not None:
            belief.set_item(ms.item)
        for names, foe_first in ((ms.faster_than, True), (ms.slower_than, False)):
            for n in names:
                mine = by_name.get(n) or next((s for s in my_team if get_dex().species.get(s.species).baseSpecies ==
                                               get_dex().species.get(n).baseSpecies), None)
                if mine is None:
                    continue
                my_speed = my_speed_of(mine)
                belief.observe_speed(ms.species, 1.0, my_speed, foe_first, False,
                                     scarf_active=ms.item in (None, 'Choice Scarf'))
        ms.belief = belief


# ---------------------------------------------------------------------------
# Opponent set prior

class SetPrior:
    """Samples plausible full sets for opponent Pokemon, consistent with what was revealed.

    If a set library is given (e.g. ``population.json`` from the team evolver, or any JSON list of
    teams), matching sets from it are preferred; otherwise moves are weighted by power / STAB
    and common utility, items by general popularity, and the Stat Point spread follows the
    species' stats.
    """

    def __init__(self, formatid: str, library_path: str | None = None):
        self.formatid = formatid
        self.pool = legal_pool(None, formatid)
        self.dex = get_dex()
        self.library: dict[str, list] = {}
        if library_path and os.path.exists(library_path):
            with open(library_path, encoding='utf-8') as f:
                data = json.load(f)
            teams = data.get('teams', []) + data.get('hall_of_fame', []) if isinstance(data, dict) else data
            for team in teams:
                for s in team:
                    self.library.setdefault(s['species'], []).append(s)

    def _move_weight(self, species, name, physical):
        m = self.dex.moves.get(name)
        if m.category == 'Status':
            return 1.2 if m.id in GOOD_STATUS else 0.08
        bp = m.basePower or 60
        if m.multihit:
            bp *= 3 if isinstance(m.multihit, list) else m.multihit
        acc = 1.0 if m.accuracy is True else m.accuracy / 100
        info = species_info(species)
        stab = 1.5 if info and m.type in info[1] else 1.0
        cat = 1.0 if (m.category == 'Physical') == physical else 0.25
        w = (min(bp, 150) * acc * stab / 100) ** 2 * cat
        if m.flags.get('recharge') or (m.flags.get('charge') and m.id != 'solarbeam'):
            w *= 0.2
        if m.selfdestruct:
            w *= 0.3
        return w + 0.01

    def sample(self, species: str, known: MonState | None, rng: random.Random, used_items: set,
               mega_allowed: bool) -> dict:
        known = known or MonState(species=species)
        if known.choices and known.view is not None:
            # behaviour-based move inference: weight candidate sets by how well they explain the
            # opponent's past choices (it would usually have used a much stronger attack if it had one)
            cands = [self._sample_set(species, known, rng, used_items, mega_allowed) for _ in range(6)]
            weights = [self._choice_likelihood(c, known) for c in cands]
            s = rng.choices(cands, weights)[0]
        else:
            s = self._sample_set(species, known, rng, used_items, mega_allowed)
        belief = known.belief
        if belief is not None and belief.observations > 0:
            # nature / Stat Points / Choice Scarf from the posterior given what the battle has shown
            nature, evs, scarf = belief.sample(rng)
            s['nature'], s['evs'] = nature, evs
            if known.item is None:
                if scarf and 'Choice Scarf' not in used_items:
                    s['item'] = 'Choice Scarf'
                elif not scarf and s['item'] == 'Choice Scarf':
                    s['item'] = self._sample_item(species, rng, used_items | {'Choice Scarf'}, mega_allowed,
                                                  known.mega)
        return s

    def _choice_likelihood(self, cand: dict, known: MonState) -> float:
        from .damage import estimate_damage
        lik = 1.0
        for move, target, view in known.choices[-6:]:
            chosen = self.dex.moves.get(move)
            if not chosen.exists:
                continue
            d_chosen = estimate_damage(known.view, target, move, view)[0] if chosen.category != 'Status' else 0.0
            best = 0.0
            for m in cand['moves']:
                if m == move or m in known.moves:
                    continue
                if self.dex.moves.get(m).category == 'Status':
                    continue
                best = max(best, estimate_damage(known.view, target, m, view)[0])
            if best >= 0.3 and best >= 1.5 * d_chosen + 0.15:
                lik *= 0.35
        return lik

    def _sample_set(self, species: str, known: MonState, rng: random.Random, used_items: set,
                    mega_allowed: bool) -> dict:
        lib = [s for s in self.library.get(species, []) if self._consistent(s, known, used_items)]
        if lib and rng.random() < 0.8:
            s = json.loads(json.dumps(rng.choice(lib)))
            s['moves'] = self._fill_moves(species, list(dict.fromkeys(known.moves + s['moves']))[:4], rng,
                                          self._is_physical(s))
            if known.item is not None:
                s['item'] = known.item
            if known.ability:
                s['ability'] = known.ability
            return s
        info = species_info(species)
        base = info[0] if info else {k: 80 for k in STATS}
        physical = base['atk'] >= base['spa']
        if known.moves:
            cats = [self.dex.moves.get(m).category for m in known.moves]
            if 'Physical' in cats and 'Special' not in cats:
                physical = True
            elif 'Special' in cats and 'Physical' not in cats:
                physical = False
        fast = base['spe'] >= 80
        r = rng.random()
        if fast and r < 0.6:
            nature = rng.choice(['Jolly', 'Adamant'] if physical else ['Timid', 'Modest'])
            evs = {'hp': 2, 'atk': 32 if physical else 0, 'def': 0, 'spa': 0 if physical else 32, 'spd': 0, 'spe': 32}
        elif r < 0.8:
            nature = 'Adamant' if physical else 'Modest'
            evs = {'hp': 32, 'atk': 32 if physical else 0, 'def': 1, 'spa': 0 if physical else 32, 'spd': 1, 'spe': 0}
        else:
            nature = rng.choice(['Impish', 'Careful'] if physical else ['Bold', 'Calm'])
            evs = {'hp': 32, 'atk': 0, 'def': 17, 'spa': 0, 'spd': 17, 'spe': 0}
        abilities = self.pool.abilities.get(species) or ['']
        ability = known.ability or rng.choice(abilities)
        if known.item is not None:
            item = known.item
        else:
            item = self._sample_item(species, rng, used_items, mega_allowed, known.mega)
        moves = self._fill_moves(species, list(known.moves)[:4], rng, physical)
        return {'species': species, 'name': species, 'item': item, 'ability': ability, 'moves': moves,
                'nature': nature, 'evs': evs, 'level': 50}

    def _is_physical(self, s):
        return (s.get('evs') or {}).get('atk', 0) >= (s.get('evs') or {}).get('spa', 0)

    def _consistent(self, s, known: MonState, used_items: set) -> bool:
        if known.ability and s['ability'] != known.ability:
            return False
        if known.item is not None and known.item != '' and s['item'] != known.item:
            return False
        if known.item is None and s['item'] in used_items:
            return False
        return True

    def _sample_item(self, species, rng, used_items, mega_allowed, is_mega):
        stones = [x for x in self.pool.mega_stones.get(species, []) if x not in used_items]
        if stones and (is_mega or (mega_allowed and rng.random() < 0.6)):
            return rng.choice(stones)
        banned = self.pool.meta['excluded_items'].get(species, set())
        options = [i for i in self.pool.items if i not in used_items and i not in banned]
        if not options:
            return ''
        weights = [COMMON_ITEMS.get(i, 0.15) for i in options]
        return rng.choices(options, weights)[0]

    def _fill_moves(self, species, moves, rng, physical):
        pool = [m for m in self.pool.moves.get(species, []) if m not in moves]
        while len(moves) < 4 and pool:
            weights = [self._move_weight(species, m, physical) for m in pool]
            pick = rng.choices(pool, weights)[0]
            moves.append(pick)
            pool.remove(pick)
        return moves


# ---------------------------------------------------------------------------
# Rebuilding the battle in the simulator

def _apply_mon_state(battle: Battle, pokemon, ms: MonState | None, is_me: bool):
    if ms is None:
        return
    if ms.mega and pokemon.canMegaEvo and not pokemon.species.isMega:
        battle.actions.runMegaEvo(pokemon)
    if ms.hp_exact is not None:
        pokemon.hp = max(0, min(pokemon.maxhp, ms.hp_exact))
    elif ms.hp is not None:
        pokemon.hp = max(1, min(pokemon.maxhp, round(ms.hp * pokemon.maxhp)))
    if ms.status and ms.status not in ('fnt',):
        if pokemon.status != ms.status:
            if not pokemon.setStatus(ms.status, ignoreImmunities=True) or pokemon.status != ms.status:
                pokemon.status = ms.status
                pokemon.statusState = battle.initEffectState(Obj(id=ms.status, target=pokemon))
        if ms.status == 'slp':
            start = pokemon.statusState.get('startTime') or 3
            if start <= ms.sleep_turns:
                start = ms.sleep_turns + 1
                pokemon.statusState.startTime = start
            pokemon.statusState.time = max(1, start - ms.sleep_turns)
        if ms.status == 'tox':
            pokemon.statusState.stage = ms.toxic_turns
    elif not ms.status and pokemon.status and pokemon.status != 'fnt':
        pokemon.status = ''
        pokemon.statusState = battle.initEffectState(Obj(id='', target=pokemon))
    # boosts are exactly as described (also undoes e.g. Intimidate from the rebuilt battle's start)
    for k in BOOST_STATS:
        pokemon.boosts[k] = max(-6, min(6, int(ms.boosts.get(k, 0))))
    if ms.item is not None:
        item = to_id(ms.item)
        if pokemon.item != item:
            pokemon.item = item
            pokemon.itemState = battle.initEffectState(Obj(id=item, target=pokemon))
    if ms.pp:
        for slot in pokemon.moveSlots:
            name = battle.dex.moves.get(slot.id).name
            if name in ms.pp:
                slot.pp = max(0, min(slot.maxpp, ms.pp[name]))
    for v in ms.volatiles:
        if v in SUPPORTED_VOLATILES and v not in pokemon.volatiles:
            src = pokemon.side.foe.active[0] if pokemon.side.foe.active else pokemon
            try:
                pokemon.addVolatile(v, src)
            except Exception:  # some volatiles need move context; ignore what cannot be recreated
                pass
            if v == 'substitute' and 'substitute' in pokemon.volatiles:
                frac = ms.substitute_hp if ms.substitute_hp is not None else 0.25
                pokemon.volatiles['substitute'].hp = max(1, math.floor(pokemon.maxhp * frac))
    if ms.locked_move and pokemon.getItem().isChoice and 'choicelock' not in pokemon.volatiles:
        # the condition's onStart needs a running move; set its state directly
        pokemon.volatiles['choicelock'] = battle.initEffectState(
            Obj(id='choicelock', target=pokemon, move=to_id(ms.locked_move)))
    if pokemon.isActive and not ms.fresh:
        pokemon.activeTurns = max(pokemon.activeTurns, 1)
        pokemon.activeMoveActions = max(pokemon.activeMoveActions, 1)


def _faint(pokemon):
    pokemon.hp = 0
    pokemon.fainted = True
    pokemon.status = 'fnt'
    pokemon.switchFlag = False
    if pokemon.side.pokemonLeft:
        pokemon.side.pokemonLeft -= 1


def build_battle(inp: AdvisorInput, foe_sets: list, foe_brought: list, seed) -> Battle:
    """Recreate the described situation in the simulator (p1 = us, p2 = opponent)."""
    my_sets = inp.me.team
    battle = Battle(inp.formatid, seed=seed, p1={'name': 'me', 'team': my_sets},
                    p2={'name': 'foe', 'team': foe_sets})
    picked = battle.ruleTable.pickedTeamSize or 6
    n_active = 2 if battle.gameType == 'doubles' else 1

    def order(names_all, active, brought, mons):
        alive = lambda n: not (mons.get(n) and mons[n].fainted)  # noqa: E731
        lead = [n for n in active][:n_active]
        rest = [n for n in brought if n not in lead]
        # leads must be able to battle; the others keep their order
        if len(lead) < n_active:
            lead += [n for n in rest if alive(n)][:n_active - len(lead)]
            rest = [n for n in rest if n not in lead]
        chosen = (lead + rest)[:picked]
        for n in names_all:
            if len(chosen) >= picked:
                break
            if n not in chosen:
                chosen.append(n)
        return tuple(names_all.index(n) for n in chosen)

    my_names = [s.species for s in my_sets]
    foe_names = [s['species'] for s in foe_sets]
    battle.choose('p1', team_choice(order(my_names, inp.me.active, inp.me.brought, inp.me.mons), len(my_names)))
    battle.choose('p2', team_choice(order(foe_names, inp.foe.active, foe_brought, inp.foe.mons), len(foe_names)))

    for side, st in ((battle.sides[0], inp.me), (battle.sides[1], inp.foe)):
        if st.mega_used:
            mega_mon = next((p for p in side.pokemon if st.mons.get(p.species.name) and
                             st.mons[p.species.name].mega), None)
            if mega_mon is not None and mega_mon.canMegaEvo:
                battle.actions.runMegaEvo(mega_mon)
            for p in side.pokemon:
                p.canMegaEvo = False
        for p in side.pokemon:
            _apply_mon_state(battle, p, st.mons.get(p.baseSpecies.name) or st.mons.get(p.set.species), side is battle.sides[0])
        for p in side.pokemon:
            ms = st.mons.get(p.set.species)
            if ms is not None and ms.fainted and not p.fainted:
                _faint(p)
        # side conditions
        src = side.active[0] if side.active else side.pokemon[0]
        for cond, val in st.conditions.items():
            c = battle.dex.conditions.get(cond)
            if not c.exists:
                continue
            layers = val if cond in ('spikes', 'toxicspikes') else 1
            for _ in range(max(1, layers)):
                side.addSideCondition(cond, src)
            if cond in side.sideConditions and cond not in ('spikes', 'toxicspikes', 'stealthrock', 'stickyweb'):
                side.sideConditions[cond].duration = max(1, val)
    # field
    src = battle.sides[0].active[0]
    battle.field.clearWeather()
    if inp.weather:
        battle.field.setWeather(inp.weather, src)
        if battle.field.weather and inp.weather_turns:
            battle.field.weatherState.duration = inp.weather_turns
    if battle.field.terrain:
        battle.field.clearTerrain()
    if inp.terrain:
        battle.field.setTerrain(inp.terrain, src)
        if battle.field.terrain and inp.terrain_turns:
            battle.field.terrainState.duration = inp.terrain_turns
    for pw, turns in inp.pseudo.items():
        battle.field.addPseudoWeather(pw, src)
        if pw in battle.field.pseudoWeather:
            battle.field.pseudoWeather[pw].duration = max(1, turns)
    battle.turn = max(1, inp.turn)
    _refresh_turn_state(battle)
    battle.makeRequest('move')
    return battle


def _refresh_turn_state(battle: Battle):
    """Recompute disabled moves and trapping like ``Battle.endTurn`` does before a request."""
    for side in battle.sides:
        for pokemon in side.active:
            if not pokemon or pokemon.fainted:
                continue
            pokemon.maybeDisabled = False
            pokemon.maybeLocked = False
            for move_slot in pokemon.moveSlots:
                move_slot.disabled = False
                move_slot.disabledSource = ''
            battle.runEvent('DisableMove', pokemon)
            for move_slot in pokemon.moveSlots:
                base_move = battle.dex.moves.getByID(move_slot.id)
                if base_move.get('onDisableMove') is not None:
                    battle.singleEvent('DisableMove', battle.dex.getActiveMove(move_slot.id), None, pokemon)
            pokemon.trapped = pokemon.maybeTrapped = False
            battle.runEvent('TrapPokemon', pokemon)


# ---------------------------------------------------------------------------
# The player's (honest) view

def input_to_view(inp: AdvisorInput, battle: Battle | None = None) -> BattleView:
    """Our view of the situation: from a rebuilt battle (own side exact) with opponent info hidden."""
    if battle is None:
        raise ValueError('need a rebuilt battle')
    return mask_view(view_from_battle(battle, 'p1'), inp, battle)


def mask_view(view: BattleView, inp: AdvisorInput, battle: Battle) -> BattleView:
    """Hide what the player cannot know in a full-information view of a (rebuilt / simulated) battle.

    The opponent side is listed like in a real battle: all six team-preview species, unseen ones
    only by species; seen ones with percentage HP and only the revealed moves / item / ability, plus
    the inferred Stat Point belief.  This keeps search positions in the same form as the positions
    the network was trained on.
    """
    foe = view.foe_side
    shown = []
    for name in inp.foe.team:
        mon = next((p for p in foe.pokemon if p.base_species == name or p.forme == name or p.species == name), None)
        ms = inp.foe.mons.get(name)
        seen = mon is not None and (name in inp.foe.brought or mon.active or mon.fainted or mon.hp < 0.999)
        if not seen:
            # unseen: only the species from team preview is known
            mon = PokemonView(species=name, base_species=name, forme=name)
            info = get_dex().species.get(name)
            mon.types = list(info.types)
            mon.revealed = False
        else:
            mon.revealed = True
            mon.stats = None
            mon.hp_exact = None
            mon.maxhp = None
            used = [mon.last_move] if mon.last_move else []
            known_moves = list(dict.fromkeys((ms.moves if ms else []) + used))
            mon.moves = {m: 0 for m in known_moves}
            mon.move_pp = {}
            if ms is None or ms.item is None:
                mon.item = '' if (mon.item == '' and mon.last_item) else None
            if ms is None or ms.ability is None:
                mon.ability = None
        mon.belief = ms.belief if ms else None
        shown.append(mon)
    foe.pokemon = shown
    foe.team_size = battle.ruleTable.pickedTeamSize or 6
    return view


# ---------------------------------------------------------------------------
# Advisor

@dataclass
class Recommendation:
    option: tuple
    label: str
    win_rate: float
    stderr: float
    prior: float
    samples: int
    meaning: tuple | None = None      # per slot: ('move', move name) / ('switch', species) / None


def option_meaning(battle: Battle, sid: str, option) -> tuple:
    req = battle.getSide(sid).activeRequest
    out = []
    for slot, a in enumerate(option):
        if a == PASS:
            out.append(None)
            continue
        kind = decode(a)
        if kind[0] == 'switch':
            out.append(('switch', req['side']['pokemon'][kind[1]]['details'].split(',')[0]))
        else:
            out.append(('move', req['active'][slot]['moves'][kind[1]]['move']))
    return tuple(out)


def describe_option(battle: Battle, sid: str, option) -> str:
    side = battle.getSide(sid)
    req = side.activeRequest
    parts = []
    for slot, a in enumerate(option):
        if a == PASS:
            continue
        kind = decode(a)
        me = side.active[slot] if slot < len(side.active) else None
        who = me.species.name if me else f'slot {slot + 1}'
        if kind[0] == 'switch':
            parts.append(f"{who}: switch to {req['side']['pokemon'][kind[1]]['details'].split(',')[0]}")
            continue
        _, i, target, mega = kind
        move = req['active'][slot]['moves'][i]['move']
        s = f"{who}: {'Mega Evolve + ' if mega else ''}{move}"
        if target in (1, 2):
            foe = side.foe.active[target - 1] if target - 1 < len(side.foe.active) else None
            s += f" -> {foe.species.name if foe else 'foe ' + str(target)}"
        elif target == 3:
            ally = side.active[slot ^ 1] if len(side.active) > 1 else None
            s += f" -> ally {ally.species.name if ally else ''}"
        parts.append(s)
    return ' / '.join(parts) or 'pass'


class Advisor:
    def __init__(self, model_path: str | None = None, library: str | None = None, seed: int = 0,
                 threads: int | None = None):
        self.model = None
        if model_path:
            import torch
            from .model import load_model
            torch.set_num_threads(threads or max(1, os.cpu_count() or 1))
            self.model = load_model(model_path)
        self.library = library
        self.rng = random.Random(seed)
        self._inp = None
        self.last_info = {}
        self.cancel_check = None   # optional callable() -> bool, polled between samples

    def _policy(self, sample: bool):
        if self.model is not None:
            from .nn_agent import NNAgent
            agent = NNAgent(self.model, sample=sample)
            agent.seed(self.rng.randrange(1 << 30))
            return agent
        from .heuristic import HeuristicAgent
        return HeuristicAgent(self.rng.randrange(1 << 30), randomness=0.1 if sample else 0.0)

    def _evaluate_leaf(self, session: RolloutSession) -> float:
        """Win probability for p1 of a (possibly unfinished) battle."""
        if session.ended:
            w = session.winner
            return 1.0 if w == 'p1' else 0.0 if w == 'p2' else 0.5
        view = session.view('p1')
        if self.model is not None:
            req = session.request('p1')
            if req and not req.get('wait'):
                from .nn_agent import NNAgent
                legal = session.legal('p1')
                if legal:
                    _, value, _ = NNAgent(self.model).evaluate(view, req, legal)
                    return 0.5 * (value + 1)
        return 0.5 * (math.tanh(2.0 * potential(view)) + 1)

    def _reply_distribution(self, battle: Battle, top_k: int):
        """The opponent's most likely replies at the root with their (renormalised) probabilities."""
        session = RolloutSession(battle)
        if 'p2' not in session.pending():
            return [(None, 1.0)]
        legal = session.legal('p2')
        if self.model is not None:
            from .nn_agent import NNAgent
            probs, _, _ = NNAgent(self.model).evaluate(session.view('p2'), session.request('p2'), legal)
            probs = [float(p) for p in probs]
        else:
            from .heuristic import HeuristicAgent
            pick = HeuristicAgent(0).choose(session, 'p2', session.view('p2'), legal)
            probs = [0.7 + 0.3 / len(legal) if o == pick else 0.3 / len(legal) for o in legal]
        ranked = sorted(zip(legal, probs), key=lambda x: -x[1])[:top_k]
        total = sum(p for _, p in ranked) or 1.0
        return [(o, p / total) for o, p in ranked]

    def _cancelled(self) -> bool:
        return bool(self.cancel_check and self.cancel_check())

    def _evaluate_leaves(self, sessions: list) -> list[float]:
        """Win probabilities for p1 of many leaf positions; one batched network call."""
        out = [None] * len(sessions)
        batch, idx = [], []
        for i, session in enumerate(sessions):
            if session.ended or self.model is None:
                out[i] = self._evaluate_leaf(session)
                continue
            req = session.request('p1')
            if req is not None and req.get('wait'):
                req = None
            from .features import encode
            batch.append(encode(session.view('p1'), req))
            idx.append(i)
        if batch:
            import torch
            from .model import collate_obs
            with torch.no_grad():
                _logits, values, _x = self.model(collate_obs(batch))
            for i, v in zip(idx, values.tolist()):
                out[i] = 0.5 * (v + 1)
        return out

    def _run_batch(self, battle: Battle, jobs: list, depth: int) -> list:
        """Play many (our option, opponent reply) pairs in lockstep; every network call is batched.

        Returns one win probability per job (nan when the engine rejected our option).
        """
        import numpy as np
        from .nn_agent import batch_policy
        inp = self._inp
        rng = np.random.default_rng(self.rng.randrange(1 << 30))
        sessions, starts, results = [], [], [None] * len(jobs)
        for i, (opt, reply) in enumerate(jobs):
            session = RolloutSession(clone_battle(battle))
            if inp is not None:
                session.view_fns['p1'] = lambda bb, inp=inp: mask_view(view_from_battle(bb, 'p1'), inp, bb)
            if not session.apply('p1', opt):
                results[i] = float('nan')
                sessions.append(None)
                starts.append(0)
                continue
            if reply is not None and 'p2' in session.pending():
                if not session.apply('p2', reply):
                    session.apply_default('p2')
            sessions.append(session)
            starts.append(session.battle.turn)
        for _ in range(8 * max(1, depth) + 8):
            todo = []
            for i, session in enumerate(sessions):
                if session is None or session.ended or session.battle.turn >= starts[i] + depth:
                    continue
                for sid in session.pending():
                    todo.append((i, sid))
            if not todo:
                break
            items = []
            for i, sid in todo:
                session = sessions[i]
                req = session.request(sid)
                items.append((session.view(sid), req, session.legal(sid)))
            probs_list, _values = batch_policy(self.model, items)
            choices = {}
            for (i, sid), (_v, _r, legal), probs in zip(todo, items, probs_list):
                if sid == 'p1':
                    k = int(np.argmax(probs))                       # our policy: greedy
                else:
                    p = probs / probs.sum()
                    k = int(rng.choice(len(legal), p=p))            # opponent: sampled
                choices.setdefault(i, []).append((sid, legal[k]))
            for i, picks in choices.items():
                for sid, option in picks:
                    if not sessions[i].apply(sid, option):
                        sessions[i].apply_default(sid)
        leaf_idx = [i for i, s in enumerate(sessions) if s is not None]
        values = self._evaluate_leaves([sessions[i] for i in leaf_idx])
        for i, v in zip(leaf_idx, values):
            results[i] = v
        return results

    def _rollout(self, battle: Battle, my_option, depth: int, foe_option=None, sample_reply=True,
                 return_session: bool = False):
        b = clone_battle(battle)
        session = RolloutSession(b)
        if self._inp is not None:
            inp = self._inp
            session.view_fns['p1'] = lambda bb: mask_view(view_from_battle(bb, 'p1'), inp, bb)
        me_policy = self._policy(sample=False)
        foe_policy = self._policy(sample=True)
        # first decision: our candidate vs the opponent's reply (given, or sampled from its policy)
        if foe_option is None and sample_reply:
            foe_legal = session.legal('p2') if 'p2' in session.pending() else None
            foe_option = foe_policy.choose(session, 'p2', session.view('p2'), foe_legal) if foe_legal else None
        if not session.apply('p1', my_option):
            return float('nan')
        if foe_option is not None and 'p2' in session.pending():
            if not session.apply('p2', foe_option):
                session.apply_default('p2')
        start_turn = b.turn
        guard = 0
        while not b.ended and b.turn < start_turn + depth and guard < 200:
            guard += 1
            pending = session.pending()
            if not pending:
                break
            choices = {}
            for sid in pending:
                agent = me_policy if sid == 'p1' else foe_policy
                choices[sid] = agent.choose(session, sid, session.view(sid), session.legal(sid))
            for sid, opt in choices.items():
                if not session.apply(sid, opt):
                    session.apply_default(sid)
        if return_session:
            return session
        return self._evaluate_leaf(session)

    def recommend(self, inp: AdvisorInput, determinizations: int = 12, depth: int = 3,
                  log=None, reply_k: int = 4) -> list[Recommendation]:
        """Rank our legal actions.  ``reply_k`` > 0 enumerates the opponent's ``reply_k`` most likely
        replies (expectimax); 0 samples one reply per rollout."""
        prior_sets = SetPrior(inp.formatid, self.library)
        self._inp = inp
        dex = get_dex()
        if log:
            for name, ms in inp.foe.mons.items():
                if ms.belief is not None and ms.belief.observations:
                    log(f"[advisor] inferred {name}: {ms.belief.most_likely(name)}")
        sums: dict = {}
        counts: dict = {}
        sq: dict = {}
        labels: dict = {}
        meanings: dict = {}
        priors: dict = {}
        picked = dex.formats.get(inp.formatid).ruleTable.pickedTeamSize or 6
        # extra results for user interfaces: value-network estimate and the opponent's likely replies
        self.last_info = {'value': None, 'foe_replies': {}, 'samples': 0}
        for d in range(determinizations):
            rng = random.Random(self.rng.randrange(1 << 30))
            # opponent team: sample hidden info for all 6
            used_items = {ms.item for ms in inp.foe.mons.values() if ms.item}
            foe_sets = []
            for name in inp.foe.team:
                s = prior_sets.sample(name, inp.foe.mons.get(name), rng, used_items,
                                      mega_allowed=not inp.foe.mega_used)
                if s['item']:
                    used_items.add(s['item'])
                foe_sets.append(s)
            # opponent's brought Pokemon: the ones seen + random others
            brought = list(inp.foe.brought)
            others = [n for n in inp.foe.team if n not in brought]
            rng.shuffle(others)
            brought = (brought + others)[:picked]
            try:
                battle = build_battle(inp, foe_sets, brought, gen5_seed(rng))
            except Exception as exc:  # report and skip a sample that could not be rebuilt
                if log:
                    log(f"[advisor] could not rebuild sample {d}: {exc}")
                continue
            req = battle.getSide('p1').activeRequest
            n_active = 2 if battle.gameType == 'doubles' else 1
            legal = legal_joint_actions(req, n_active)
            if d == 0 or not priors:
                view = input_to_view(inp, battle)
                if self.model is not None:
                    from .nn_agent import NNAgent
                    probs, value, _ = NNAgent(self.model).evaluate(view, req, legal)
                    for opt, p in zip(legal, probs):
                        priors[opt] = float(p)
                    self.last_info['value'] = 0.5 * (value + 1)
                    if log:
                        log(f"[advisor] value network: win probability {0.5 * (value + 1) * 100:.1f}%")
                else:
                    from .heuristic import HeuristicAgent
                    pick = HeuristicAgent(0).choose(RolloutSession(battle), 'p1', view, legal)
                    for opt in legal:
                        priors[opt] = 1.0 if opt == pick else 0.0
            replies = self._reply_distribution(battle, reply_k) if reply_k else None
            self.last_info['samples'] += 1
            for reply, pr in (replies or []):
                if reply is not None:
                    lab = describe_option(battle, 'p2', reply)
                    self.last_info['foe_replies'][lab] = self.last_info['foe_replies'].get(lab, 0.0) + pr
            if self._cancelled():
                break
            # play every (our option, opponent reply) pair to its leaf, then score all leaves in one batch
            jobs = []
            for opt in legal:
                labels.setdefault(opt, describe_option(battle, 'p1', opt))
                meanings.setdefault(opt, option_meaning(battle, 'p1', opt))
            if self.model is not None and replies is not None:
                # all rollouts of this sample advance together with batched network calls
                pairs = [(opt, reply, pr) for opt in legal for reply, pr in replies]
                values = self._run_batch(battle, [(o, r) for o, r, _ in pairs], depth)
                jobs = [(o, pr, v) for (o, _r, pr), v in zip(pairs, values)]
            else:
                for opt in legal:
                    pairs = [(None, 1.0)] if replies is None else replies
                    for reply, pr in pairs:
                        res = self._rollout(battle, opt, depth, foe_option=reply, sample_reply=replies is None,
                                            return_session=True)
                        jobs.append((opt, pr, res))
            leaf_sessions = [res for _, _, res in jobs if not isinstance(res, float)]
            leaf_values = iter(self._evaluate_leaves(leaf_sessions))
            per_opt = {}
            for opt, pr, res in jobs:
                x = res if isinstance(res, float) else next(leaf_values)
                if x == x:  # nan = choice rejected by the engine
                    acc = per_opt.setdefault(opt, [0.0, 0.0])
                    acc[0] += pr * x
                    acc[1] += pr
            for opt in legal:
                if opt not in per_opt or not per_opt[opt][1]:
                    continue
                # expectimax over the opponent's likely replies (much lower variance than sampling one)
                v = per_opt[opt][0] / per_opt[opt][1]
                sums[opt] = sums.get(opt, 0.0) + v
                sq[opt] = sq.get(opt, 0.0) + v * v
                counts[opt] = counts.get(opt, 0) + 1
            if log:
                best = max(sums, key=lambda o: sums[o] / counts[o]) if sums else None
                if best is not None:
                    log(f"[advisor] sample {d + 1}/{determinizations}: best so far {labels[best]} "
                        f"({sums[best] / counts[best] * 100:.1f}%)")
        out = []
        for opt, n in counts.items():
            mean = sums[opt] / n
            var = max(0.0, sq[opt] / n - mean * mean)
            out.append(Recommendation(opt, labels[opt], mean, math.sqrt(var / max(1, n)), priors.get(opt, 0.0), n,
                                      meanings.get(opt)))
        out.sort(key=lambda r: (-r.win_rate, -r.prior))
        return out


def format_recommendations(recs: list[Recommendation], top: int = 10) -> str:
    lines = [f"{'#':>2}  {'win%':>6}  {'±':>4}  {'policy':>6}  action"]
    for i, r in enumerate(recs[:top]):
        lines.append(f"{i + 1:>2}  {r.win_rate * 100:6.1f}  {r.stderr * 100:4.1f}  {r.prior * 100:5.1f}%  {r.label}")
    return '\n'.join(lines)


__all__ = ['Advisor', 'AdvisorInput', 'SetPrior', 'build_battle', 'format_recommendations', 'parse_input']
