"""Player-perspective battle state.

A :class:`BattleView` holds exactly what one player can know during a battle:

* their own team in full (from the battle request: exact HP, stats, moves, PP, item, ability),
* everything revealed about the opponent (team preview species, HP percentage,
  status, boosts, moves / item / ability once they have been shown),
* the field (weather, terrain, rooms, screens, hazards) and the turn number.

The same structure is used by the AI during self-play training (built
incrementally from the battle protocol by :class:`LogTracker`) and by the live
advisor, where it is filled in from a human-entered description of a battle in
the real game.  That keeps training and live use consistent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..sim.dex import get_dex
from ..sim.js import to_id

BOOST_STATS = ('atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion')
STATUSES = ('brn', 'par', 'slp', 'frz', 'psn', 'tox')


@dataclass
class PokemonView:
    species: str                       # current species / forme, e.g. "Charizard-Mega-Y"
    base_species: str = ''             # species before Mega Evolution / forme changes
    name: str = ''                     # nickname used in the protocol
    level: int = 50
    gender: str = ''
    hp: float = 1.0                    # fraction of max HP (0..1)
    hp_exact: int | None = None        # own Pokemon only
    maxhp: int | None = None           # own Pokemon only
    status: str = ''                   # '', brn, par, slp, frz, psn, tox
    fainted: bool = False
    active: bool = False
    slot: int = -1                     # active position (0/1 in doubles), -1 when benched
    revealed: bool = False             # has been sent out (or is ours)
    brought: bool | None = None        # picked at team preview (None = unknown)
    boosts: dict = field(default_factory=lambda: {s: 0 for s in BOOST_STATS})
    volatiles: set = field(default_factory=set)
    moves: dict = field(default_factory=dict)        # move name -> PP used so far (opponent: revealed only)
    move_pp: dict = field(default_factory=dict)      # own: move name -> (pp, maxpp)
    disabled_moves: set = field(default_factory=set)
    item: str | None = None            # None = unknown, '' = known to hold nothing
    last_item: str = ''                # consumed / removed item
    ability: str | None = None         # current ability (None = unknown)
    base_ability: str | None = None    # ability it switches back to (None = unknown)
    forme: str = ''                    # species from the details string (permanent forme)
    stats: dict | None = None          # own Pokemon: exact stats (atk..spe)
    mega: bool = False
    can_mega: bool = False
    types: list = field(default_factory=list)
    sleep_turns: int = 0
    toxic_turns: int = 0
    last_move: str = ''
    times_attacked: int = 0
    trapped: bool = False

    @property
    def alive(self) -> bool:
        return not self.fainted and self.hp > 0

    def reset_switch_out(self):
        if 'abilitychanged' in self.volatiles or self.base_ability is not None:
            self.ability = self.base_ability
        self.boosts = {s: 0 for s in BOOST_STATS}
        self.volatiles = set()
        self.active = False
        self.slot = -1
        self.toxic_turns = 0
        self.last_move = ''
        self.disabled_moves = set()
        self.trapped = False
        if self.forme:
            self.species = self.forme  # temporary forme changes / Transform end
        dex_species = get_dex().species.get(self.species)
        if dex_species.exists:
            self.types = list(dex_species.types)

    def reveal_ability(self, ability: str):
        """The Pokemon showed ``ability`` as its own (not copied/swapped this switch-in)."""
        self.ability = ability
        if 'abilitychanged' not in self.volatiles:
            self.base_ability = ability


@dataclass
class SideView:
    sideid: str = 'p1'
    name: str = ''
    pokemon: list = field(default_factory=list)       # list[PokemonView] in team order
    conditions: dict = field(default_factory=dict)    # condition id -> layers (or 1)
    condition_turns: dict = field(default_factory=dict)  # condition id -> turn it started
    team_size: int = 6                                # number brought to battle
    mega_used: bool = False

    def active(self) -> list:
        act = [p for p in self.pokemon if p.active]
        act.sort(key=lambda p: p.slot)
        return act

    def active_at(self, slot: int):
        for p in self.pokemon:
            if p.active and p.slot == slot:
                return p
        return None

    def find(self, name: str):
        for p in self.pokemon:
            if p.name == name:
                return p
        return None

    def alive_count(self) -> int:
        """Pokemon still able to battle (counting unrevealed brought Pokemon)."""
        known_alive = sum(1 for p in self.pokemon if p.revealed and not p.fainted)
        revealed = sum(1 for p in self.pokemon if p.revealed)
        return known_alive + max(0, self.team_size - revealed)


@dataclass
class FieldView:
    weather: str = ''
    weather_turn: int = 0
    terrain: str = ''
    terrain_turn: int = 0
    pseudo: dict = field(default_factory=dict)        # trickroom/gravity/... -> turn started


@dataclass
class BattleView:
    me: str = 'p1'
    gametype: str = 'singles'
    formatid: str = ''
    turn: int = 0
    sides: dict = field(default_factory=dict)         # 'p1'/'p2' -> SideView
    field: FieldView = field(default_factory=FieldView)
    ended: bool = False
    winner: str | None = None

    @property
    def foe(self) -> str:
        return 'p2' if self.me == 'p1' else 'p1'

    @property
    def my_side(self) -> SideView:
        return self.sides[self.me]

    @property
    def foe_side(self) -> SideView:
        return self.sides[self.foe]

    @property
    def n_active(self) -> int:
        return 2 if self.gametype == 'doubles' else 1


# ---------------------------------------------------------------------------
# Protocol parsing

def _species_of(details: str) -> tuple[str, int, str]:
    parts = [p.strip() for p in details.split(',')]
    species = parts[0]
    level = 100
    gender = ''
    for p in parts[1:]:
        if p.startswith('L') and p[1:].isdigit():
            level = int(p[1:])
        elif p in ('M', 'F'):
            gender = p
    return species, level, gender


def _parse_ident(ident: str) -> tuple[str, int, str]:
    """'p1a: Garchomp' -> ('p1', 0, 'Garchomp'); 'p2: Garchomp' -> ('p2', -1, 'Garchomp')."""
    head, _, name = ident.partition(': ')
    side = head[:2]
    pos = -1
    if len(head) > 2:
        pos = ord(head[2]) - ord('a')
    return side, pos, name.strip()


def _parse_condition(cond: str) -> tuple[float, int | None, int | None, str, bool]:
    """'123/200 par' -> (frac, 123, 200, 'par', False); '0 fnt' -> fainted; '55/100' for percentages."""
    cond = cond.strip()
    if not cond:
        return 1.0, None, None, '', False
    parts = cond.split(' ')
    hp_part = parts[0]
    status = parts[1] if len(parts) > 1 else ''
    if status == 'fnt' or hp_part == '0':
        return 0.0, 0, None, '', True
    if '/' in hp_part:
        cur, _, mx = hp_part.partition('/')
        mx_digits = ''.join(ch for ch in mx if ch.isdigit())
        cur_i = int(cur)
        mx_i = int(mx_digits) if mx_digits else 100
        return (cur_i / mx_i if mx_i else 0.0), cur_i, mx_i, status, False
    return 1.0, None, None, status, False


def _is_ident(s: str) -> bool:
    return len(s) > 4 and s[0] == 'p' and s[1] in '1234' and ': ' in s[:5]


def _effect_name(s: str) -> tuple[str, str]:
    """'item: Leftovers' -> ('item', 'Leftovers'); 'move: Protect' -> ('move', 'Protect'); 'brn' -> ('', 'brn')."""
    s = s.strip()
    for prefix in ('item: ', 'ability: ', 'move: ', 'pokemon: '):
        if s.startswith(prefix):
            return prefix[:-2], s[len(prefix):]
    return '', s


def _kwargs(parts: list[str]) -> dict:
    out = {}
    for p in parts:
        if p.startswith('[') and ']' in p:
            key = p[1:p.index(']')]
            out[key] = p[p.index(']') + 1:].strip()
    return out


TERRAINS = {'electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain'}
COPY_ABILITIES = {'Trace', 'Receiver', 'Power of Alchemy', 'Mummy', 'Lingering Aroma', 'Wandering Spirit'}
BATON_PASS_VOLATILES = {'substitute', 'aquaring', 'confusion', 'curse', 'embargo', 'focusenergy', 'gastroacid',
                        'healblock', 'ingrain', 'leechseed', 'magnetrise', 'perishsong', 'powertrick',
                        'telekinesis', 'laserfocus', 'dragoncheer'}


class LogTracker:
    """Builds a :class:`BattleView` for one player from the battle protocol (+ requests).

    Feed it protocol lines with :meth:`feed` (``|split|`` blocks are resolved for
    this player: their own secret lines, everyone else's public ones) and the
    player's latest request with :meth:`update_request`.
    """

    def __init__(self, me: str, formatid: str = ''):
        self.view = BattleView(me=me, formatid=formatid)
        for sid in ('p1', 'p2'):
            self.view.sides[sid] = SideView(sideid=sid)
        self._split: str | None = None
        self._split_left = 0
        self._preview: dict[str, list[PokemonView]] = {'p1': [], 'p2': []}
        self.dex = get_dex()

    # -- public API ----------------------------------------------------------
    def feed(self, lines):
        for line in lines:
            self._feed_line(line)

    def _feed_line(self, line: str):
        if self._split_left:
            # first line after |split|<side> is the secret one, second is the public one
            secret = self._split_left == 2
            self._split_left -= 1
            if (secret and self._split != self.view.me) or (not secret and self._split == self.view.me):
                return
        if not line.startswith('|'):
            return
        parts = line.split('|')[1:]
        if not parts:
            return
        cmd = parts[0]
        if cmd == 'split':
            self._split = parts[1] if len(parts) > 1 else None
            self._split_left = 2
            return
        handler = getattr(self, '_on_' + cmd.replace('-', '_'), None)
        if handler is not None:
            try:
                handler(parts[1:])
            except (IndexError, ValueError, KeyError):
                pass  # never let an unexpected message break the AI

    # -- helpers -----------------------------------------------------------
    def _get(self, ident: str, details: str | None = None, create: bool = True) -> PokemonView | None:
        sid, _pos, name = _parse_ident(ident)
        side = self.view.sides.get(sid)
        if side is None:
            return None
        mon = side.find(name)
        if mon is not None:
            return mon
        if not create:
            return None
        species = name
        level, gender = 50, ''
        if details:
            species, level, gender = _species_of(details)
        # match a team preview entry (or own team entry) by species
        mon = self._match_unnamed(side, species)
        if mon is None:
            mon = PokemonView(species=species, base_species=species, level=level, gender=gender, forme=species)
            self._set_types(mon)
            side.pokemon.append(mon)
        mon.name = name
        return mon

    def _match_unnamed(self, side: SideView, species: str) -> PokemonView | None:
        dex = self.dex
        sp = dex.species.get(species)
        base = sp.baseSpecies if sp.exists else species
        for exact in (True, False):
            for p in side.pokemon:
                if p.name:
                    continue
                if exact and p.species == species:
                    return p
                if not exact:
                    psp = dex.species.get(p.species)
                    if (psp.baseSpecies if psp.exists else p.species) == base:
                        return p
        return None

    def _set_types(self, mon: PokemonView):
        sp = self.dex.species.get(mon.species)
        if sp.exists:
            mon.types = list(sp.types)

    def _reveal_from(self, mon: PokemonView | None, kw: dict, of_mon: PokemonView | None = None,
                     owner_is_target: bool = False):
        src = kw.get('from')
        if not src:
            return
        kind, name = _effect_name(src)
        # "[of]" names the effect's holder (Rough Skin, Rocky Helmet, Intimidate, ...) except on -heal,
        # where it is the other Pokemon involved (Shell Bell, Volt Absorb heal their own holder)
        owner = mon if (owner_is_target and mon is not None) else (of_mon or mon)
        if owner is None:
            return
        if kind == 'item':
            if owner.item == '' and owner.last_item == name:
                return  # e.g. "-heal ... [from] item: Oran Berry" right after eating it
            owner.item = name
        elif kind == 'ability':
            owner.reveal_ability(name)

    # -- message handlers ----------------------------------------------------
    def _on_gametype(self, args):
        self.view.gametype = args[0]

    def _on_player(self, args):
        if len(args) >= 2 and args[0] in self.view.sides:
            self.view.sides[args[0]].name = args[1]

    def _on_teamsize(self, args):
        self.view.sides[args[0]].team_size = int(args[1])

    def _on_clearpoke(self, args):
        for sid in ('p1', 'p2'):
            if sid != self.view.me:
                self.view.sides[sid].pokemon = []

    def _on_poke(self, args):
        sid, details = args[0], args[1]
        if sid == self.view.me:
            return  # our own team comes from the request
        species, level, gender = _species_of(details)
        mon = PokemonView(species=species, base_species=species, level=level, gender=gender, forme=species)
        self._set_types(mon)
        self.view.sides[sid].pokemon.append(mon)

    def _on_turn(self, args):
        self.view.turn = int(args[0])

    def _on_win(self, args):
        self.view.ended = True
        name = args[0]
        for sid, side in self.view.sides.items():
            if side.name == name:
                self.view.winner = sid

    def _on_tie(self, args):
        self.view.ended = True

    def _switch_in(self, args, is_drag=False):
        ident, details = args[0], args[1]
        sid, pos, _name = _parse_ident(ident)
        side = self.view.sides[sid]
        mon = self._get(ident, details)
        species, level, gender = _species_of(details)
        prev = side.active_at(pos)
        kw = _kwargs(args[3:])
        passed = None
        if prev is not None and prev is not mon:
            if kw.get('from') in ('Baton Pass', 'Shed Tail'):
                passed = (dict(prev.boosts), set(prev.volatiles), kw['from'])
            prev.reset_switch_out()
        mon.reset_switch_out()
        if mon.species != species or mon.forme != species:
            mon.species = species
            mon.forme = species
            self._set_types(mon)
            if '-Mega' in species:
                mon.mega = True
        if passed is not None:
            boosts, vols, how = passed
            keep = {'substitute'} if how == 'Shed Tail' else BATON_PASS_VOLATILES
            if how == 'Baton Pass':
                mon.boosts = boosts
            mon.volatiles = {v for v in vols if v in keep}
        mon.level = level
        mon.gender = gender
        mon.active = True
        mon.slot = pos
        mon.revealed = True
        mon.brought = True
        if len(args) > 2:
            frac, cur, mx, status, fainted = _parse_condition(args[2])
            mon.hp = frac
            if sid == self.view.me and cur is not None:
                mon.hp_exact, mon.maxhp = cur, mx
            mon.status = status
            mon.fainted = fainted

    _on_switch = _switch_in

    def _on_drag(self, args):
        self._switch_in(args, True)

    def _on_replace(self, args):
        self._switch_in(args)

    def _on_swap(self, args):
        sid, pos, _ = _parse_ident(args[0])
        mon = self._get(args[0])
        target = int(args[1])
        other = self.view.sides[sid].active_at(target)
        if other is not None and other is not mon:
            other.slot = mon.slot
        mon.slot = target

    def _on_detailschange(self, args):
        mon = self._get(args[0], args[1])
        species, _, _ = _species_of(args[1])
        mon.species = species
        mon.forme = species
        self._set_types(mon)
        sp = self.dex.species.get(species)
        if sp.exists and sp.isMega:
            # Mega Evolution sets the (single) ability of the Mega forme
            mon.mega = True
            ability = sp.abilities.get('0')
            if ability:
                mon.ability = mon.base_ability = ability
        if '-Mega' in species:
            mon.mega = True

    def _on__formechange(self, args):
        mon = self._get(args[0])
        mon.species = args[1]
        self._set_types(mon)

    def _on__mega(self, args):
        mon = self._get(args[0])
        mon.mega = True
        if len(args) > 2 and args[2]:
            mon.item = args[2]
        self.view.sides[_parse_ident(args[0])[0]].mega_used = True

    def _on_faint(self, args):
        mon = self._get(args[0])
        mon.fainted = True
        mon.hp = 0.0
        if mon.hp_exact is not None:
            mon.hp_exact = 0
        mon.status = ''

    def _on_move(self, args):
        mon = self._get(args[0])
        move = args[1]
        kw = _kwargs(args[3:])
        src = kw.get('from', '')
        mon.last_move = move
        if move == 'Struggle':
            return
        if src and src != 'lockedmove':
            # called by another move (Sleep Talk, Metronome, ...) or an ability (Dancer)
            return
        mon.moves[move] = mon.moves.get(move, 0) + 1
        # Pressure costs an extra PP; ignored (we only need "revealed" here)

    def _on__damage(self, args, heal=False):
        mon = self._get(args[0])
        frac, cur, mx, status, fainted = _parse_condition(args[1])
        mon.hp = frac
        if self._is_me(args[0]) and cur is not None:
            mon.hp_exact, mon.maxhp = cur, mx
        if fainted:
            mon.fainted = True
        else:
            mon.status = status
        kw = _kwargs(args[2:])
        of_mon = self._get(kw['of']) if kw.get('of') else None
        self._reveal_from(mon, kw, of_mon, owner_is_target=heal)
        if 'from' not in kw and not heal:
            mon.times_attacked += 1

    def _on__heal(self, args):
        self._on__damage(args, heal=True)

    def _on__sethp(self, args):
        self._on__damage(args, heal=True)

    def _is_me(self, ident: str) -> bool:
        return _parse_ident(ident)[0] == self.view.me

    def _on__status(self, args):
        mon = self._get(args[0])
        mon.status = args[1]
        if args[1] == 'slp':
            mon.sleep_turns = 0
        if args[1] == 'tox':
            mon.toxic_turns = 0
        kw = _kwargs(args[2:])
        of_mon = self._get(kw['of']) if kw.get('of') else None
        self._reveal_from(mon, kw, of_mon)

    def _on__curestatus(self, args):
        mon = self._get(args[0])
        mon.status = ''
        kw = _kwargs(args[2:])
        self._reveal_from(mon, kw)

    def _on__cureteam(self, args):
        sid = _parse_ident(args[0])[0]
        for p in self.view.sides[sid].pokemon:
            p.status = ''

    def _on_cant(self, args):
        mon = self._get(args[0])
        if len(args) > 1 and args[1] == 'slp':
            mon.sleep_turns += 1
        kw = _kwargs(args[2:])
        self._reveal_from(mon, kw)

    def _boost(self, args, sign):
        mon = self._get(args[0])
        stat = args[1]
        amount = int(args[2])
        if stat in mon.boosts:
            mon.boosts[stat] = max(-6, min(6, mon.boosts[stat] + sign * amount))
        kw = _kwargs(args[3:])
        of_mon = self._get(kw['of']) if kw.get('of') else None
        self._reveal_from(mon, kw, of_mon)

    def _on__boost(self, args):
        self._boost(args, 1)

    def _on__unboost(self, args):
        self._boost(args, -1)

    def _on__setboost(self, args):
        mon = self._get(args[0])
        if args[1] in mon.boosts:
            mon.boosts[args[1]] = int(args[2])

    def _on__clearboost(self, args):
        mon = self._get(args[0])
        mon.boosts = {s: 0 for s in BOOST_STATS}

    def _on__clearallboost(self, args):
        for side in self.view.sides.values():
            for p in side.active():
                p.boosts = {s: 0 for s in BOOST_STATS}

    def _on__clearnegativeboost(self, args):
        mon = self._get(args[0])
        mon.boosts = {s: max(0, v) for s, v in mon.boosts.items()}

    def _on__clearpositiveboost(self, args):
        mon = self._get(args[0])
        mon.boosts = {s: min(0, v) for s, v in mon.boosts.items()}

    def _on__invertboost(self, args):
        mon = self._get(args[0])
        mon.boosts = {s: -v for s, v in mon.boosts.items()}

    def _on__copyboost(self, args):
        mon = self._get(args[0])
        src = self._get(args[1])
        mon.boosts = dict(src.boosts)

    def _on__swapboost(self, args):
        a = self._get(args[0])
        b = self._get(args[1])
        stats = [s.strip() for s in args[2].split(',')] if len(args) > 2 and args[2] else list(BOOST_STATS)
        for s in stats:
            if s in a.boosts:
                a.boosts[s], b.boosts[s] = b.boosts[s], a.boosts[s]

    def _on__item(self, args):
        mon = self._get(args[0])
        mon.item = args[1]
        kw = _kwargs(args[2:])
        of_mon = self._get(kw['of']) if kw.get('of') else None
        src = kw.get('from', '')
        if src.startswith('ability:') and of_mon is not None:
            of_mon.reveal_ability(_effect_name(src)[1])
        elif src.startswith('ability:'):
            mon.reveal_ability(_effect_name(src)[1])

    def _on__enditem(self, args):
        mon = self._get(args[0])
        mon.last_item = args[1]
        mon.item = ''
        kw = _kwargs(args[2:])
        src = kw.get('from', '')
        of_mon = self._get(kw['of']) if kw.get('of') else None
        if src.startswith('ability:') and of_mon is not None:
            of_mon.reveal_ability(_effect_name(src)[1])

    def _on__ability(self, args):
        mon = self._get(args[0])
        kw = _kwargs(args[2:])
        src = kw.get('from', '')
        of_mon = self._get(kw['of']) if kw.get('of') else None
        kind, name = _effect_name(src) if src else ('', '')
        if kind == 'move' or (kind == 'ability' and name in COPY_ABILITIES):
            # Role Play / Entrainment / Worry Seed / Simple Beam / Doodle, Trace / Receiver / ...:
            # the new ability lasts until it switches out
            old_ability = [a for a in args[2:] if a and not a.startswith('[')]
            if 'abilitychanged' not in mon.volatiles:
                if kind == 'ability':
                    mon.base_ability = name
                elif old_ability:
                    mon.base_ability = old_ability[0]
            mon.volatiles.add('abilitychanged')
            mon.ability = args[1]
            if of_mon is not None and of_mon is not mon and name in ('Trace', 'Role Play', 'Doodle'):
                of_mon.reveal_ability(args[1])
        else:
            mon.reveal_ability(args[1])

    def _on__endability(self, args):
        mon = self._get(args[0])
        mon.volatiles.add('gastroacid')

    def _on__transform(self, args):
        mon = self._get(args[0])
        target = self._get(args[1])
        mon.volatiles.add('transform')
        mon.species = target.species
        mon.types = list(target.types)
        mon.boosts = dict(target.boosts)
        kw = _kwargs(args[2:])
        self._reveal_from(mon, kw)

    def _on__start(self, args):
        mon = self._get(args[0])
        kind, name = _effect_name(args[1])
        vid = to_id(name)
        if vid == 'typechange' and len(args) > 2:
            mon.types = [t.strip() for t in args[2].split('/') if t.strip()]
        elif vid == 'typeadd' and len(args) > 2 and args[2] not in mon.types:
            mon.types = mon.types + [args[2]]
        mon.volatiles.add(vid)
        kw = _kwargs(args[2:])
        of_mon = self._get(kw['of']) if kw.get('of') else None
        self._reveal_from(mon, kw, of_mon)
        if kind == 'ability':
            mon.reveal_ability(name)

    def _on__end(self, args):
        mon = self._get(args[0])
        _kind, name = _effect_name(args[1])
        mon.volatiles.discard(to_id(name))

    def _on__activate(self, args):
        if not args or not args[0]:
            return
        if ':' not in args[0] or len(args[0].split(':')[0]) > 3:
            return
        mon = self._get(args[0])
        if mon is None or len(args) < 2:
            return
        kind, name = _effect_name(args[1])
        kw = _kwargs(args[2:])
        of_mon = self._get(kw['of']) if kw.get('of') else None
        plain = [a for a in args[2:] if a and not a.startswith('[')]
        if kind == 'item':
            mon.item = name
        elif kind == 'ability' and name == 'Symbiosis' and of_mon is not None and plain:
            # the Symbiosis holder passes its item to the ally
            mon.reveal_ability(name)
            mon.item = ''
            mon.last_item = plain[0]
            of_mon.item = plain[0]
        elif kind == 'ability' and name in ('Mummy', 'Lingering Aroma') and plain and _is_ident(plain[0]):
            mon.reveal_ability(name)
            other = self._get(plain[0])
            if other is not None:
                if 'abilitychanged' not in other.volatiles and kw.get('ability'):
                    other.base_ability = kw['ability']
                other.volatiles.add('abilitychanged')
                other.ability = name
        elif kind == 'ability':
            mon.reveal_ability(name)
        elif kind == 'move' and name in ('Trick', 'Switcheroo') and of_mon is not None:
            # the items are exchanged; the following -item lines say what each one got
            mon.item, of_mon.item = '', ''
        elif name == 'Skill Swap' and of_mon is not None:
            if len(plain) >= 2:
                target_ability, source_ability = plain[0], plain[1]
                if 'abilitychanged' not in of_mon.volatiles:
                    of_mon.base_ability = target_ability
                if 'abilitychanged' not in mon.volatiles:
                    mon.base_ability = source_ability
                mon.ability, of_mon.ability = target_ability, source_ability
            mon.volatiles.add('abilitychanged')
            of_mon.volatiles.add('abilitychanged')

    def _on__weather(self, args):
        weather = args[0]
        kw = _kwargs(args[1:])
        if 'upkeep' in kw:
            return
        f = self.view.field
        f.weather = '' if weather == 'none' else to_id(weather)
        f.weather_turn = self.view.turn
        if kw.get('of'):
            self._reveal_from(None, kw, self._get(kw['of']))

    def _on__fieldstart(self, args):
        _kind, name = _effect_name(args[0])
        cid = to_id(name)
        f = self.view.field
        if cid in TERRAINS:
            f.terrain = cid
            f.terrain_turn = self.view.turn
        else:
            f.pseudo[cid] = self.view.turn
        kw = _kwargs(args[1:])
        if kw.get('of') and kw.get('from'):
            self._reveal_from(None, kw, self._get(kw['of']))

    def _on__fieldend(self, args):
        _kind, name = _effect_name(args[0])
        cid = to_id(name)
        f = self.view.field
        if cid in TERRAINS:
            if f.terrain == cid:
                f.terrain = ''
        else:
            f.pseudo.pop(cid, None)

    def _on__sidestart(self, args):
        sid = args[0][:2]
        _kind, name = _effect_name(args[1])
        cid = to_id(name)
        side = self.view.sides[sid]
        side.conditions[cid] = side.conditions.get(cid, 0) + 1
        side.condition_turns.setdefault(cid, self.view.turn)

    def _on__sideend(self, args):
        sid = args[0][:2]
        _kind, name = _effect_name(args[1])
        cid = to_id(name)
        side = self.view.sides[sid]
        side.conditions.pop(cid, None)
        side.condition_turns.pop(cid, None)

    def _on__swapsideconditions(self, args):
        a, b = self.view.sides['p1'], self.view.sides['p2']
        a.conditions, b.conditions = b.conditions, a.conditions
        a.condition_turns, b.condition_turns = b.condition_turns, a.condition_turns

    # -- request -------------------------------------------------------------
    def update_request(self, request: dict | None):
        """Merge the player's own (complete) information from a battle request."""
        if not request or 'side' not in request:
            return
        dex = self.dex
        side = self.view.my_side
        req_side = request['side']
        side.name = req_side.get('name') or side.name
        existing = {p.name: p for p in side.pokemon if p.name}
        new_list = []
        for i, entry in enumerate(req_side['pokemon']):
            _sid, _pos, name = _parse_ident(entry['ident'])
            species, level, gender = _species_of(entry['details'])
            mon = existing.get(name)
            if mon is None:
                mon = PokemonView(species=species, base_species=species, name=name, forme=species)
            if mon.forme != species:
                # permanent forme change (Mega Evolution); temporary ones come from the log
                mon.forme = species
                mon.species = species
                if '-Mega' in species:
                    mon.mega = True
            sp = dex.species.get(mon.species)
            if sp.exists and not ({'typechange', 'transform'} & mon.volatiles):
                mon.types = list(sp.types)
            mon.level, mon.gender = level, gender
            frac, cur, mx, status, fainted = _parse_condition(entry['condition'])
            mon.hp, mon.hp_exact, mon.maxhp = frac, cur, mx
            mon.status = status
            mon.fainted = fainted
            mon.stats = dict(entry.get('stats') or {})
            mon.item = dex.items.get(entry.get('item') or '').name if entry.get('item') else ''
            mon.ability = dex.abilities.get(entry.get('ability') or entry.get('baseAbility') or '').name
            mon.base_ability = dex.abilities.get(entry.get('baseAbility') or entry.get('ability') or '').name
            mon.revealed = True
            moves = {}
            for mid in entry.get('moves') or []:
                mname = dex.moves.get(mid).name or mid
                moves[mname] = mon.moves.get(mname, 0)
            mon.moves = moves
            if entry.get('active'):
                # the first entries of the request are the active Pokemon, in slot order
                mon.active = True
                mon.slot = i
            elif mon.active:
                mon.reset_switch_out()
            new_list.append(mon)
        side.pokemon = new_list
        if request.get('teamPreview'):
            side.team_size = request.get('maxChosenTeamSize') or len(new_list)
        active_reqs = request.get('active') or []
        actives = [p for p in new_list if p.active]
        actives.sort(key=lambda p: p.slot)
        for mon, areq in zip(actives, active_reqs):
            if not areq:
                continue
            mon.move_pp = {}
            mon.disabled_moves = set()
            for m in areq.get('moves') or []:
                mname = m.get('move')
                if m.get('pp') is not None:
                    mon.move_pp[mname] = (m.get('pp'), m.get('maxpp'))
                if m.get('disabled'):
                    mon.disabled_moves.add(mname)
            mon.can_mega = bool(areq.get('canMegaEvo'))
            mon.trapped = bool(areq.get('trapped'))
        if not any(p.can_mega for p in actives):
            for p in new_list:
                if not p.active:
                    p.can_mega = False
