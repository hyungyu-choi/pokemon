"""Search agent: plays every turn with the live advisor (determinized Monte-Carlo lookahead).

It sees only what a player sees (its :class:`BattleView`), converts that into the
advisor's input format, samples the opponent's hidden information, simulates each
candidate action and plays the best one.  This is the same procedure the advisor
uses for a human player, so its win rate measures the advisor's strength.
"""
from __future__ import annotations

from ..env.view import BattleView
from .advisor import Advisor, AdvisorInput, MonState, SideState

DEFAULT_DURATIONS = {'reflect': 5, 'lightscreen': 5, 'auroraveil': 5, 'tailwind': 4, 'safeguard': 5, 'mist': 5}
SEARCH_VOLATILES = ('substitute', 'confusion', 'leechseed', 'taunt', 'encore', 'yawn', 'focusenergy', 'aquaring',
                    'ingrain', 'magnetrise', 'curse', 'saltcure', 'healblock')


def _team_name(set_) -> str:
    return set_.get('species') or set_.get('name')


def view_to_input(view: BattleView, team: list, formatid: str) -> AdvisorInput:
    """Convert a player's view (+ their own team sets) into the advisor's input."""
    my_names = [_team_name(s) for s in team]

    def base_of(mon, names):
        for n in (mon.forme, mon.base_species, mon.species):
            if n in names:
                return n
        from ..sim.dex import get_dex
        base = get_dex().species.get(mon.species).baseSpecies
        for n in names:
            if get_dex().species.get(n).baseSpecies == base:
                return n
        return mon.base_species or mon.species

    def mon_state(mon, name, mine):
        ms = MonState(species=name)
        if mine and mon.hp_exact is not None:
            ms.hp_exact = mon.hp_exact
        ms.hp = mon.hp
        ms.status = mon.status
        ms.boosts = {k: v for k, v in mon.boosts.items() if v}
        ms.item = mon.item
        ms.ability = mon.ability if not mine else None
        ms.moves = list(mon.moves) if not mine else []
        ms.mega = mon.mega
        ms.fainted = mon.fainted
        ms.volatiles = [v for v in mon.volatiles if v in SEARCH_VOLATILES]
        if mine and mon.move_pp:
            ms.pp = {k: v[0] for k, v in mon.move_pp.items()}
        ms.sleep_turns = mon.sleep_turns
        ms.toxic_turns = mon.toxic_turns
        if not mine:
            ms.belief = mon.belief
            ms.view = mon
            ms.choices = [(h['move'], h['target'], view) for h in view.history if h['foe'] is mon]
        return ms

    me, foe = view.my_side, view.foe_side
    me_mons, me_brought, me_active = {}, [], []
    for mon in me.pokemon:
        name = base_of(mon, my_names)
        me_mons[name] = mon_state(mon, name, True)
        me_brought.append(name)
        if mon.active and not mon.fainted:
            me_active.append(name)
    # team-preview names (a Mega Evolved Pokemon is still listed under its base forme)
    foe_names = [p.base_species or p.forme or p.species for p in foe.pokemon]
    foe_mons, foe_brought, foe_active = {}, [], []
    for mon, name in zip(foe.pokemon, foe_names):
        if mon.revealed:
            foe_mons[name] = mon_state(mon, name, False)
            foe_brought.append(name)
            if mon.active and not mon.fainted:
                foe_active.append(name)

    def conds(side):
        out = {}
        for c, layers in side.conditions.items():
            if c in DEFAULT_DURATIONS:
                started = side.condition_turns.get(c, view.turn)
                out[c] = max(1, DEFAULT_DURATIONS[c] - (view.turn - started))
            else:
                out[c] = layers
        return out

    f = view.field
    return AdvisorInput(
        formatid=formatid, turn=view.turn,
        me=SideState(team=team, brought=me_brought, active=me_active, mons=me_mons, conditions=conds(me),
                     mega_used=me.mega_used),
        foe=SideState(team=foe_names, brought=foe_brought, active=foe_active, mons=foe_mons, conditions=conds(foe),
                      mega_used=foe.mega_used),
        weather=f.weather, weather_turns=max(1, 5 - (view.turn - f.weather_turn)) if f.weather else 0,
        terrain=f.terrain, terrain_turns=max(1, 5 - (view.turn - f.terrain_turn)) if f.terrain else 0,
        pseudo={k: max(1, 5 - (view.turn - t)) for k, t in f.pseudo.items()},
    )


class SearchAgent:
    name = 'search'

    def __init__(self, seed=None, model_path: str | None = None, library: str | None = None,
                 determinizations: int = 6, depth: int = 2, fallback=None, prior_weight: float = 0.04,
                 reply_k: int = 4):
        self.advisor = Advisor(model_path=model_path, library=library, seed=seed or 0, threads=1)
        self.determinizations = determinizations
        self.depth = depth
        self.prior_weight = prior_weight
        self.reply_k = reply_k
        if fallback is None:
            from .heuristic import HeuristicAgent
            fallback = HeuristicAgent(seed)
        self.fallback = fallback
        self.stats = {'search': 0, 'fallback': 0}

    def choose(self, session, sid, view, legal):
        request = session.request(sid)
        if request.get('teamPreview') or 'forceSwitch' in request or len(legal) == 1:
            return self.fallback.choose(session, sid, view, legal)
        try:
            team = session.battle.getSide(sid).team
            inp = view_to_input(view, team, session.battle.format.id)
            recs = self.advisor.recommend(inp, determinizations=self.determinizations, depth=self.depth,
                                          reply_k=self.reply_k)
        except Exception:
            recs = []
        legal_set = set(legal)
        # close calls (within the sampling noise) go to the policy's preferred action
        recs = sorted(recs, key=lambda r: -(r.win_rate + self.prior_weight * r.prior))
        for r in recs:
            option = translate_option(r.option, r.meaning, request)
            if option in legal_set:
                self.stats['search'] += 1
                return option
        self.stats['fallback'] += 1
        return self.fallback.choose(session, sid, view, legal)


def translate_option(option, meaning, request):
    """Map an option of the rebuilt battle onto the real request (team positions can differ)."""
    if meaning is None:
        return option
    from ..env.actions import decode, move_action, switch_action
    out = []
    names = [p['ident'].split(': ', 1)[1] for p in request['side']['pokemon']]
    species = [p['details'].split(',')[0] for p in request['side']['pokemon']]
    for slot, (a, m) in enumerate(zip(option, meaning)):
        if a < 0 or m is None:
            out.append(a)
            continue
        kind, value = m
        if kind == 'switch':
            pos = next((i for i, (n, sp) in enumerate(zip(names, species)) if value in (n, sp)), None)
            if pos is None:
                return None
            out.append(switch_action(pos))
        else:
            moves = [x['move'] for x in (request.get('active') or [{}] * 2)[slot]['moves']]
            if value not in moves:
                return None
            _, _i, target, mega = decode(a)
            out.append(move_action(moves.index(value), target, mega))
    return tuple(out)


class SearchFactory:
    """Picklable factory (for multiprocessing evaluation): ``SearchFactory(model)(seed) -> SearchAgent``."""

    def __init__(self, model_path=None, library=None, determinizations=8, depth=1, reply_k=4):
        self.model_path, self.library = model_path, library
        self.determinizations, self.depth, self.reply_k = determinizations, depth, reply_k
        self.name = 'search'

    def __call__(self, seed=None):
        fallback = None
        if self.model_path:
            from .train import _NNFactory
            fallback = _NNFactory(self.model_path)(seed)
        return SearchAgent(seed, model_path=self.model_path, library=self.library,
                           determinizations=self.determinizations, depth=self.depth, fallback=fallback,
                           reply_k=self.reply_k)
