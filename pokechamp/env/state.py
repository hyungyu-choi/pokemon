"""Direct access to engine state: full-information views and a session wrapper for rollouts."""
from __future__ import annotations

from ..sim.battle import Battle
from .actions import joint_choice, legal_joint_actions, team_choice, team_preview_options
from .view import BOOST_STATS, BattleView, FieldView, PokemonView, SideView


def _mon_view(p, battle: Battle) -> PokemonView:
    dex = battle.dex
    status = p.status if p.status in ('brn', 'par', 'slp', 'frz', 'psn', 'tox') else ''
    v = PokemonView(
        species=p.species.name, base_species=p.baseSpecies.name, name=p.name, level=p.level,
        gender=p.gender or '', hp=(p.hp / p.maxhp if p.maxhp else 0.0), hp_exact=p.hp, maxhp=p.maxhp,
        status=status, fainted=bool(p.fainted), active=bool(p.isActive), slot=p.position if p.isActive else -1,
        revealed=True,
    )
    v.boosts = {s: p.boosts[s] for s in BOOST_STATS}
    v.volatiles = set(p.volatiles.keys())
    v.moves = {dex.moves.get(ms.id).name: (ms.maxpp - ms.pp) for ms in p.moveSlots}
    v.move_pp = {dex.moves.get(ms.id).name: (ms.pp, ms.maxpp) for ms in p.moveSlots}
    v.disabled_moves = {dex.moves.get(ms.id).name for ms in p.moveSlots if ms.disabled}
    v.item = dex.items.get(p.item).name if p.item else ''
    v.last_item = dex.items.get(p.lastItem).name if p.lastItem else ''
    v.ability = dex.abilities.get(p.ability).name
    v.base_ability = dex.abilities.get(p.baseAbility).name
    v.forme = p.species.name
    v.stats = {s: p.storedStats[s] for s in ('atk', 'def', 'spa', 'spd', 'spe')}
    v.mega = bool(p.species.isMega)
    v.can_mega = bool(p.canMegaEvo)
    v.types = list(p.getTypes())
    if status == 'slp' and p.statusState.get('startTime'):
        v.sleep_turns = max(0, (p.statusState.startTime or 0) - (p.statusState.time or 0))
    if status == 'tox':
        v.toxic_turns = p.statusState.get('stage') or 0
    v.last_move = dex.moves.get(p.lastMove.id).name if p.lastMove else ''
    v.times_attacked = p.timesAttacked or 0
    return v


def view_from_battle(battle: Battle, sid: str) -> BattleView:
    """Full-information :class:`BattleView` of ``battle`` from ``sid``'s side (for search rollouts)."""
    view = BattleView(me=sid, gametype=battle.gameType, formatid=battle.format.id, turn=battle.turn)
    for side in battle.sides:
        sv = SideView(sideid=side.id, name=side.name)
        sv.pokemon = [_mon_view(p, battle) for p in side.pokemon]
        sv.conditions = {k: (st.get('layers') or 1) for k, st in side.sideConditions.items()}
        sv.team_size = len(side.pokemon)
        sv.mega_used = not any(p.canMegaEvo for p in side.pokemon) and any(p.species.isMega for p in side.pokemon)
        view.sides[side.id] = sv
    f = FieldView()
    f.weather = battle.field.weather or ''
    f.terrain = battle.field.terrain or ''
    f.pseudo = {k: battle.turn for k in battle.field.pseudoWeather}
    view.field = f
    view.ended = battle.ended
    return view


class RolloutSession:
    """Minimal session over an existing :class:`Battle` with full-information views."""

    def __init__(self, battle: Battle):
        self.battle = battle
        self.n_active = 2 if battle.gameType == 'doubles' else 1
        self.picked = battle.ruleTable.pickedTeamSize or 6
        self.view_fns = {}  # side id -> callable(battle) -> BattleView (default: full information)

    def pending(self) -> list[str]:
        out = []
        for side in self.battle.sides:
            req = side.activeRequest
            if req and not req.get('wait') and not side.isChoiceDone():
                out.append(side.id)
        return out

    def request(self, sid: str):
        return self.battle.getSide(sid).activeRequest

    def view(self, sid: str) -> BattleView:
        fn = self.view_fns.get(sid)
        return fn(self.battle) if fn is not None else view_from_battle(self.battle, sid)

    def foe_alive(self, sid: str):
        foe = self.battle.getSide(sid).foe
        return tuple(bool(p and not p.fainted) for p in foe.active) + (True,) * (2 - len(foe.active))

    def legal(self, sid: str) -> list:
        req = self.request(sid)
        if req.get('teamPreview'):
            size = len(req['side']['pokemon'])
            return team_preview_options(size, min(self.picked, size), self.n_active)
        return legal_joint_actions(req, self.n_active, self.foe_alive(sid))

    def apply(self, sid: str, option) -> bool:
        req = self.request(sid)
        if req.get('teamPreview'):
            choice = team_choice(option, len(req['side']['pokemon']))
        else:
            choice = joint_choice(req, option)
        ok = self.battle.choose(sid, choice)
        if not ok:
            side = self.battle.getSide(sid)
            if side.activeRequest is not None and side.choice is not None:
                side.clearChoice()
        return ok

    def apply_default(self, sid: str):
        self.battle.choose(sid, 'default')

    @property
    def ended(self) -> bool:
        return self.battle.ended

    @property
    def winner(self) -> str | None:
        w = self.battle.winner
        if not w:
            return None
        for side in self.battle.sides:
            if side.name == w:
                return side.id
        return None
