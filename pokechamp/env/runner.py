"""Run battles between agents (headless; only results and optional logs are recorded)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..sim.battle import Battle
from .actions import (joint_choice, legal_joint_actions, team_choice, team_preview_options)
from .view import LogTracker

FORMAT_SINGLES = 'gen9championsbssregmc'
FORMAT_DOUBLES = 'gen9championsvgc2026regmc'


def gen5_seed(rng: random.Random) -> list[int]:
    return [rng.randrange(0x10000) for _ in range(4)]


@dataclass
class AgentChoice:
    """What an agent returns: the chosen option plus optional data kept for training."""
    option: object
    obs: object = None
    info: dict = field(default_factory=dict)


@dataclass
class Decision:
    """One decision of one player (kept for training)."""
    side: str
    kind: str                 # 'preview' or 'move'
    obs: object               # whatever the agent returned as its observation (or None)
    action: object            # chosen option (team-preview tuple or joint action tuple)
    legal: list               # legal options at that point
    info: dict = field(default_factory=dict)


@dataclass
class BattleResult:
    winner: str | None        # 'p1', 'p2' or None (tie / turn limit)
    turns: int
    log: list | None = None
    decisions: list = field(default_factory=list)
    teams: tuple = ()
    brought: dict = field(default_factory=dict)  # side -> list of team indices brought
    error: str | None = None


class BattleSession:
    """A battle plus one :class:`LogTracker` per player."""

    def __init__(self, formatid: str, team1, team2, seed=None, names=('p1', 'p2')):
        self.formatid = formatid
        self.battle = Battle(formatid, seed=seed, p1={'name': names[0], 'team': team1},
                             p2={'name': names[1], 'team': team2})
        self.trackers = {sid: LogTracker(sid, formatid) for sid in ('p1', 'p2')}
        self._log_pos = {'p1': 0, 'p2': 0}
        self.n_active = 2 if self.battle.gameType == 'doubles' else 1
        self.picked = self.battle.ruleTable.pickedTeamSize or len(team1)

    def pending(self) -> list[str]:
        out = []
        for side in self.battle.sides:
            req = side.activeRequest
            if req and not req.get('wait') and not side.isChoiceDone():
                out.append(side.id)
        return out

    def view(self, sid: str):
        tracker = self.trackers[sid]
        log = self.battle.log
        tracker.feed(log[self._log_pos[sid]:])
        self._log_pos[sid] = len(log)
        tracker.update_request(self.request(sid))
        return tracker.view

    def request(self, sid: str):
        return self.battle.getSide(sid).activeRequest

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


def play_battle(team1, team2, agent1, agent2, formatid: str = FORMAT_SINGLES, seed=None,
                max_turns: int = 200, keep_log: bool = False, record: bool = False,
                names=('p1', 'p2')) -> BattleResult:
    """Play one battle.

    Agents implement ``choose(session, sid, view, legal) -> option | AgentChoice`` where ``legal`` is the
    list of legal options (team-preview orders or joint slot actions, see :mod:`pokechamp.env.actions`).
    """
    session = BattleSession(formatid, team1, team2, seed=seed, names=names)
    agents = {'p1': agent1, 'p2': agent2}
    for sid, agent in agents.items():
        if hasattr(agent, 'start_battle'):
            agent.start_battle(session, sid)
    battle = session.battle
    decisions = []
    brought = {}
    guard = 0
    error = None
    try:
        while not battle.ended and battle.turn <= max_turns:
            guard += 1
            if guard > 5000:
                raise RuntimeError('battle loop guard exceeded')
            pending = session.pending()
            if not pending:
                break
            # decide for every pending side before applying (simultaneous choices)
            choices = {}
            for sid in pending:
                view = session.view(sid)
                legal = session.legal(sid)
                out = agents[sid].choose(session, sid, view, legal)
                choices[sid] = (out, legal)
            for sid in pending:
                (option, obs, info), legal = _unpack(choices[sid])
                req = session.request(sid)
                kind = 'preview' if req.get('teamPreview') else 'move'
                if kind == 'preview':
                    brought[sid] = list(option[:session.picked])
                ok = session.apply(sid, option)
                if not ok:
                    remaining = [o for o in legal if o != option]
                    while remaining and not ok:
                        option = remaining.pop(0)
                        ok = session.apply(sid, option)
                    if not ok:
                        session.apply_default(sid)
                        option = None
                if record and option is not None:
                    decisions.append(Decision(sid, kind, obs, option, legal, info))
                if battle.ended:
                    break
    except Exception as exc:  # keep long training runs alive; report the problem
        error = f"{type(exc).__name__}: {exc}"
    for sid, agent in agents.items():
        if hasattr(agent, 'end_battle'):
            agent.end_battle(session, sid)
    return BattleResult(
        winner=session.winner if battle.ended else None,
        turns=battle.turn,
        log=list(battle.log) if keep_log else None,
        decisions=decisions,
        teams=(team1, team2),
        brought=brought,
        error=error,
    )


def _unpack(choice):
    out, legal = choice
    if isinstance(out, AgentChoice):
        return (out.option, out.obs, out.info), legal
    return (out, None, {}), legal
