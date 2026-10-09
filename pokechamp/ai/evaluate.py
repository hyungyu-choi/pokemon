"""Head-to-head evaluation of agents (paired games with swapped teams, optional multiprocessing).

Besides :func:`evaluate` (fresh random teams from a seed) this module provides fixed evaluation sets for
training runs: :func:`make_eval_pairs` draws seeded team pairs once (plain JSON-able data, stored in the
training state so every evaluation of a run uses the same games) and :func:`play_pairs` plays them.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from ..env.runner import FORMAT_SINGLES, gen5_seed, play_battle
from ..teambuilder.random_teams import RandomTeamGenerator


@dataclass
class MatchResult:
    wins: int = 0
    losses: int = 0
    ties: int = 0
    errors: int = 0
    turns: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.ties) / max(1, self.games)

    def ci95(self) -> float:
        n = max(1, self.games)
        p = self.win_rate
        return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n)

    def to_dict(self) -> dict:
        return {'wins': self.wins, 'losses': self.losses, 'ties': self.ties, 'errors': self.errors,
                'games': self.games, 'turns': self.turns, 'win_rate': round(self.win_rate, 4),
                'ci95': round(self.ci95(), 4)}

    @classmethod
    def from_dict(cls, d: dict) -> 'MatchResult':
        return cls(d.get('wins', 0), d.get('losses', 0), d.get('ties', 0), d.get('errors', 0), d.get('turns', 0))

    def __add__(self, other: 'MatchResult') -> 'MatchResult':
        return MatchResult(self.wins + other.wins, self.losses + other.losses, self.ties + other.ties,
                           self.errors + other.errors, self.turns + other.turns)

    def __str__(self) -> str:
        return (f"{self.win_rate * 100:5.1f}% ± {self.ci95() * 100:.1f}  "
                f"(W {self.wins} / L {self.losses} / T {self.ties}, {self.games} games, "
                f"{self.turns / max(1, self.games):.1f} turns/game, errors {self.errors})")


def _play_pair(t1, t2, agent_a, agent_b, formatid, battle_seed, res: MatchResult, max_turns: int = 200):
    """Agent A (always p1) plays both teams of a pair against agent B with the same battle seed."""
    for swap in (False, True):
        ta, tb = (t2, t1) if swap else (t1, t2)
        r = play_battle(ta, tb, agent_a, agent_b, formatid, seed=list(battle_seed), max_turns=max_turns)
        res.turns += r.turns
        if r.error:
            res.errors += 1
        if r.winner == 'p1':
            res.wins += 1
        elif r.winner == 'p2':
            res.losses += 1
        else:
            res.ties += 1


def _play_pairs(args):
    factory_a, factory_b, n_pairs, formatid, seed, team_source = args
    rng = random.Random(seed)
    gen = RandomTeamGenerator(rng.randrange(1 << 30), formatid=formatid)
    agent_a = factory_a(rng.randrange(1 << 30))
    agent_b = factory_b(rng.randrange(1 << 30))
    res = MatchResult()
    for _ in range(n_pairs):
        if team_source is not None:
            t1, t2 = team_source(rng)
        else:
            t1, t2 = gen.random_team(), gen.random_team()
        battle_seed = gen5_seed(rng)
        _play_pair(t1, t2, agent_a, agent_b, formatid, battle_seed, res)
    return res


def evaluate(factory_a, factory_b, n_games: int = 100, formatid: str = FORMAT_SINGLES, seed: int = 0,
             workers: int = 1, team_source=None, executor=None) -> MatchResult:
    """Win rate of agent A against agent B.

    ``factory_x(seed) -> agent`` must be picklable (a module-level function or class) when
    ``workers > 1``.  Each pair of teams is played twice with the sides swapped, so team
    strength cancels out.  ``team_source(rng) -> (team1, team2)`` overrides random teams.
    ``executor``: an existing process pool to run the games on (instead of starting a new one).
    """
    n_pairs = max(1, n_games // 2)
    if workers <= 1 and executor is None:
        return _play_pairs((factory_a, factory_b, n_pairs, formatid, seed, team_source))
    workers = max(1, workers)
    chunks = []
    per = math.ceil(n_pairs / workers)
    for w in range(workers):
        k = min(per, n_pairs - w * per)
        if k > 0:
            chunks.append((factory_a, factory_b, k, formatid, seed * 1000 + w, team_source))
    total = MatchResult()
    if executor is not None:
        for r in executor.map(_play_pairs, chunks):
            total = total + r
        return total
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as ex:
        for r in ex.map(_play_pairs, chunks):
            total = total + r
    return total


# ---------------------------------------------------------------------------
# Fixed evaluation sets (training runs)

def plain_team(team) -> list:
    """A team as plain JSON data (lists / dicts / numbers), e.g. to store it in a checkpoint."""
    return json.loads(json.dumps(team))


def make_eval_pairs(n_pairs: int, formatid: str = FORMAT_SINGLES, seed: int = 0, team_source=None) -> list:
    """``n_pairs`` seeded team pairs ``{'teams': [team1, team2], 'seed': battle_seed}`` (plain data).

    Every pair is played twice (teams swapped) by :func:`play_pairs`, so ``2 * n_pairs`` games.
    """
    rng = random.Random(seed)
    gen = RandomTeamGenerator(rng.randrange(1 << 30), formatid=formatid)
    out = []
    for _ in range(max(0, n_pairs)):
        t1, t2 = team_source(rng) if team_source is not None else (gen.random_team(), gen.random_team())
        out.append({'teams': [plain_team(t1), plain_team(t2)], 'seed': gen5_seed(rng)})
    return out


def play_pairs(agent_a, agent_b, pairs: list, formatid: str = FORMAT_SINGLES, max_turns: int = 200) -> MatchResult:
    """Play fixed pairs from :func:`make_eval_pairs`: agent A is p1, each pair with both team assignments."""
    res = MatchResult()
    for pair in pairs:
        t1, t2 = pair['teams']
        _play_pair(t1, t2, agent_a, agent_b, formatid, pair['seed'], res, max_turns)
    return res


def combined_score(results: list) -> tuple:
    """Mean win rate over several match-ups and its 95% confidence half-width -> ``(score, ci95)``."""
    results = [r for r in results if r is not None and r.games]
    if not results:
        return 0.0, 0.0
    k = len(results)
    score = sum(r.win_rate for r in results) / k
    var = sum(max(r.win_rate * (1 - r.win_rate), 1e-9) / r.games for r in results) / (k * k)
    return score, 1.96 * math.sqrt(var)
