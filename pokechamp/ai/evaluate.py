"""Head-to-head evaluation of agents (paired games with swapped teams, optional multiprocessing)."""
from __future__ import annotations

import math
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

    def __add__(self, other: 'MatchResult') -> 'MatchResult':
        return MatchResult(self.wins + other.wins, self.losses + other.losses, self.ties + other.ties,
                           self.errors + other.errors, self.turns + other.turns)

    def __str__(self) -> str:
        return (f"{self.win_rate * 100:5.1f}% ± {self.ci95() * 100:.1f}  "
                f"(W {self.wins} / L {self.losses} / T {self.ties}, {self.games} games, "
                f"{self.turns / max(1, self.games):.1f} turns/game, errors {self.errors})")


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
        for swap in (False, True):
            ta, tb = (t2, t1) if swap else (t1, t2)
            r = play_battle(ta, tb, agent_a, agent_b, formatid, seed=list(battle_seed))
            res.turns += r.turns
            if r.error:
                res.errors += 1
            if r.winner == 'p1':
                res.wins += 1
            elif r.winner == 'p2':
                res.losses += 1
            else:
                res.ties += 1
    return res


def evaluate(factory_a, factory_b, n_games: int = 100, formatid: str = FORMAT_SINGLES, seed: int = 0,
             workers: int = 1, team_source=None) -> MatchResult:
    """Win rate of agent A against agent B.

    ``factory_x(seed) -> agent`` must be picklable (a module-level function or class) when
    ``workers > 1``.  Each pair of teams is played twice with the sides swapped, so team
    strength cancels out.  ``team_source(rng) -> (team1, team2)`` overrides random teams.
    """
    n_pairs = max(1, n_games // 2)
    if workers <= 1:
        return _play_pairs((factory_a, factory_b, n_pairs, formatid, seed, team_source))
    chunks = []
    per = math.ceil(n_pairs / workers)
    for w in range(workers):
        k = min(per, n_pairs - w * per)
        if k > 0:
            chunks.append((factory_a, factory_b, k, formatid, seed * 1000 + w, team_source))
    total = MatchResult()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_play_pairs, chunks):
            total = total + r
    return total
