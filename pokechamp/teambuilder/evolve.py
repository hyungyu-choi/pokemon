"""Evolutionary team building: start from completely random legal teams and evolve them.

Every gene starts uniformly random - species, ability, held item, the four moves,
nature and the Stat Point spread.  Each generation the teams battle each other
(and a hall of fame of earlier champions, plus a few fresh random teams) with a
battle agent; the win rate is the fitness.  Selection keeps the best teams and
refills the population with mutated / recombined children.  Mutations are biased
by what has worked so far: moves, items, abilities and natures that appear on
winning teams are proposed more often (an estimation-of-distribution step).

All produced teams are legal (checked with :class:`TeamValidator`).
"""
from __future__ import annotations

import copy
import json
import multiprocessing
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

from ..env.runner import FORMAT_SINGLES, gen5_seed, play_battle
from ..sim.js import Obj
from ..sim.teams import export_team
from .random_teams import SP_MAX, SP_TOTAL, STATS, RandomTeamGenerator, legal_pool
from .validator import TeamValidator


@dataclass
class EvolveConfig:
    formatid: str = FORMAT_SINGLES
    out: str = 'runs/teams'
    population: int = 32
    generations: int = 50
    games_per_team: int = 24          # games played per team per generation (half as each side)
    hall_of_fame: int = 8
    elite: int = 6
    random_opponents: int = 4         # fresh random teams per generation (keeps fitness grounded)
    mutation_rate: float = 0.3        # per-set probability of a mutation
    crossover_rate: float = 0.3
    workers: int = 3
    agent: str = 'heuristic'          # 'heuristic' or a path to a policy checkpoint (.pt)
    seed: int = 0
    max_turns: int = 120


def team_key(team) -> str:
    return json.dumps([[s['species'], s['item'], s['ability'], sorted(s['moves']), s['nature'],
                        [s['evs'][k] for k in STATS]] for s in team], sort_keys=True)


def _move_prior(species, move_name, set_=None) -> float:
    """Weak prior for proposing a move: strong, accurate, STAB attacks and well-known utility."""
    from ..sim.dex import get_dex
    dex = get_dex()
    m = dex.moves.get(move_name)
    if m.category == 'Status':
        return 1.0 if m.id in _UTILITY else 0.25
    sp = dex.species.get(species)
    bp = min(m.basePower or 60, 150) * (3 if m.multihit else 1)
    acc = 1.0 if m.accuracy is True else m.accuracy / 100
    stab = 1.5 if m.type in sp.types else 1.0
    w = (bp * acc * stab / 90) ** 2
    if set_ is not None:
        evs = set_.get('evs') or {}
        physical = evs.get('atk', 0) >= evs.get('spa', 0)
        if (m.category == 'Physical') != physical:
            w *= 0.5
    if m.flags.get('recharge') or (m.flags.get('charge') and m.id != 'solarbeam'):
        w *= 0.3
    return 0.2 + w


_UTILITY = {
    'protect', 'detect', 'swordsdance', 'nastyplot', 'calmmind', 'dragondance', 'quiverdance', 'bulkup', 'recover',
    'roost', 'slackoff', 'shoreup', 'moonlight', 'synthesis', 'willowisp', 'thunderwave', 'toxic', 'spore',
    'sleeppowder', 'yawn', 'stealthrock', 'spikes', 'taunt', 'trickroom', 'tailwind', 'encore', 'substitute',
    'leechseed', 'reflect', 'lightscreen', 'auroraveil', 'haze', 'partingshot', 'followme', 'ragepowder',
    'helpinghand', 'wideguard', 'irondefense', 'shellsmash', 'bellydrum', 'agility', 'kingsshield', 'spikyshield',
    'banefulbunker', 'stickyweb', 'trick', 'painsplit', 'wish', 'strengthsap', 'glare', 'coil', 'fakeout',
    'uturn', 'voltswitch', 'flipturn', 'knockoff', 'suckerpunch', 'extremespeed', 'aquajet', 'machpunch',
}


def _make_coherent(s, rng):
    """Align nature and Stat Points with the set's attacks (a common, sensible variation)."""
    from ..sim.dex import get_dex
    dex = get_dex()
    cats = [dex.moves.get(m).category for m in s['moves']]
    phys, spec = cats.count('Physical'), cats.count('Special')
    physical = phys >= spec
    atk = 'atk' if physical else 'spa'
    fast = rng.random() < 0.6
    second = 'spe' if fast else 'hp'
    evs = {k: 0 for k in STATS}
    evs[atk] = 32
    evs[second] = 32
    evs[rng.choice([k for k in STATS if k not in (atk, second)])] = 2
    s['evs'] = evs
    if physical:
        s['nature'] = 'Jolly' if fast and rng.random() < 0.5 else 'Adamant'
    else:
        s['nature'] = 'Timid' if fast and rng.random() < 0.5 else 'Modest'
    return s


def _plain(team):
    return [{'species': s['species'], 'name': s.get('name') or s['species'], 'item': s['item'],
             'ability': s['ability'], 'moves': list(s['moves']), 'nature': s['nature'],
             'evs': {k: int(s['evs'][k]) for k in STATS}, 'level': 50} for s in team]


# ---------------------------------------------------------------------------
# Agents in worker processes

def _make_agent(spec: str, seed: int):
    if spec == 'heuristic':
        from ..ai.heuristic import HeuristicAgent
        return HeuristicAgent(seed)
    if spec == 'maxdamage':
        from ..ai.heuristic import MaxDamageAgent
        return MaxDamageAgent(seed)
    import torch
    from ..ai.model import load_model
    from ..ai.nn_agent import NNAgent
    torch.set_num_threads(1)
    agent = NNAgent(load_model(spec), sample=False)
    agent.seed(seed)
    return agent


_AGENT = {}


def _play_pairs_job(args):
    spec, formatid, pairs, max_turns = args
    if spec not in _AGENT:
        _AGENT.clear()
        _AGENT[spec] = (_make_agent(spec, 1), _make_agent(spec, 2))
    a1, a2 = _AGENT[spec]
    out = []
    for i, j, ti, tj, seed in pairs:
        rng = random.Random(seed)
        bseed = gen5_seed(rng)
        res = []
        for swap in (False, True):
            t1, t2 = (tj, ti) if swap else (ti, tj)
            r = play_battle(Obj_team(t1), Obj_team(t2), a1, a2, formatid, seed=list(bseed), max_turns=max_turns)
            if r.winner is None:
                res.append(0.5)
            else:
                first_won = (r.winner == 'p1')
                res.append(1.0 if first_won != swap else 0.0)
        out.append((i, j, res))
    return out


def Obj_team(team):
    return [Obj({**s, 'evs': Obj(s['evs'])}) for s in team]


# ---------------------------------------------------------------------------
# Evolution

class TeamEvolver:
    def __init__(self, cfg: EvolveConfig, log=print):
        self.cfg = cfg
        self.log = log
        self.rng = random.Random(cfg.seed)
        self.gen = RandomTeamGenerator(self.rng.randrange(1 << 30), formatid=cfg.formatid)
        self.pool = legal_pool(None, cfg.formatid)
        self.validator = TeamValidator(cfg.formatid)
        self.population = [_plain(self.gen.random_team()) for _ in range(cfg.population)]
        self.fitness = [0.5] * cfg.population
        self.hof: list = []
        self.generation = 0
        # learned preferences: (species, gene) -> [wins, games]
        self.gene_stats = defaultdict(lambda: [0.0, 0.0])
        self.species_stats = defaultdict(lambda: [0.0, 0.0])

    # -- variation -------------------------------------------------------------
    def _weighted(self, species, kind, options, team_set=None):
        """Pick among ``options`` with weights from past success (optimistic for untried genes).

        For moves the learned score is multiplied by a weak domain prior (power, accuracy, STAB,
        matching the set's attacking stat), which only speeds up the search - selection decides.
        """
        ws = []
        for o in options:
            w, n = self.gene_stats[(species, kind, o)]
            score = ((w + 1.0) / (n + 2.0)) ** 3 + 0.02
            if kind == 'move':
                score *= _move_prior(species, o, team_set)
            ws.append(score)
        return self.rng.choices(options, ws)[0]

    def mutate_set(self, s, used_items: set):
        rng = self.rng
        species = s['species']
        pool = self.pool
        r = rng.random()
        if r < 0.40:
            # replace one move
            moves = list(s['moves'])
            candidates = [m for m in pool.moves[species] if m not in moves]
            if candidates:
                k = rng.randrange(len(moves)) if len(moves) >= 4 else len(moves)
                new = self._weighted(species, 'move', candidates, s)
                if k < len(moves):
                    moves[k] = new
                else:
                    moves.append(new)
                s['moves'] = moves
        elif r < 0.55:
            banned = pool.meta['excluded_items'].get(species, set())
            stones = [x for x in pool.mega_stones.get(species, []) if x not in used_items and x not in banned]
            options = [i for i in pool.items if i not in used_items and i not in banned] + stones
            if options:
                used_items.discard(s['item'])
                s['item'] = self._weighted(species, 'item', options)
                used_items.add(s['item'])
        elif r < 0.65:
            s['ability'] = self._weighted(species, 'ability', pool.abilities[species])
        elif r < 0.72:
            s['nature'] = self._weighted(species, 'nature', pool.natures)
        elif r < 0.82:
            _make_coherent(s, rng)
        else:
            # move Stat Points between stats
            evs = dict(s['evs'])
            for _ in range(rng.randint(1, 3)):
                src = rng.choice([k for k in STATS if evs[k] > 0] or list(STATS))
                dst = rng.choice([k for k in STATS if evs[k] < SP_MAX] or list(STATS))
                amount = min(evs[src], SP_MAX - evs[dst], rng.choice((2, 4, 8, 16)))
                evs[src] -= amount
                evs[dst] += amount
            spare = SP_TOTAL - sum(evs.values())
            while spare > 0:
                k = rng.choice([k for k in STATS if evs[k] < SP_MAX] or [None])
                if k is None:
                    break
                add = min(spare, SP_MAX - evs[k])
                evs[k] += add
                spare -= add
            s['evs'] = evs
        return s

    def new_member(self, team, used_items):
        """Replace one Pokemon of ``team`` with a new species (biased towards successful species)."""
        rng = self.rng
        nums = self.pool.meta['num']
        used_nums = {nums[s['species']] for s in team}
        options = [sp for sp in self.pool.species if nums[sp] not in used_nums]
        ws = []
        for sp in options:
            w, n = self.species_stats[sp]
            ws.append(((w + 1.0) / (n + 2.0)) ** 2 + 0.05)
        species = rng.choices(options, ws)[0]
        new = self.gen.random_set(species, used_items)
        return _plain([new])[0]

    def child(self, parents):
        rng = self.rng
        if len(parents) == 2 and rng.random() < self.cfg.crossover_rate:
            a, b = parents
            mons = copy.deepcopy(a + b)
            rng.shuffle(mons)
            team, nums, items = [], set(), set()
            numtab = self.pool.meta['num']
            for s in mons:
                if numtab[s['species']] in nums:
                    continue
                if s['item'] and s['item'] in items:
                    continue
                team.append(s)
                nums.add(numtab[s['species']])
                if s['item']:
                    items.add(s['item'])
                if len(team) == 6:
                    break
            while len(team) < 6:
                team.append(self.new_member(team, items))
        else:
            team = copy.deepcopy(parents[0])
        used_items = {s['item'] for s in team if s['item']}
        mutated = False
        for k in range(len(team)):
            if rng.random() < self.cfg.mutation_rate:
                if rng.random() < 0.12:
                    used_items.discard(team[k]['item'])
                    rest = team[:k] + team[k + 1:]
                    team[k] = self.new_member(rest, used_items)
                else:
                    self.mutate_set(team[k], used_items)
                mutated = True
        if not mutated:
            self.mutate_set(team[rng.randrange(len(team))], used_items)
        for s in team:
            s['name'] = s['species']
        if self.validator.validate_team(Obj_team(team)):
            return copy.deepcopy(parents[0])
        return team

    # -- evaluation ------------------------------------------------------------
    def evaluate(self, ex):
        cfg = self.cfg
        rng = self.rng
        teams = list(self.population)
        n_pop = len(teams)
        opponents = list(self.hof) + [_plain(self.gen.random_team()) for _ in range(cfg.random_opponents)]
        all_teams = teams + opponents
        pairs = []
        per_team = max(1, cfg.games_per_team // 2)
        for i in range(n_pop):
            for _ in range(per_team):
                j = rng.randrange(len(all_teams) - 1)
                if j >= i:
                    j += 1
                pairs.append((i, j, all_teams[i], all_teams[j], rng.randrange(1 << 30)))
        chunks = [pairs[k::cfg.workers] for k in range(cfg.workers)]
        jobs = [(cfg.agent, cfg.formatid, c, cfg.max_turns) for c in chunks if c]
        score = [0.0] * len(all_teams)
        games = [0] * len(all_teams)
        results = ex.map(_play_pairs_job, jobs) if ex is not None else map(_play_pairs_job, jobs)
        for out in results:
            for i, j, res in out:
                for r in res:
                    score[i] += r
                    games[i] += 1
                    score[j] += 1 - r
                    games[j] += 1
        fitness = []
        for i in range(n_pop):
            # Bayesian-smoothed win rate
            fitness.append((score[i] + 1) / (games[i] + 2))
        # credit genes / species of each team with its results
        for i, team in enumerate(teams):
            w, n = score[i], games[i]
            for s in team:
                sp = s['species']
                self.species_stats[sp][0] += w / 6
                self.species_stats[sp][1] += n / 6
                for m in s['moves']:
                    self.gene_stats[(sp, 'move', m)][0] += w / 6
                    self.gene_stats[(sp, 'move', m)][1] += n / 6
                for kind in ('item', 'ability', 'nature'):
                    self.gene_stats[(sp, kind, s[kind])][0] += w / 6
                    self.gene_stats[(sp, kind, s[kind])][1] += n / 6
        hof_scores = [(score[n_pop + k] + 1) / (games[n_pop + k] + 2) for k in range(len(self.hof))]
        return fitness, hof_scores

    def step(self, ex=None):
        cfg = self.cfg
        t = time.time()
        self.fitness, hof_scores = self.evaluate(ex)
        order = sorted(range(len(self.population)), key=lambda i: -self.fitness[i])
        best = self.population[order[0]]
        # hall of fame: best team of each generation (deduplicated)
        keys = {team_key(t_) for t_ in self.hof}
        if team_key(best) not in keys:
            self.hof.append(copy.deepcopy(best))
            self.hof = self.hof[-cfg.hall_of_fame:]
        elite = [self.population[i] for i in order[:cfg.elite]]
        new_pop = [copy.deepcopy(t_) for t_ in elite]
        while len(new_pop) < cfg.population:
            parents = [self._tournament(order), self._tournament(order)]
            new_pop.append(self.child(parents))
        prev_fitness = [self.fitness[i] for i in order]
        self.population = new_pop
        self.generation += 1
        mean = sum(prev_fitness) / len(prev_fitness)
        self.log(f"[gen {self.generation}] best {prev_fitness[0]:.3f} mean {mean:.3f} "
                 f"hof {', '.join(f'{x:.2f}' for x in hof_scores) or '-'} | best team: "
                 f"{', '.join(s['species'] for s in best)} ({time.time() - t:.0f}s)")
        return best, prev_fitness[0]

    def _tournament(self, order, k=3):
        idx = [self.rng.randrange(len(order)) for _ in range(k)]
        return self.population[order[min(idx)]]

    # -- persistence -----------------------------------------------------------
    def save(self, path_dir):
        os.makedirs(path_dir, exist_ok=True)
        ranked = sorted(zip(self.fitness, self.population), key=lambda x: -x[0])
        data = {'format': self.cfg.formatid, 'generation': self.generation,
                'teams': [t for _, t in ranked], 'fitness': [f for f, _ in ranked],
                'hall_of_fame': self.hof, 'config': asdict(self.cfg)}
        with open(os.path.join(path_dir, 'population.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        with open(os.path.join(path_dir, 'best_team.txt'), 'w', encoding='utf-8') as f:
            f.write(export_team([Obj(s) for s in (self.hof[-1] if self.hof else ranked[0][1])]))
        with open(os.path.join(path_dir, 'usage.json'), 'w', encoding='utf-8') as f:
            json.dump(self.usage(), f, ensure_ascii=False, indent=1)

    def usage(self):
        """Learned set statistics: which species / moves / items did best (by win rate)."""
        out = {}
        for sp, (w, n) in sorted(self.species_stats.items(), key=lambda kv: -kv[1][1]):
            if n < 4:
                continue
            genes = defaultdict(list)
            for (s, kind, g), (gw, gn) in self.gene_stats.items():
                if s == sp and gn >= 2:
                    genes[kind].append((g, round((gw + 1) / (gn + 2), 3), round(gn, 1)))
            out[sp] = {'win_rate': round((w + 1) / (n + 2), 3), 'games': round(n, 1),
                       **{k: sorted(v, key=lambda x: -x[1])[:8] for k, v in genes.items()}}
        return out


def evolve(cfg: EvolveConfig, log=print, resume: str | None = None) -> TeamEvolver:
    evo = TeamEvolver(cfg, log)
    if resume and os.path.exists(resume):
        with open(resume, encoding='utf-8') as f:
            data = json.load(f)
        evo.population = data['teams'][:cfg.population]
        while len(evo.population) < cfg.population:
            evo.population.append(_plain(evo.gen.random_team()))
        evo.hof = data.get('hall_of_fame', [])
        evo.generation = data.get('generation', 0)
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=ctx) as ex:
        for _ in range(cfg.generations):
            evo.step(ex if cfg.workers > 1 else None)
            evo.save(cfg.out)
    return evo


def benchmark_team(team, formatid=FORMAT_SINGLES, n_opponents=50, agent='heuristic', seed=0, workers=3):
    """Win rate of ``team`` against fresh random teams (both sides use the same agent)."""
    rng = random.Random(seed)
    gen = RandomTeamGenerator(rng.randrange(1 << 30), formatid=formatid)
    pairs = [(0, 1, team, _plain(gen.random_team()), rng.randrange(1 << 30)) for _ in range(n_opponents)]
    chunks = [pairs[k::workers] for k in range(workers)]
    jobs = [(agent, formatid, c, 120) for c in chunks if c]
    wins, games = 0.0, 0
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for out in ex.map(_play_pairs_job, jobs):
            for _i, _j, res in out:
                wins += sum(res)
                games += len(res)
    return wins / max(1, games), games


__all__ = ['EvolveConfig', 'TeamEvolver', 'evolve', 'benchmark_team']
