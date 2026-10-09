"""Training of the battle policy: behaviour cloning from the heuristic agent, then PPO self-play.

Usage (see ``python -m pokechamp train --help``)::

    # fresh model: behaviour cloning, then PPO (CPU or GPU)
    python -m pokechamp train --out runs/singles --bc-games 3000 --iters 200
    # long cloud / GPU run that continues from the shipped model; resumable, stops before the time budget
    python -m pokechamp train --preset cloud --out runs/cloud --time-budget-h 11.5 --sync-dir /content/drive/MyDrive/pokechamp

Rollouts run in worker processes (one CPU thread each, CPU inference); the learner updates the network
between rollout rounds on ``cfg.device`` (``auto`` = CUDA when available).  With ``vectorized`` (the default)
every worker keeps ``envs_per_worker`` games in play and decides all their pending network moves with one
batched forward pass per model (:mod:`pokechamp.ai.vec_rollout`); an iteration is split into up to
``jobs_per_worker`` jobs per worker that the pool hands out as workers become free, the weights are written once
per iteration (``out/_weights_N.pt``, removed afterwards) and workers cache them, and observations come back as
float16 (the actors act on the same rounded values).  ``vectorized=False`` keeps the one-game-at-a-time actors
(:func:`_rollout_job`) for comparison.  ``python -m pokechamp.tools.bench_rollout`` measures both.  Opponents come
from a league: the
current policy (self-play, both sides learn), earlier snapshots chosen by prioritised fictitious self-play
(snapshots we lose to are picked more often; the warm-start / behaviour-cloning models stay in the pool as
anchors) and the heuristic agent.  Teams are a mixture of random legal teams, the evolved team pool and
realistic sets built from the usage statistics.

Rewards and the two value heads
-------------------------------
The reward is the final result ``z`` (+1 win / -1 loss / 0 tie or turn limit, from the learner's side) plus
potential-based shaping on the HP balance ``phi = shaping.potential(view)`` (does not change the optimal
policy).  The shaped return from step t is exactly ``gamma**(T-t) * z - shaping * phi_t``.

* ``value`` head (``forward()[1]``): fitted to the *unshaped* outcome ``z`` at every recorded step (behaviour
  cloning and PPO), so ``0.5 * (value + 1)`` is a win probability for the advisor and the UI.
* ``critic`` head (``model.critic(x)``, when ``config['critic']``): fitted to the GAE(lambda) return of the
  shaped rewards and used as the PPO baseline.  Models without it use ``value - shaping * phi`` instead,
  which is the same quantity for ``gamma = 1``.

GAE baselines are recomputed by the learner (one batched no-grad pass over the iteration's samples), so
actors only need the policy.

Trajectory record format (what an actor must produce)
-----------------------------------------------------
A rollout job returns ``(trajectories, stats)``.  ``trajectories`` holds one entry per learner-controlled side
per game (self-play games give two; their order does not matter)::

    {'z': float,                 # final result from this side: +1 win, -1 loss, 0 tie / turn limit / error
                                 #   (a truncated game is treated as terminal with z = 0, no bootstrapping)
     'steps': [step, ...]}       # this side's decisions in battle order (team preview first); GAE runs
                                 #   backwards over this list, so it must be complete and in order

    step = {
      'obs':    dict,            # features.encode(view, request), numpy arrays (see obs_spec()):
                                 #   ids [12,3] int64, mon [12,MON_F] f32, move_ids [12,4] int64,
                                 #   move [12,4,MOVE_F] f32, glob [GLOB_F] f32, active_idx [2] int64
      'legal':  list[tuple],     # the options the agent chose from, in order: move decisions = joint
                                 #   slot actions, tuples of 2 ints (env.actions, -1 = pass); team preview =
                                 #   tuples of up to 4 team indices (brought Pokemon, leads first)
      'idx':    int,             # index of the chosen option in 'legal'
      'kind':   int,             # 0 move decision, 1 team preview
      'n_lead': int,             # active slots (view.n_active): the first n_lead entries of a preview option
                                 #   are the leads (required for team preview)
      'phi':    float,           # shaping.potential(view) of this side at this decision (HP balance, -1..1)
      'logp':   float,           # log-probability of the chosen option under the policy that acted, i.e. the
                                 #   weights shipped for this iteration, sampling at temperature 1 over
                                 #   'legal' (PPO only; behaviour-cloning records omit it)
      'value':  float,           # optional, ignored: the learner recomputes critic values itself
    }

    stats = {'games': int, 'turns': int, 'errors': int,
             'by_opp':  {'heuristic' | 'snapshot': [score, games]},   # learner score: win 1, tie 0.5
             'by_snap': {snapshot path: [score, games]},               # results per snapshot (PFSP)
             'teams':   {source: count}}

:func:`check_trajectory` validates one record against this contract (the learner checks a sample of every
iteration); :class:`StepBatch` collates a list of records into device tensors once per iteration.  Vectorised
actors may send a record compressed (:func:`vec_rollout.compress_trajectory`: the steps without ``obs`` plus
``'packed'`` arrays ``[T, ...]`` in float16 / small ints); the learner restores the format above with
:func:`vec_rollout.decompress_trajectory` before checking or collating.  Their job stats add ``decisions``,
``nn_decisions``, ``forwards``, ``nn_seconds``, ``job_seconds`` and ``jobs`` (throughput in the iteration log).

Files in ``cfg.out``
--------------------
``state.pt`` (everything needed to resume: model, optimizer, iteration, counters, snapshot pool with
results, fixed evaluation set, RNG states, config), ``latest.pt`` / ``best.pt`` / ``final.pt`` (models for
``load_model``), ``iterNNNN.pt`` (rolling snapshots), ``init.pt`` / ``bc.pt`` (anchors), ``incumbent.pt``
(the model evaluations compare against), ``log.jsonl`` (one JSON line per iteration / evaluation),
``train.log`` (console output) and ``config.json``.  Everything is written atomically; with ``sync_dir``
the files are mirrored there after every checkpoint, and a run whose ``out`` is gone (Colab VM reset)
resumes from the mirror.
"""
from __future__ import annotations

import io
import json
import math
import multiprocessing
import os
import random
import re
import shutil
import signal
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np
import torch
import torch.nn.functional as F

from ..env.runner import AgentChoice, FORMAT_SINGLES, gen5_seed, play_battle
from ..teambuilder.random_teams import RandomTeamGenerator
from ..teambuilder.validator import TeamValidator
from .evaluate import MatchResult, combined_score, make_eval_pairs, plain_team, play_pairs
from .features import encode
from .heuristic import HeuristicAgent
from .model import (PolicyValueNet, atomic_save, checkpoint_dict, collate_obs, cpu_state_dict, joint_logits,
                    load_model, model_from_checkpoint, preview_logits, save_model)
from .nn_agent import NNAgent, potential
from . import vec_rollout

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
DEFAULT_TEAM_MIX = {'random': 0.5, 'pool': 0.25, 'usage': 0.25}
TEAM_SOURCES = ('random', 'pool', 'usage')
OPPONENT_KINDS = ('self', 'snapshot', 'heuristic')
STATE_VERSION = 1


@dataclass
class TrainConfig:
    formatid: str = FORMAT_SINGLES
    out: str = 'runs/battle'
    workers: int | str = 3                # rollout processes; 'auto' = os.cpu_count()
    device: str = 'auto'                  # learner device: 'auto' (CUDA if available), 'cpu' or 'cuda'
    init: str | None = None               # warm start from this checkpoint (no behaviour cloning)
    # model (a warm start keeps the checkpoint's architecture)
    d: int = 128
    layers: int = 2
    heads: int = 4
    critic: bool = True                   # separate shaped critic head (the value head stays P(win))
    # behaviour cloning (fresh models only)
    bc_games: int = 2000
    bc_epochs: int = 4
    bc_lr: float = 1e-3
    # PPO
    iters: int = 100
    games_per_iter: int = 96
    lr: float = 3e-4
    lr_final: float | None = None         # linear decay to this over `iters` (None: constant)
    ppo_epochs: int = 4
    minibatch: int | None = None          # None: 2048 on CUDA, 512 on CPU
    clip: float = 0.2
    target_kl: float | None = 0.03        # stop an update early once approx. KL > 1.5 * target_kl
    vf_coef: float = 0.5                  # critic loss (shaped return)
    value_coef: float = 0.5               # value head loss (outcome z)
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    lam: float = 0.95
    shaping: float = 1.0
    shaping_final: float | None = None    # anneal the shaping weight linearly to this ...
    shaping_anneal_iters: int = 0         # ... over this many iterations (0: no annealing)
    # league
    opponents: dict = field(default_factory=lambda: {'self': 0.5, 'snapshot': 0.35, 'heuristic': 0.15})
    heuristic_randomness: float = 0.05
    snapshot_every: int = 5
    snapshot_keep: int = 20               # rolling iterNNNN.pt snapshots kept (<= 0: all); anchors always
    pfsp_floor: float = 0.05              # minimum snapshot weight; weight = (1 - our score vs it) ** 2
    pfsp_decay: float = 0.9               # per-iteration decay of the recorded results vs each snapshot
    # teams
    team_pool: str | None = 'auto'        # JSON of teams ('auto': models/teams_singles.json for singles)
    team_pool_prob: float = 0.5           # legacy: P(pool team) when team_pool is a path and team_mix is None
    usage_stats: str | None = 'auto'      # usage statistics ('auto': models/usage_singles.json for singles)
    team_mix: dict | None = None          # source weights; None: random 0.5 / pool 0.25 / usage 0.25
    # evaluation (fixed seeded team pairs, both team assignments per pair)
    eval_every: int = 10
    eval_games: int = 400                 # per opponent: greedy policy vs HeuristicAgent(0) and vs incumbent
    incumbent: str | None = 'auto'        # model to beat ('auto': models/battle_singles.pt for singles)
    # actors (rollout workers)
    vectorized: bool = True               # each worker plays envs_per_worker games at once with one batched
                                          #   forward per step (False: one game at a time, the old path)
    envs_per_worker: int = 48             # games a worker keeps in play at once (vectorized)
    jobs_per_worker: int = 2              # rollout jobs per worker and iteration (balances uneven jobs)
    compress_obs: bool = True             # float16 observations from the workers (actors act on the same)
    snapshots_per_job: int = 1            # snapshot models one rollout job plays against (bigger batches)
    # run control
    time_budget_h: float | None = None    # stop in time for the final evaluation and save (None: no limit)
    checkpoint_minutes: float = 10.0
    resume: str = 'auto'                  # 'auto': continue from out/state.pt or sync_dir/state.pt; 'never'
    sync_dir: str | None = None           # mirror checkpoints and logs here (e.g. Google Drive)
    log_file: str | None = 'train.log'    # copy of the console output in `out`
    seed: int = 0
    max_turns: int = 150


PRESETS = {
    # `python -m pokechamp train` without a preset: a fresh model, behaviour cloning, then PPO
    'default': {'workers': 'auto', 'bc_games': 3000, 'iters': 150, 'eval_every': 10},
    # long runs on a cloud GPU (Colab, Kaggle) or your PC: continue from the shipped model; the time budget
    # (or Ctrl-C / SIGTERM) decides when to stop, `--resume auto` continues in the next session
    'cloud': {'workers': 'auto', 'init': 'models/battle_singles.pt', 'bc_games': 0, 'critic': True,
              'games_per_iter': 512, 'iters': 100000, 'eval_every': 10, 'eval_games': 400,
              'snapshot_every': 5, 'checkpoint_minutes': 10},
}


def load_config_arg(config) -> dict:
    """``--config``: a dict, an inline JSON object or the path of a JSON file."""
    if isinstance(config, dict):
        return dict(config)
    text = str(config).strip()
    if not text.startswith('{'):
        with open(text, encoding='utf-8-sig') as f:
            text = f.read()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError('the training config must be a JSON object of TrainConfig fields')
    return data


def build_config(preset: str = 'default', config=None, overrides: dict | None = None) -> TrainConfig:
    """TrainConfig defaults < preset < ``config`` (dict / JSON / JSON file) < ``overrides`` (CLI flags)."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r} (choose from {', '.join(PRESETS)})")
    values = dict(PRESETS[preset])
    if config:
        values.update(load_config_arg(config))
    values.update(overrides or {})
    names = {f.name for f in fields(TrainConfig)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f"unknown training option(s): {', '.join(unknown)}; valid: {', '.join(sorted(names))}")
    cfg = TrainConfig(**values)
    validate_config(cfg)
    return cfg


def _check_weights(name: str, weights, allowed):
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f'{name} must be a non-empty JSON object of weights, e.g. {{"{allowed[0]}": 1.0}}')
    unknown = sorted(set(weights) - set(allowed))
    if unknown:
        raise ValueError(f"{name}: unknown key(s) {', '.join(unknown)} (use {', '.join(allowed)})")
    try:
        values = [float(w) for w in weights.values()]
    except (TypeError, ValueError):
        raise ValueError(f'{name}: weights must be numbers') from None
    if min(values) < 0 or sum(values) <= 0:
        raise ValueError(f'{name}: weights must be >= 0 with a positive sum')


def validate_config(cfg: TrainConfig):
    """Reject settings that would otherwise fail late or silently do something else (e.g. a misspelt
    opponent kind).  Raises ValueError with a readable message."""
    _check_weights('opponents', cfg.opponents, OPPONENT_KINDS)
    if cfg.team_mix is not None:
        _check_weights('team_mix', cfg.team_mix, TEAM_SOURCES)
    if cfg.resume not in ('auto', 'never'):
        raise ValueError(f"resume must be 'auto' or 'never', not {cfg.resume!r}")
    if str(cfg.device or 'auto').lower().split(':')[0] not in ('auto', 'cpu', 'cuda'):
        raise ValueError(f"device must be 'auto', 'cpu' or 'cuda', not {cfg.device!r}")
    if cfg.games_per_iter < 1 or cfg.iters < 0 or cfg.ppo_epochs < 1:
        raise ValueError('games_per_iter and ppo_epochs must be >= 1, iters >= 0')
    if cfg.time_budget_h is not None and cfg.time_budget_h <= 0:
        raise ValueError('time_budget_h must be positive (or null for no limit)')
    if cfg.envs_per_worker < 1 or cfg.jobs_per_worker < 1 or cfg.snapshots_per_job < 1:
        raise ValueError('envs_per_worker, jobs_per_worker and snapshots_per_job must be >= 1')


def repo_path(path: str | None) -> str | None:
    """``path`` if it exists (relative to the working directory) or is absolute, else relative to the project."""
    if not path or os.path.isabs(path) or os.path.exists(path):
        return path
    alt = os.path.join(REPO_ROOT, path)
    return alt if os.path.exists(alt) else path


def _auto_path(value, formatid: str, filename: str):
    if value == 'auto':
        path = os.path.join(MODELS_DIR, filename)
        return path if formatid == FORMAT_SINGLES and os.path.exists(path) else None
    return os.path.abspath(repo_path(value)) if value else None


def resolve_workers(workers) -> int:
    if workers in (None, 'auto'):
        return max(1, os.cpu_count() or 1)
    return max(1, int(workers))


def resolve_device(name: str | None = 'auto') -> torch.device:
    name = (name or 'auto').lower()
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('device cuda was requested but PyTorch sees no CUDA GPU '
                           '(Colab: Runtime > Change runtime type > T4 GPU; or use --device cpu)')
    return torch.device(name)


# ---------------------------------------------------------------------------
# Team sources

class UsageTeamBuilder:
    """Realistic teams from the usage statistics of the team evolver (``models/usage_singles.json``).

    Species are drawn by how often they were used (square root, so the pool stays varied; Species Clause),
    moves / held item / ability / nature by usage counts; the rest of each set (Stat Points, missing moves)
    comes from the advisor's :class:`SetPrior` (which also prefers matching sets of the team library).
    """

    def __init__(self, formatid: str, usage_path: str, library_path: str | None = None):
        from .advisor import SetPrior  # torch-free
        with open(usage_path, encoding='utf-8-sig') as f:
            usage = json.load(f)
        self.prior = SetPrior(formatid, library_path)
        self.pool = self.prior.pool
        self.usage = {sp: info for sp, info in usage.items() if isinstance(info, dict) and sp in self.pool.moves}
        if len(self.usage) < 6:
            raise ValueError(f'{usage_path}: not enough usable species')
        self.species = list(self.usage)
        self.weights = [math.sqrt(max(1.0, float(self.usage[s].get('games', 1)))) for s in self.species]

    @staticmethod
    def _weighted(rng, entries, allowed, exclude=()):
        cands = [(e[0], float(e[2]) if len(e) > 2 else 1.0) for e in entries or []
                 if e and e[0] in allowed and e[0] not in exclude]
        if not cands:
            return None
        return rng.choices([c[0] for c in cands], [max(c[1], 1e-3) for c in cands])[0]

    def make_set(self, species: str, rng: random.Random, used_items: set) -> dict:
        from .advisor import MonState
        info = self.usage[species]
        legal_moves = set(self.pool.moves.get(species, []))
        entries = [e for e in info.get('move', []) if e and e[0] in legal_moves]
        moves = []
        want = rng.randint(2, 4)
        while entries and len(moves) < want:
            m = self._weighted(rng, entries, legal_moves)
            moves.append(m)
            entries = [e for e in entries if e[0] != m]
        banned = set(self.pool.meta['excluded_items'].get(species, set()))
        items_ok = (set(self.pool.items) | set(self.pool.mega_stones.get(species, []))) - banned
        item = self._weighted(rng, info.get('item', []), items_ok, used_items)
        ability = self._weighted(rng, info.get('ability', []), set(self.pool.abilities.get(species, [])))
        known = MonState(species=species, moves=moves, item=item, ability=ability)
        s = self.prior._sample_set(species, known, rng, used_items, True)
        nature = self._weighted(rng, info.get('nature', []), set(self.pool.natures))
        if nature and rng.random() < 0.7:
            s['nature'] = nature
        s['level'] = 50
        if s.get('item'):
            used_items.add(s['item'])
        return s

    def team(self, rng: random.Random, size: int = 6):
        nums = self.pool.meta['num']
        cands, weights = list(self.species), list(self.weights)
        species, used = [], set()
        while cands and len(species) < size:
            i = rng.choices(range(len(cands)), weights)[0]
            sp = cands.pop(i)
            weights.pop(i)
            if nums.get(sp) in used:
                continue
            used.add(nums.get(sp))
            species.append(sp)
        if len(species) < size:
            return None
        used_items: set = set()
        return [self.make_set(sp, rng, used_items) for sp in species]


class TeamSampler:
    """Teams for training games: a weighted mixture of random legal teams (``random``), the evolved team pool
    (``pool``) and usage-based realistic teams (``usage``).  Unavailable sources are dropped; every pool or
    usage team is checked with :class:`TeamValidator` and replaced by a random team if it is not legal."""

    def __init__(self, formatid: str, seed: int, pool_path: str | None = None, pool_prob: float = 0.5,
                 mix: dict | None = None, usage_path: str | None = None):
        self.rng = random.Random(seed)
        self.gen = RandomTeamGenerator(self.rng.randrange(1 << 30), formatid=formatid)
        self.validator = TeamValidator(formatid)
        if mix is None:
            mix = {'pool': pool_prob, 'random': 1.0 - pool_prob}
        unknown = set(mix) - set(TEAM_SOURCES)
        if unknown:
            raise ValueError(f"unknown team source(s) {sorted(unknown)} (use {', '.join(TEAM_SOURCES)})")
        wanted = {k for k, w in mix.items() if float(w) > 0}
        self.pool = []
        if 'pool' in wanted and pool_path and os.path.exists(pool_path):
            with open(pool_path, encoding='utf-8-sig') as f:
                data = json.load(f)
            teams = data['teams'] if isinstance(data, dict) else data
            self.pool = [t for t in teams if self.legal(t)]
        self.usage = None
        if 'usage' in wanted and usage_path and os.path.exists(usage_path):
            try:
                self.usage = UsageTeamBuilder(formatid, usage_path, pool_path)
            except Exception as exc:  # optional source: train without it
                print(f'[teams] usage statistics not used ({type(exc).__name__}: {exc})', file=sys.stderr)
        available = {'random': True, 'pool': bool(self.pool), 'usage': self.usage is not None}
        self.mix = {k: float(w) for k, w in mix.items() if available.get(k) and float(w) > 0}
        if not self.mix:
            self.mix = {'random': 1.0}
        self.counts = Counter()

    def reseed(self, seed: int):
        """Same random stream as a freshly constructed sampler with this seed."""
        self.rng.seed(seed)
        self.gen.rng.seed(self.rng.randrange(1 << 30))
        self.counts = Counter()

    def legal(self, team) -> bool:
        try:
            return not self.validator.validate_team(team)
        except Exception:
            return False

    def team(self):
        names = list(self.mix)
        src = self.rng.choices(names, [self.mix[k] for k in names])[0] if len(names) > 1 else names[0]
        if src == 'pool':
            self.counts['pool'] += 1
            return plain_team(self.rng.choice(self.pool))
        if src == 'usage':
            try:
                team = self.usage.team(self.rng)
            except Exception:
                team = None
            if team is not None and self.legal(team):
                self.counts['usage'] += 1
                return team
            self.counts['usage_fallback'] += 1
        else:
            self.counts['random'] += 1
        return self.gen.random_team()


def team_settings(cfg: TrainConfig) -> tuple:
    """-> (pool path or None, usage path or None, source weights) for a config."""
    pool = _auto_path(cfg.team_pool, cfg.formatid, 'teams_singles.json')
    usage = _auto_path(cfg.usage_stats, cfg.formatid, 'usage_singles.json')
    mix = cfg.team_mix
    if mix is None:
        if cfg.team_pool not in (None, '', 'auto'):  # explicit pool, legacy weighting
            mix = {'pool': cfg.team_pool_prob, 'random': 1.0 - cfg.team_pool_prob}
        else:
            mix = dict(DEFAULT_TEAM_MIX)
    return pool, usage, dict(mix)


# ---------------------------------------------------------------------------
# Recording wrapper for rule-based teachers

class RecordingAgent:
    """Wraps any agent so its decisions carry observations (for behaviour cloning)."""

    def __init__(self, base):
        self.base = base

    def choose(self, session, sid, view, legal):
        option = self.base.choose(session, sid, view, legal)
        request = session.request(sid)
        obs = encode(view, request)
        info = {'idx': legal.index(option), 'kind': 1 if request.get('teamPreview') else 0,
                'n_lead': view.n_active, 'phi': potential(view)}
        return AgentChoice(option, obs=obs, info=info)


# ---------------------------------------------------------------------------
# Worker side (CPU only: one thread per process, weights always loaded to the CPU)

_WORKER = {}


def _worker_init():
    # Ctrl-C in a terminal reaches every process of the group: only the learner reacts (it finishes the
    # iteration and saves); the workers keep playing until then
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass
    torch.set_num_threads(1)


def _state_bytes(model) -> bytes:
    """The model for the workers: ``{'config', 'state_dict'}`` with CPU tensors, serialised."""
    buf = io.BytesIO()
    torch.save({'config': dict(model.config), 'state_dict': cpu_state_dict(model)}, buf)
    return buf.getvalue()


def _worker_model(cfg: dict, state: bytes) -> PolicyValueNet:
    torch.set_num_threads(1)
    data = torch.load(io.BytesIO(state), map_location='cpu', weights_only=False)
    if isinstance(data, dict) and 'state_dict' in data and 'config' in data:
        conf, sd = data['config'], data['state_dict']
    else:  # bare state dict (architecture from the training config)
        conf, sd = {'d': cfg['d'], 'layers': cfg['layers'], 'heads': cfg['heads']}, data
    key = (conf['d'], conf['layers'], conf['heads'], bool(conf.get('critic', False)))
    model = _WORKER.get('model')
    if model is None or _WORKER.get('model_key') != key:
        model = PolicyValueNet(d=conf['d'], layers=conf['layers'], heads=conf['heads'], sizes=conf.get('sizes'),
                               critic=bool(conf.get('critic', False)))
        _WORKER['model'], _WORKER['model_key'] = model, key
    model.load_state_dict(sd)
    model.eval()
    return model


def _snapshot_model(path: str) -> PolicyValueNet:
    cache = _WORKER.setdefault('snapshots', {})
    if path not in cache:
        if len(cache) >= 32:
            cache.clear()
        cache[path] = load_model(path, map_location='cpu')
    return cache[path]


def _team_sampler(cfg: dict, seed: int) -> TeamSampler:
    key = (cfg['formatid'], cfg.get('team_pool'), cfg.get('usage_stats'),
           json.dumps(cfg.get('team_mix'), sort_keys=True), cfg.get('team_pool_prob', 0.5))
    cache = _WORKER.setdefault('teams', {})
    sampler = cache.get(key)
    if sampler is None:
        sampler = TeamSampler(cfg['formatid'], seed, cfg.get('team_pool'), cfg.get('team_pool_prob', 0.5),
                              mix=cfg.get('team_mix'), usage_path=cfg.get('usage_stats'))
        cache[key] = sampler
    else:
        sampler.reseed(seed)
    return sampler


def _trajectories(result, learner_sides):
    """Split recorded decisions into per-side trajectories with terminal rewards."""
    out = []
    for sid in learner_sides:
        steps = [d for d in result.decisions if d.side == sid and d.obs is not None]
        if not steps:
            continue
        if result.winner is None:
            z = 0.0
        else:
            z = 1.0 if result.winner == sid else -1.0
        out.append({'steps': [{'obs': d.obs, 'legal': d.legal, **d.info} for d in steps], 'z': z})
    return out


def obs_spec() -> dict:
    """Shape and dtype of every observation array (``features.encode``), as the learner expects them."""
    from ..env.actions import N_MOVES
    from .features import GLOB_F, MON_F, MOVE_F, N_MONS
    return {'ids': ((N_MONS, 3), np.int64), 'mon': ((N_MONS, MON_F), np.float32),
            'move_ids': ((N_MONS, N_MOVES), np.int64), 'move': ((N_MONS, N_MOVES, MOVE_F), np.float32),
            'glob': ((GLOB_F,), np.float32), 'active_idx': ((2,), np.int64)}


def check_trajectory(traj: dict, need_logp: bool = True):
    """Raise ValueError when ``traj`` does not follow the trajectory record format of the module docstring
    (the contract between actors and the learner; the learner checks a sample of every iteration)."""
    if not isinstance(traj, dict) or 'steps' not in traj or 'z' not in traj:
        raise ValueError("a trajectory is a dict {'z': float, 'steps': [...]}")
    if float(traj['z']) not in (-1.0, 0.0, 1.0):
        raise ValueError(f"trajectory z must be +1, -1 or 0, not {traj['z']!r}")
    spec = obs_spec()
    for t, s in enumerate(traj['steps']):
        where = f'step {t}'
        for key in ('obs', 'legal', 'idx', 'kind', 'phi') + (('logp',) if need_logp else ()):
            if key not in s:
                raise ValueError(f'{where}: missing {key!r}')
        for key, (shape, dtype) in spec.items():
            a = s['obs'].get(key)
            if not isinstance(a, np.ndarray) or a.shape != shape or a.dtype != dtype:
                got = (a.shape, a.dtype) if isinstance(a, np.ndarray) else type(a).__name__
                raise ValueError(f"{where}: obs[{key!r}] must be a numpy {np.dtype(dtype).name} array of shape "
                                 f"{shape}, got {got}")
        if s['kind'] not in (0, 1):
            raise ValueError(f"{where}: kind must be 0 (move) or 1 (team preview), not {s['kind']!r}")
        if not s['legal'] or not 0 <= int(s['idx']) < len(s['legal']):
            raise ValueError(f"{where}: idx {s['idx']!r} is not an index into its {len(s['legal'])} legal options")
        width = 2 if s['kind'] == 0 else 4
        if s['kind'] == 1 and int(s.get('n_lead', 0)) < 1:
            raise ValueError(f'{where}: team-preview steps need n_lead (number of active slots, >= 1)')
        if any(len(o) > width for o in s['legal']):
            raise ValueError(f'{where}: options must be tuples of at most {width} ints')
        if need_logp and not float(s['logp']) <= 1e-4:  # NaN fails too
            raise ValueError(f"{where}: logp must be a log-probability (<= 0), not {s['logp']!r}")


_SUMMED_STATS = ('games', 'turns', 'errors', 'decisions', 'nn_decisions', 'forwards', 'nn_seconds', 'job_seconds',
                 'jobs')


def rollout_throughput(stats: dict, seconds: float, workers: int) -> dict:
    """Actor throughput of one rollout round from the merged job stats: games/s, decisions/s (both players),
    mean rows per batched forward, share of worker time spent in the network and worker utilisation."""
    seconds = max(1e-6, seconds)
    out = {'rollout_games_per_s': stats.get('games', 0) / seconds}
    if stats.get('decisions'):
        out['decisions'] = stats['decisions']
        out['decisions_per_s'] = stats['decisions'] / seconds
    if stats.get('forwards'):
        out['forwards'] = stats['forwards']
        out['mean_batch'] = stats.get('nn_decisions', 0) / stats['forwards']
    if stats.get('job_seconds'):
        out['nn_share'] = stats.get('nn_seconds', 0.0) / stats['job_seconds']
        out['worker_util'] = stats['job_seconds'] / (seconds * max(1, workers))
    if stats.get('jobs'):
        out['jobs'] = stats['jobs']
    return out


def _merge_stats(total: dict, s: dict):
    for k in _SUMMED_STATS:
        if k in s or k in ('games', 'turns', 'errors'):
            total[k] = total.get(k, 0) + s.get(k, 0)
    for group in ('by_opp', 'by_snap'):
        dst = total.setdefault(group, {})
        for k, (w, n) in s.get(group, {}).items():
            cur = dst.setdefault(k, [0.0, 0])
            cur[0] += w
            cur[1] += n
    teams = total.setdefault('teams', {})
    for k, n in s.get('teams', {}).items():
        teams[k] = teams.get(k, 0) + n
    return total


def _rollout_job(args):
    """Play ``n_games`` and return ``(trajectories, stats)`` (format in the module docstring).

    ``snapshots``: list of ``(path, weight)`` (or plain paths) the snapshot opponents are drawn from.
    """
    cfg, state, seed, n_games, snapshots, mode = args
    rng = random.Random(seed)
    torch.manual_seed(seed)
    teams = _team_sampler(cfg, rng.randrange(1 << 30))
    max_turns = cfg.get('max_turns', 150)
    trajs = []
    stats = {'games': 0, 'turns': 0, 'errors': 0, 'by_opp': {}, 'by_snap': {}, 'teams': {}}
    if mode == 'bc':
        teacher = HeuristicAgent(rng.randrange(1 << 30), randomness=0.05)
        for _ in range(n_games):
            a1, a2 = RecordingAgent(teacher), RecordingAgent(HeuristicAgent(rng.randrange(1 << 30), 0.05))
            r = play_battle(teams.team(), teams.team(), a1, a2, cfg['formatid'], seed=gen5_seed(rng),
                            record=True, max_turns=max_turns)
            trajs.extend(_trajectories(r, ('p1', 'p2')))
            stats['games'] += 1
            stats['turns'] += r.turns
            stats['errors'] += int(bool(r.error))
        stats['teams'] = dict(teams.counts)
        return trajs, stats
    model = _worker_model(cfg, state)
    learner = NNAgent(model, sample=True, record=True)
    learner.seed(rng.randrange(1 << 30))
    snaps = [(s, 1.0) if isinstance(s, str) else (s[0], float(s[1])) for s in snapshots or []]
    opp_names = list(cfg['opponents'])
    opp_w = [cfg['opponents'][k] for k in opp_names]
    for _ in range(n_games):
        kind = rng.choices(opp_names, opp_w)[0]
        if kind == 'snapshot' and not snaps:
            kind = 'self'
        snap = None
        if kind == 'self':
            opp = NNAgent(model, sample=True, record=True)
            opp.seed(rng.randrange(1 << 30))
            sides = ('p1', 'p2')
        elif kind == 'snapshot':
            snap = rng.choices([p for p, _ in snaps], [max(w, 1e-6) for _, w in snaps])[0]
            opp = NNAgent(_snapshot_model(snap), sample=True)
            opp.seed(rng.randrange(1 << 30))
            sides = ('p1',)
        else:
            opp = HeuristicAgent(rng.randrange(1 << 30), randomness=cfg.get('heuristic_randomness', 0.05))
            sides = ('p1',)
        swap = rng.random() < 0.5
        t1, t2 = teams.team(), teams.team()
        if swap and kind != 'self':
            r = play_battle(t1, t2, opp, learner, cfg['formatid'], seed=gen5_seed(rng), record=True,
                            max_turns=max_turns)
            sides = ('p2',)
        else:
            r = play_battle(t1, t2, learner, opp, cfg['formatid'], seed=gen5_seed(rng), record=True,
                            max_turns=max_turns)
        trajs.extend(_trajectories(r, sides))
        stats['games'] += 1
        stats['turns'] += r.turns
        stats['errors'] += int(bool(r.error))
        stats['decisions'] = stats.get('decisions', 0) + len(r.decisions)
        if kind != 'self':
            me = sides[0]
            score = 1.0 if r.winner == me else (0.5 if r.winner is None else 0.0)
            for group, key in (('by_opp', kind), ('by_snap', snap)):
                if key is not None:
                    cur = stats[group].setdefault(key, [0.0, 0])
                    cur[0] += score
                    cur[1] += 1
    stats['teams'] = dict(teams.counts)
    return trajs, stats


def _eval_job(args):
    """Greedy policy (p1) vs ``opponent`` ('heuristic' = HeuristicAgent without randomness, or a checkpoint
    path) on fixed pairs -> (opponent, MatchResult)."""
    cfg, state, opponent, pairs, seed = args
    agent = NNAgent(_worker_model(cfg, state), sample=False)
    if opponent == 'heuristic':
        opp = HeuristicAgent(seed, randomness=0.0)
    else:
        opp = NNAgent(_snapshot_model(opponent), sample=False)
    return opponent, play_pairs(agent, opp, pairs, cfg['formatid'], max_turns=cfg.get('eval_max_turns', 200))


# ---------------------------------------------------------------------------
# Learner side

def compute_gae(phis, values, z: float, gamma: float, lam: float, shaping: float):
    """GAE(lambda) for one trajectory with shaped rewards -> (advantages, returns) as float64 arrays.

    Reward after step t: ``shaping * (gamma * phi[t+1] - phi[t])``; after the last step ``z - shaping *
    phi[-1]`` (the potential of the terminal state is 0).  ``values`` are the critic's estimates of the
    shaped return.  With ``values = 0`` and ``lam = 1`` the returns are the discounted shaped returns.
    """
    phis = np.asarray(phis, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n = len(phis)
    rewards = np.empty(n)
    rewards[:-1] = shaping * (gamma * phis[1:] - phis[:-1])
    rewards[-1] = z - shaping * phis[-1]
    next_v = np.append(values[1:], 0.0)
    deltas = rewards + gamma * next_v - values
    adv = np.empty(n)
    last = 0.0
    k = gamma * lam
    for t in range(n - 1, -1, -1):
        last = deltas[t] + k * last
        adv[t] = last
    return adv, adv + values


def _gae(traj, gamma, lam, shaping):
    """Legacy helper: GAE from the actor-recorded ``value`` of each step (stored as 'adv' / 'ret')."""
    steps = traj['steps']
    adv, ret = compute_gae([s.get('phi', 0.0) for s in steps], [s.get('value', 0.0) for s in steps],
                           traj['z'], gamma, lam, shaping)
    for s, a, r in zip(steps, adv, ret):
        s['adv'], s['ret'] = float(a), float(r)


def _fill_options(arr, mask, row, options):
    try:
        a = np.asarray(options, dtype=np.int64).reshape(len(options), -1)
        arr[row, :len(options), :a.shape[1]] = a
    except ValueError:  # ragged option tuples
        for i, o in enumerate(options):
            arr[row, i, :len(o)] = o
    mask[row, :len(options)] = True


class StepBatch:
    """All decisions of one iteration, collated once into tensors on the learner's device.

    Minibatches index these tensors (no Python-side re-collation).  Move decisions keep their joint actions
    in ``combos`` [N,L,2]; team-preview decisions keep their options in ``options`` [P,O,4], referenced by
    ``prow`` (row in ``options`` or -1).  ``adv`` / ``ret`` (critic target) are filled by
    :func:`compute_targets`; ``z`` is the value-head target.
    """

    def __init__(self, trajs: list, device='cpu'):
        steps = [s for tr in trajs for s in tr['steps']]
        n = len(steps)
        if n == 0:
            raise ValueError('no decisions to train on')
        self.n = n
        self.device = torch.device(device)
        self.bounds = []
        z = np.empty(n, dtype=np.float32)
        pos = 0
        for tr in trajs:
            k = len(tr['steps'])
            if k:
                self.bounds.append((pos, pos + k, float(tr['z'])))
                z[pos:pos + k] = tr['z']
                pos += k
        kind = np.fromiter((s['kind'] for s in steps), dtype=np.int64, count=n)
        idx = np.fromiter((s['idx'] for s in steps), dtype=np.int64, count=n)
        n_lead = np.fromiter((s.get('n_lead', 1) for s in steps), dtype=np.int64, count=n)
        self.phi_np = np.fromiter((s.get('phi', 0.0) for s in steps), dtype=np.float32, count=n)
        logp = np.fromiter((s.get('logp', 0.0) for s in steps), dtype=np.float32, count=n)
        move_rows = np.flatnonzero(kind == 0)
        prev_rows = np.flatnonzero(kind == 1)
        L = max((len(steps[i]['legal']) for i in move_rows), default=1)
        O = max((len(steps[i]['legal']) for i in prev_rows), default=1)
        combos = np.full((n, L, 2), -1, dtype=np.int64)
        cmask = np.zeros((n, L), dtype=bool)
        for i in move_rows:
            _fill_options(combos, cmask, i, steps[i]['legal'])
        P = max(1, len(prev_rows))
        options = np.full((P, O, 4), -1, dtype=np.int64)
        omask = np.zeros((P, O), dtype=bool)
        prow = np.full(n, -1, dtype=np.int64)
        for j, i in enumerate(prev_rows):
            _fill_options(options, omask, j, steps[i]['legal'])
            prow[i] = j
        dev = self.device

        def t(a):
            return torch.from_numpy(np.ascontiguousarray(a)).to(dev)
        self.obs = collate_obs([s['obs'] for s in steps], dev)
        self.kind, self.idx, self.n_lead = t(kind), t(idx), t(n_lead)
        self.phi, self.logp_old, self.z = t(self.phi_np), t(logp), t(z)
        self.combos, self.cmask = t(combos), t(cmask)
        self.options, self.omask, self.prow = t(options), t(omask), t(prow)
        self.z_np = z
        self.adv = self.ret = None

    def rows(self, start: int, end: int) -> torch.Tensor:
        return torch.arange(start, end, device=self.device)


def _group_stats(logits, idx):
    """Vectorised log-prob of the chosen option, entropy and greedy agreement for padded logits."""
    lp = F.log_softmax(logits, dim=-1)
    p = lp.exp()
    chosen = lp.gather(1, idx.view(-1, 1)).squeeze(1)
    ent = -(p * torch.where(p > 0, lp, torch.zeros_like(lp))).sum(-1)
    agree = (logits.argmax(-1) == idx).float()
    return chosen, ent, agree


def _forward_stats(model, b: StepBatch, rows: torch.Tensor, shaping: float):
    """-> (logp of the chosen options, entropy, value, critic estimate, greedy agreement), each [len(rows)].

    Move and team-preview rows are evaluated with the same tensor code (each row's other branch is fully
    masked and discarded), so there are no data-dependent Python branches or host syncs.
    """
    obs = {k: v[rows] for k, v in b.obs.items()}
    slot_logits, value, x = model(obs)
    is_prev = b.kind[rows] == 1
    idx = b.idx[rows]
    zero = torch.zeros_like(idx)
    lg = joint_logits(slot_logits, b.combos[rows], b.cmask[rows])
    lp_m, ent_m, ag_m = _group_stats(lg, torch.where(is_prev, zero, idx))
    pr = b.prow[rows]
    prc = pr.clamp(min=0)
    pick, lead = model.preview_scores(x)
    lgp = preview_logits(pick, lead, b.options[prc], b.n_lead[rows], b.omask[prc] & is_prev.unsqueeze(1))
    lp_p, ent_p, ag_p = _group_stats(lgp, torch.where(is_prev, idx, zero))
    logp = torch.where(is_prev, lp_p, lp_m)
    ent = torch.where(is_prev, ent_p, ent_m)
    agree = torch.where(is_prev, ag_p, ag_m)
    critic = model.critic(x) if model.has_critic else value - shaping * b.phi[rows]
    return logp, ent, value, critic, agree


def _policy_eval(model, batch_steps, device=None):
    """-> (logp of the chosen options [B], entropy [B], value [B], greedy agreement [B]) for a list of steps.

    The steps are collated on ``device`` (default: the model's device), so this works for a CUDA model too."""
    b = StepBatch([{'steps': batch_steps, 'z': 0.0}], device if device is not None else _model_device(model))
    logp, ent, value, _critic, agree = _forward_stats(model, b, b.rows(0, b.n), 0.0)
    return logp, ent, value, agree


@torch.no_grad()
def predict_values(model, b: StepBatch, shaping: float, chunk: int = 2048):
    """Critic estimates and value-head outputs for every row (no grad, batched) -> two numpy arrays."""
    was_training = model.training
    model.eval()
    crit = torch.empty(b.n, device=b.device)
    val = torch.empty(b.n, device=b.device)
    for start in range(0, b.n, chunk):
        end = min(b.n, start + chunk)
        obs = {k: v[start:end] for k, v in b.obs.items()}
        _logits, value, x = model(obs)
        crit[start:end] = model.critic(x) if model.has_critic else value - shaping * b.phi[start:end]
        val[start:end] = value
    model.train(was_training)
    return crit.cpu().numpy(), val.cpu().numpy()


def compute_targets(b: StepBatch, critic_values, gamma: float, lam: float, shaping: float):
    """Fill ``b.adv`` and ``b.ret`` (critic target) from GAE over each trajectory."""
    adv = np.zeros(b.n, dtype=np.float32)
    ret = np.zeros(b.n, dtype=np.float32)
    for s, e, z in b.bounds:
        a, r = compute_gae(b.phi_np[s:e], critic_values[s:e], z, gamma, lam, shaping)
        adv[s:e] = a
        ret[s:e] = r
    b.adv = torch.from_numpy(adv).to(b.device)
    b.ret = torch.from_numpy(ret).to(b.device)
    b.adv_np, b.ret_np = adv, ret
    return adv, ret


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def bc_train(model, trajs, epochs, lr, minibatch, log=print, device=None, gamma: float = 0.99,
             shaping: float = 1.0, value_coef: float = 0.5, vf_coef: float = 0.5, generator=None):
    """Behaviour cloning: cross-entropy on the teacher's choices, value head on the outcome z, critic (if
    any) on the discounted shaped return."""
    device = torch.device(device) if device is not None else _model_device(model)
    b = trajs if isinstance(trajs, StepBatch) else StepBatch(trajs, device)
    compute_targets(b, np.zeros(b.n, dtype=np.float32), gamma, 1.0, shaping)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(b.n, generator=generator).to(b.device)
        tot, acc, n = 0.0, 0.0, 0
        for start in range(0, b.n, minibatch):
            rows = perm[start:start + minibatch]
            logp, _, value, critic, agree = _forward_stats(model, b, rows, shaping)
            loss = -logp.mean() + value_coef * F.mse_loss(value, b.z[rows])
            if model.has_critic:
                loss = loss + vf_coef * F.mse_loss(critic, b.ret[rows])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            k = len(rows)
            tot += float(loss.detach()) * k
            acc += float(agree.sum())
            n += k
        log(f"[bc] epoch {ep + 1}/{epochs} loss {tot / n:.4f} teacher-agreement {acc / n * 100:.1f}% "
            f"({n} samples)")
    model.eval()


def ppo_update(model, opt, b: StepBatch, cfg: TrainConfig, log=print, shaping: float | None = None,
               generator=None, minibatch: int | None = None) -> dict:
    """PPO epochs over a collated batch with ``adv`` / ``ret`` filled.  Stops early (before the step that
    would exceed it) once the approximate KL to the acting policy passes ``1.5 * cfg.target_kl``."""
    shaping = cfg.shaping if shaping is None else shaping
    mb = minibatch or cfg.minibatch or (2048 if b.device.type == 'cuda' else 512)
    model.train()
    adv_n = (b.adv - b.adv.mean()) / (b.adv.std() + 1e-8) if b.n > 1 else b.adv * 0
    sums = {'pl': 0.0, 'cl': 0.0, 'vl': 0.0, 'ent': 0.0, 'kl': 0.0, 'clipfrac': 0.0}
    n_seen, updates, early, epochs_done = 0, 0, False, 0
    last_kl = 0.0
    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(b.n, generator=generator).to(b.device)
        for start in range(0, b.n, mb):
            rows = perm[start:start + mb]
            logp, ent, value, critic, _ = _forward_stats(model, b, rows, shaping)
            log_ratio = logp - b.logp_old[rows]
            ratio = log_ratio.exp()
            approx_kl = ((ratio - 1) - log_ratio).mean()
            last_kl = float(approx_kl.detach())
            if cfg.target_kl is not None and last_kl > 1.5 * cfg.target_kl:
                early = True
                break
            a = adv_n[rows]
            pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
            vl = F.mse_loss(value, b.z[rows])
            cl = F.mse_loss(critic, b.ret[rows]) if model.has_critic else torch.zeros((), device=b.device)
            ent_m = ent.mean()
            loss = pl + cfg.vf_coef * cl + cfg.value_coef * vl - cfg.ent_coef * ent_m
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            k = len(rows)
            clipfrac = ((ratio - 1).abs() > cfg.clip).float().mean()
            for key, v in (('pl', pl), ('cl', cl), ('vl', vl), ('ent', ent_m), ('kl', approx_kl),
                           ('clipfrac', clipfrac)):
                sums[key] += float(v.detach()) * k
            n_seen += k
            updates += 1
        if early:
            break
        epochs_done += 1
    model.eval()
    out = {k: v / max(1, n_seen) for k, v in sums.items()}
    out.update({'updates': updates, 'epochs': epochs_done, 'early_stop': early, 'last_kl': last_kl})
    return out


# ---------------------------------------------------------------------------
# Small helpers

def pfsp_weight(score: float, games: float, floor: float = 0.05) -> float:
    """Prioritised fictitious self-play: opponents we score badly against are picked more often."""
    p = (score + 1.0) / (games + 2.0)
    return max(floor, (1.0 - p) ** 2)


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().to('cpu')
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def copy_atomic(src: str, dst: str, only_if_changed: bool = True) -> bool:
    """Copy ``src`` to ``dst`` through a temporary file + rename; skipped when ``dst`` already matches
    (same size and modification time, which the copy preserves)."""
    if only_if_changed and os.path.exists(dst):
        a, b = os.stat(src), os.stat(dst)
        if a.st_size == b.st_size and a.st_mtime_ns == b.st_mtime_ns:
            return False
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = f'{dst}.tmp{os.getpid()}'
    try:
        shutil.copyfile(src, tmp)
        try:  # keep the modification time (network drives may refuse; the copy is still good)
            shutil.copystat(src, tmp)
        except OSError:
            pass
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return True


def write_text_atomic(path: str, text: str):
    """Write a text file through a temporary file + rename (never half-written)."""
    tmp = f'{path}.tmp{os.getpid()}'
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _fmt_s(seconds) -> str:
    if seconds is None:
        return '-'
    seconds = max(0, int(seconds))
    return f'{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}'


_SNAPSHOT_RE = re.compile(r'^iter\d+\.pt$')


# ---------------------------------------------------------------------------
# The training run

class Trainer:
    """One training run: setup / resume, behaviour cloning, PPO iterations, evaluations, checkpoints."""

    def __init__(self, cfg: TrainConfig, log=print):
        validate_config(cfg)
        self.cfg = cfg
        self._print = log
        self.out = cfg.out
        os.makedirs(self.out, exist_ok=True)
        self.log_path = os.path.join(self.out, 'log.jsonl')
        self.text_log = os.path.join(self.out, cfg.log_file) if cfg.log_file else None
        self.session_start = time.time()
        self.deadline = self.session_start + cfg.time_budget_h * 3600 if cfg.time_budget_h else None
        self.workers = resolve_workers(cfg.workers)
        self.device = resolve_device(cfg.device)
        self.minibatch = cfg.minibatch or (2048 if self.device.type == 'cuda' else 512)
        self.stop_flag = False
        self.stop_signal = None
        self.ex = None
        self.model = self.opt = None
        self.it = 0
        self.counters = {'games': 0, 'samples': 0, 'wall_seconds': 0.0, 'eval_games': 0, 'bc_games': 0,
                         'sessions': 0}
        self.wall_base = 0.0
        self.best = {'score': -1.0, 'iter': None}
        self.snapshots = []           # {'path', 'iter', 'anchor', 'score', 'games'} (paths relative to out)
        self.eval_set = None
        self.eval_history = []
        self.last_eval_it = None
        self.incumbent = None         # file name in out
        self.incumbent_source = None
        self._synced = {}             # name -> (size, mtime_ns) of the copy last mirrored to sync_dir
        self.iter_seconds = None      # moving average of the iteration duration (rollout + update)
        self.last_iter_seconds = None
        self.eval_seconds = None
        self.games_per_s = None
        self.last_ckpt = time.time()
        self.need_bc = False
        self._prev_handlers = {}
        self._weights_n = 0

    # -- logging ----------------------------------------------------------------------------------------------
    def log(self, msg: str):
        self._print(msg)
        if self.text_log:
            try:
                with open(self.text_log, 'a', encoding='utf-8') as f:
                    f.write(msg + '\n')
            except OSError:
                pass

    def _jsonl(self, entry: dict):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')

    def _trim_jsonl(self, it: int) -> int:
        """Drop log.jsonl entries of iterations after ``it`` (written after the checkpoint a resumed run starts
        from; they are played again), so every iteration appears once.  Returns the number dropped."""
        if not os.path.exists(self.log_path):
            return 0
        keep, dropped = [], 0
        with open(self.log_path, encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    n = entry['eval']['iter'] if 'eval' in entry else entry.get('iter')
                except (ValueError, KeyError, TypeError):
                    n = None
                if n is not None and n > it:
                    dropped += 1
                else:
                    keep.append(line if line.endswith('\n') else line + '\n')
        if dropped:
            write_text_atomic(self.log_path, ''.join(keep))
        return dropped

    def _path(self, name: str) -> str:
        return os.path.join(self.out, name)

    # -- signals ----------------------------------------------------------------------------------------------
    def _on_signal(self, signum, frame):
        if self.stop_flag:
            sys.stderr.write('\n[train] second signal: exiting immediately (the last checkpoint is kept)\n')
            sys.stderr.flush()
            self._kill_workers()
            os._exit(128 + signum)
        self.stop_flag = True
        self.stop_signal = signal.Signals(signum).name
        sys.stderr.write(f'\n[train] {self.stop_signal}: finishing the current iteration, then saving and '
                         'exiting (send it again to exit immediately)\n')
        sys.stderr.flush()

    def _install_signals(self):
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                self._prev_handlers[s] = signal.signal(s, self._on_signal)
            except (ValueError, OSError):  # not the main thread: no graceful stop
                pass

    def _restore_signals(self):
        for s, h in self._prev_handlers.items():
            try:
                signal.signal(s, h)
            except (ValueError, OSError, TypeError):
                pass
        self._prev_handlers = {}

    # -- worker pool ------------------------------------------------------------------------------------------
    def _start_pool(self):
        self.ex = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context('spawn'),
                                      initializer=_worker_init)

    def _kill_workers(self):
        procs = getattr(self.ex, '_processes', None) or {}
        for p in list(procs.values()):
            try:
                p.kill()
            except Exception:
                pass

    def _shutdown_pool(self):
        if self.ex is not None:
            self.ex.shutdown(wait=True, cancel_futures=True)
            self.ex = None

    def _map(self, fn, jobs):
        for attempt in range(2):
            try:
                return list(self.ex.map(fn, jobs))
            except BrokenProcessPool as exc:
                if attempt:
                    raise
                self.log(f'[warn] a worker process died ({exc}); restarting the pool and retrying')
                self.ex.shutdown(wait=False, cancel_futures=True)
                self._start_pool()

    def _gather(self, fn, jobs: list, transform=None) -> list:
        """Run ``jobs`` on the pool, each worker taking the next job as soon as it is free; returns the results
        in job order (so the learner's data never depend on which worker finished first).  ``transform(result)``
        is applied as each result arrives.  A dead worker restarts the pool and the round is retried once."""
        for attempt in range(2):
            try:
                futures = {self.ex.submit(fn, job): i for i, job in enumerate(jobs)}
                out = [None] * len(jobs)
                for f in as_completed(futures):
                    r = f.result()
                    out[futures[f]] = transform(r) if transform else r
                return out
            except BrokenProcessPool as exc:
                if attempt:
                    raise
                self.log(f'[warn] a worker process died ({exc}); restarting the pool and retrying')
                self.ex.shutdown(wait=False, cancel_futures=True)
                self._start_pool()

    def _run_jobs(self, state: bytes, n_games: int, seed: int, snapshots: list, mode: str):
        """The one-game-at-a-time path (``vectorized=False``): one job per worker, weights in every job."""
        W = self.workers
        per = [n_games // W + (1 if w < n_games % W else 0) for w in range(W)]
        jobs = [(self.wcfg, state, seed * 1000 + w, k, snapshots, mode) for w, k in enumerate(per) if k > 0]
        trajs, stats = [], {}
        for t, s in self._map(_rollout_job, jobs):
            trajs.extend(t)
            _merge_stats(stats, s)
        stats['jobs'] = len(jobs)
        return trajs, stats

    # -- vectorised actors --------------------------------------------------------------------------------------
    def _vec_opts(self) -> dict:
        cfg = self.cfg
        return {'envs': cfg.envs_per_worker, 'compress': cfg.compress_obs, 'snapshots_per_job': cfg.snapshots_per_job}

    def job_sizes(self, n_games: int) -> list:
        """Split ``n_games`` into ``workers * j`` jobs with ``j`` <= jobs_per_worker chosen so a job has at least
        envs_per_worker games when possible (full batches) and every worker gets the same number of jobs."""
        W, K = self.workers, self.cfg.envs_per_worker
        j = max(1, min(self.cfg.jobs_per_worker, int(n_games // max(1, W * K))))
        n_jobs = max(1, min(n_games, W * j))
        return [n_games // n_jobs + (1 if i < n_games % n_jobs else 0) for i in range(n_jobs)]

    def _publish_weights(self) -> 'vec_rollout.WeightsRef':
        self._weights_n += 1
        return vec_rollout.publish_weights(self.model, self._path(f'_weights_{self._weights_n}.pt'))

    def _remove_weights(self, ref=None):
        """Delete the published weights file ``ref`` (default: every ``_weights_*.pt`` in out)."""
        paths = [ref.path] if ref is not None else [self._path(n) for n in os.listdir(self.out)
                                                   if n.startswith('_weights_') and n.endswith('.pt')]
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass

    def _run_vec_jobs(self, weights, n_games: int, seed: int, snapshots: list, mode: str):
        jobs = [(self.wcfg, weights, seed * 1000 + i, k, snapshots, mode, self._vec_opts())
                for i, k in enumerate(self.job_sizes(n_games))]

        def unpack(res):
            t, s = res
            for tr in t:
                vec_rollout.decompress_trajectory(tr)
            return t, s
        trajs, stats = [], {}
        for t, s in self._gather(vec_rollout.rollout_job, jobs, unpack):
            trajs.extend(t)
            _merge_stats(stats, s)
        return trajs, stats

    def rollout(self, n_games: int, seed: int, snapshots: list, mode: str = 'ppo'):
        """Play ``n_games`` on the worker pool with the current weights -> ``(trajectories, stats)``."""
        if not self.cfg.vectorized:
            return self._run_jobs(_state_bytes(self.model) if mode == 'ppo' else b'', n_games, seed, snapshots, mode)
        ref = self._publish_weights() if mode == 'ppo' else None
        try:
            return self._run_vec_jobs(ref, n_games, seed, snapshots, mode)
        finally:
            if ref is not None:
                self._remove_weights(ref)

    # -- setup / resume ---------------------------------------------------------------------------------------
    def setup(self):
        cfg = self.cfg
        if self.device.type == 'cuda':
            torch.set_num_threads(2)
        else:
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        pool, usage, mix = team_settings(cfg)
        self.wcfg = {'formatid': cfg.formatid, 'max_turns': cfg.max_turns, 'opponents': dict(cfg.opponents),
                     'heuristic_randomness': cfg.heuristic_randomness, 'team_pool': pool,
                     'team_pool_prob': cfg.team_pool_prob, 'usage_stats': usage, 'team_mix': mix,
                     'd': cfg.d, 'layers': cfg.layers, 'heads': cfg.heads}
        self._remove_weights()  # left over by an interrupted session
        resumed = self._resume()
        if not resumed:
            self._fresh_start()
        self.counters['sessions'] += 1
        self.wall_base = self.counters['wall_seconds']
        self._setup_incumbent()
        self._setup_eval_set()
        write_text_atomic(self._path('config.json'), json.dumps(asdict(cfg), indent=2) + '\n')
        m = self.model.config
        dev = self.device.type
        if dev == 'cuda':
            dev += f' ({torch.cuda.get_device_name(self.device)})'
        self.log(f"[setup] device {dev}, {self.workers} rollout workers, minibatch {self.minibatch}, model "
                 f"d={m['d']} layers={m['layers']} critic={'on' if self.model.has_critic else 'off'}, "
                 f"{sum(p.numel() for p in self.model.parameters())} parameters")
        teams = ', '.join(f'{k} {w:g}' for k, w in mix.items())
        self.log(f"[setup] teams: {teams} (pool: {pool or '-'}, usage: {usage or '-'}); opponents: "
                 f"{', '.join(f'{k} {w:g}' for k, w in cfg.opponents.items())}; incumbent: "
                 f"{self.incumbent_source or '-'}")
        if self.deadline:
            self.log(f'[setup] time budget {cfg.time_budget_h:g} h: stopping by '
                     f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.deadline))}')
        return resumed

    def _restore_from_sync(self):
        sync = self.cfg.sync_dir
        n = 0
        for name in sorted(os.listdir(sync)):
            src = os.path.join(sync, name)
            if '.tmp' in name or not os.path.isfile(src):
                continue
            if name.endswith(('.pt', '.jsonl', '.json', '.log')) or name == self.cfg.log_file:
                copy_atomic(src, self._path(name))
                n += 1
        self.log(f'[resume] {self.out} had no state.pt: restored {n} files from {sync}')

    def _resume(self) -> bool:
        cfg = self.cfg
        state_path = self._path('state.pt')
        if cfg.resume == 'never':
            if os.path.exists(state_path):
                self.log(f'[setup] resume=never: starting over (the old {state_path} will be overwritten)')
            return False
        if cfg.resume != 'auto':
            raise ValueError(f"resume must be 'auto' or 'never', not {cfg.resume!r}")
        if not os.path.exists(state_path) and cfg.sync_dir and os.path.exists(os.path.join(cfg.sync_dir, 'state.pt')):
            self._restore_from_sync()
        if not os.path.exists(state_path):
            return False
        st = torch.load(state_path, map_location='cpu', weights_only=False)
        model = model_from_checkpoint(st['model'])
        restore_opt = True
        if cfg.critic and model.enable_critic():
            self.log('[resume] critic head added to the resumed model (optimizer state starts fresh)')
            restore_opt = False
        self.model = model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        if restore_opt and st.get('optimizer'):
            try:
                self.opt.load_state_dict(st['optimizer'])
            except (ValueError, KeyError) as exc:
                self.log(f'[resume] optimizer state not restored ({exc})')
        self.it = int(st['it'])
        dropped = self._trim_jsonl(self.it)
        if dropped:
            self.log(f'[resume] removed {dropped} log.jsonl entries written after the checkpoint (they are redone)')
        self.counters.update(st.get('counters', {}))
        self.best = st.get('best', self.best)
        self.snapshots = []
        for rec in st.get('snapshots', []):
            if os.path.exists(self._path(rec['path'])):
                self.snapshots.append(rec)
            else:
                self.log(f"[resume] snapshot {rec['path']} is missing; dropped from the opponent pool")
        self.eval_set = st.get('eval_set')
        self.eval_history = st.get('eval_history', [])
        self.last_eval_it = st.get('last_eval_it')
        self.incumbent = st.get('incumbent')
        self.iter_seconds = st.get('timing', {}).get('iter_seconds')
        self.eval_seconds = st.get('timing', {}).get('eval_seconds')
        self.games_per_s = st.get('timing', {}).get('games_per_s')
        rng = st.get('rng') or {}
        try:
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'])
            if self.device.type == 'cuda' and rng.get('cuda') and len(rng['cuda']) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(rng['cuda'])
        except (KeyError, TypeError, ValueError, RuntimeError):
            pass
        self.log(f"[resume] continuing from {state_path}: iteration {self.it} done, {self.counters['games']} games, "
                 f"{self.counters['samples']} samples, {_fmt_s(self.counters['wall_seconds'])} trained, "
                 f"best score {self.best.get('score', -1) * 100:.1f}% (iter {self.best.get('iter')})")
        return True

    def _fresh_start(self):
        cfg = self.cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed % (1 << 32))
        torch.manual_seed(cfg.seed)
        init = repo_path(cfg.init) if cfg.init else None
        if init:
            if not os.path.exists(init):
                raise FileNotFoundError(f'warm-start checkpoint not found: {cfg.init}')
            model = load_model(init)
            added = cfg.critic and model.enable_critic()
            self.log(f"[setup] warm start from {init}{' (new critic head = copy of the value head)' if added else ''}")
            save_model(model, self._path('init.pt'), {'source': os.path.abspath(init)})
            self.snapshots.append({'path': 'init.pt', 'iter': 0, 'anchor': True, 'score': 0.0, 'games': 0.0})
        else:
            model = PolicyValueNet(d=cfg.d, layers=cfg.layers, heads=cfg.heads, critic=cfg.critic)
        self.model = model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.need_bc = not init and cfg.bc_games > 0

    def _setup_incumbent(self):
        cfg = self.cfg
        self.incumbent_source = None
        if self.incumbent and os.path.exists(self._path(self.incumbent)):
            self.incumbent_source = self._path(self.incumbent)
            return
        self.incumbent = None
        src = cfg.incumbent
        if src == 'auto':
            path = os.path.join(MODELS_DIR, 'battle_singles.pt')
            src = path if cfg.formatid == FORMAT_SINGLES and os.path.exists(path) else None
        else:
            src = repo_path(src) if src else None
        if not src:
            return
        try:
            save_model(load_model(src), self._path('incumbent.pt'), {'source': os.path.abspath(src)})
        except Exception as exc:
            self.log(f'[warn] incumbent {src} not usable ({type(exc).__name__}: {exc}); evaluating vs heuristic only')
            return
        self.incumbent = 'incumbent.pt'
        self.incumbent_source = src

    def _setup_eval_set(self):
        cfg = self.cfg
        n_pairs = max(0, cfg.eval_games // 2)
        es = self.eval_set
        if es and es.get('n_pairs') == n_pairs and es.get('formatid') == cfg.formatid:
            return
        if es:
            self.log(f'[setup] eval_games changed: new fixed evaluation set of {2 * n_pairs} games')
        seed = cfg.seed + 54321
        self.eval_set = {'n_pairs': n_pairs, 'formatid': cfg.formatid, 'seed': seed,
                         'pairs': make_eval_pairs(n_pairs, cfg.formatid, seed)}

    # -- schedules --------------------------------------------------------------------------------------------
    def shaping_at(self, it: int) -> float:
        cfg = self.cfg
        if cfg.shaping_final is None or cfg.shaping_anneal_iters <= 0:
            return cfg.shaping
        frac = min(1.0, max(0, it - 1) / cfg.shaping_anneal_iters)
        return cfg.shaping + (cfg.shaping_final - cfg.shaping) * frac

    def lr_at(self, it: int) -> float:
        cfg = self.cfg
        if cfg.lr_final is None:
            return cfg.lr
        frac = min(1.0, max(0, it - 1) / max(1, cfg.iters - 1))
        return cfg.lr + (cfg.lr_final - cfg.lr) * frac

    # -- league -----------------------------------------------------------------------------------------------
    def _snapshot_weights(self) -> list:
        return [(os.path.abspath(self._path(s['path'])), pfsp_weight(s['score'], s['games'], self.cfg.pfsp_floor))
                for s in self.snapshots]

    def _record_snapshot_results(self, by_snap: dict):
        by_name = {os.path.abspath(self._path(s['path'])): s for s in self.snapshots}
        for s in self.snapshots:
            s['score'] *= self.cfg.pfsp_decay
            s['games'] *= self.cfg.pfsp_decay
        for path, (score, games) in by_snap.items():
            rec = by_name.get(path)
            if rec is not None:
                rec['score'] += score
                rec['games'] += games

    def _add_snapshot(self, it: int):
        name = f'iter{it:04d}.pt'
        save_model(self.model, self._path(name), {'iter': it})
        self.snapshots.append({'path': name, 'iter': it, 'anchor': False, 'score': 0.0, 'games': 0.0})
        rolling = [s for s in self.snapshots if not s['anchor']]
        keep = self.cfg.snapshot_keep
        for s in (rolling[:-keep] if keep > 0 else []):
            self.snapshots.remove(s)
            try:
                os.remove(self._path(s['path']))
            except OSError:
                pass
        # the copies in sync_dir are removed by sync(), after the new state.pt (which no longer refers to
        # them) has been mirrored: the mirror always holds every snapshot its state.pt lists

    # -- phases -----------------------------------------------------------------------------------------------
    def _generator(self, salt: int) -> torch.Generator:
        return torch.Generator().manual_seed((self.cfg.seed * 1_000_003 + salt) % (1 << 62))

    def behaviour_cloning(self):
        cfg = self.cfg
        t = time.time()
        trajs, stats = self.rollout(cfg.bc_games, cfg.seed + 7, [], 'bc')
        n = sum(len(tr['steps']) for tr in trajs)
        self.log(f"[bc] {n} samples from {stats.get('games', 0)} heuristic games in {time.time() - t:.0f}s")
        bc_train(self.model, StepBatch(trajs, self.device), cfg.bc_epochs, cfg.bc_lr, self.minibatch, self.log,
                 gamma=cfg.gamma, shaping=self.shaping_at(1), value_coef=cfg.value_coef, vf_coef=cfg.vf_coef,
                 generator=self._generator(-7))
        del trajs
        self.counters['bc_games'] += stats.get('games', 0)
        save_model(self.model, self._path('bc.pt'), {'iter': 0})
        self.snapshots.append({'path': 'bc.pt', 'iter': 0, 'anchor': True, 'score': 0.0, 'games': 0.0})
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.need_bc = False
        self.evaluate(0)
        self.checkpoint()

    def iteration(self, it: int) -> dict:
        cfg = self.cfg
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.time()
        seed = cfg.seed + 100 + it
        trajs, stats = self.rollout(cfg.games_per_iter, seed, self._snapshot_weights(), 'ppo')
        t1 = time.time()
        games = stats.get('games', 0)
        self.games_per_s = games / max(1e-6, t1 - t0)
        shaping = self.shaping_at(it)
        lr = self.lr_at(it)
        upd, diag = {}, {}
        samples = sum(len(tr['steps']) for tr in trajs)
        if samples:
            for tr in trajs[:1] + trajs[-1:]:  # cheap contract check of the actors' records
                check_trajectory(tr)
            b = StepBatch(trajs, self.device)
            del trajs
            crit, val = predict_values(self.model, b, shaping, chunk=self.minibatch)
            _adv, ret = compute_targets(b, crit, cfg.gamma, cfg.lam, shaping)
            var = float(np.var(ret))
            diag['ev'] = float(1 - np.var(ret - crit) / var) if var > 1e-8 else 0.0
            if np.std(val) > 1e-6 and np.std(b.z_np) > 1e-6:
                diag['v_corr'] = float(np.corrcoef(val, b.z_np)[0, 1])
            for g in self.opt.param_groups:
                g['lr'] = lr
            upd = ppo_update(self.model, self.opt, b, cfg, self.log, shaping=shaping,
                             generator=self._generator(it), minibatch=self.minibatch)
            del b
        t2 = time.time()
        self._record_snapshot_results(stats.get('by_snap', {}))
        self.it = it
        self.counters['games'] += games
        self.counters['samples'] += samples
        dt = t2 - t0
        self.iter_seconds = dt if self.iter_seconds is None else 0.7 * self.iter_seconds + 0.3 * dt
        self.last_iter_seconds = dt
        elapsed = time.time() - self.session_start
        left = self.deadline - time.time() if self.deadline else None
        eta = left if left is not None else ((cfg.iters - it) * self.iter_seconds if cfg.iters < 100000 else None)
        gpu = torch.cuda.max_memory_allocated(self.device) / 2 ** 20 if self.device.type == 'cuda' else None
        opp = ', '.join(f'{k} {w / max(1, n) * 100:.0f}% of {n}' for k, (w, n) in sorted(stats.get('by_opp', {}).items()))
        tp = rollout_throughput(stats, t1 - t0, self.workers)
        line = (f"[it {it}] {games} games {samples} samples | {games / max(1e-6, dt):.2f} games/s "
                f"{samples / max(1e-6, dt):.0f} samples/s | rollout {tp['rollout_games_per_s']:.1f} games/s")
        if 'decisions_per_s' in tp:
            line += f" {tp['decisions_per_s']:.0f} dec/s"
        if 'mean_batch' in tp:
            line += f" batch {tp['mean_batch']:.1f}"
        line += f" | total {self.counters['games']} games | vs {opp or '-'}"
        if upd:
            line += (f" | pl {upd['pl']:.3f} cl {upd['cl']:.3f} vl {upd['vl']:.3f} ent {upd['ent']:.3f} "
                     f"kl {upd['kl']:.4f} clip {upd['clipfrac']:.2f} ev {diag.get('ev', 0):.2f} "
                     f"vcorr {diag.get('v_corr', 0):.2f} ep {upd['epochs']}/{cfg.ppo_epochs}"
                     f"{' (kl stop)' if upd['early_stop'] else ''}")
        line += (f" | lr {lr:.1e} shaping {shaping:.2f} | rollout {t1 - t0:.0f}s update {t2 - t1:.0f}s | "
                 f"elapsed {_fmt_s(elapsed)} {'left' if left is not None else 'eta'} {_fmt_s(eta)}")
        if gpu is not None:
            line += f' | gpu {gpu:.0f}MB'
        if stats.get('errors'):
            line += f" | {stats['errors']} games ended by an engine error"
        self.log(line)
        entry = {'iter': it, 'seed': seed, 'games': games, 'samples': samples, 'turns': stats.get('turns', 0),
                 'errors': stats.get('errors', 0), 'games_per_s': games / max(1e-6, dt),
                 'samples_per_s': samples / max(1e-6, dt), 'rollout_s': t1 - t0, 'update_s': t2 - t1,
                 'total_games': self.counters['games'], 'total_samples': self.counters['samples'],
                 'elapsed_s': elapsed, 'wall_s': self.wall_base + elapsed, 'eta_s': eta,
                 'by_opp': stats.get('by_opp', {}), 'teams': stats.get('teams', {}), 'lr': lr,
                 'shaping': shaping, 'gpu_mem_mb': gpu, 'time': time.time(), **tp, **upd, **diag}
        self._jsonl(entry)
        if cfg.snapshot_every and it % cfg.snapshot_every == 0:
            self._add_snapshot(it)
        return entry

    def evaluate(self, it: int, final: bool = False):
        cfg = self.cfg
        pairs = (self.eval_set or {}).get('pairs') or []
        if cfg.eval_games <= 0 or not pairs:
            return None
        t = time.time()
        inc_path = os.path.abspath(self._path(self.incumbent)) if self.incumbent else None
        opponents = ['heuristic'] + ([inc_path] if inc_path else [])
        res = {opp: MatchResult() for opp in opponents}
        if cfg.vectorized:
            for opp, r in self._vec_eval(pairs, opponents).items():
                res[opp] = res[opp] + r
        else:
            state = _state_bytes(self.model)
            W = self.workers
            per = math.ceil(len(pairs) / W)
            chunks = [pairs[i:i + per] for i in range(0, len(pairs), per)]
            jobs = [(self.wcfg, state, opp, chunk, cfg.seed + 999) for opp in opponents for chunk in chunks]
            for opp, r in self._map(_eval_job, jobs):
                res[opp] = res[opp] + r
        heur, inc = res['heuristic'], res.get(inc_path)
        score, ci = combined_score([heur, inc])
        dt = time.time() - t
        self.eval_seconds = dt
        self.last_eval_it = it
        self.counters['eval_games'] += sum(r.games for r in res.values())
        rec = {'iter': it, 'final': final, 'heuristic': heur.to_dict(), 'incumbent': inc.to_dict() if inc else None,
               'score': score, 'ci95': ci, 'seconds': dt, 'total_games': self.counters['games']}
        self.eval_history.append(rec)
        save_model(self.model, self._path('latest.pt'), {'iter': it, 'eval': rec})
        msg = f'[eval {it}] vs heuristic {heur}'
        if inc:
            msg += f' | vs incumbent {inc}'
        msg += f' | score {score * 100:.1f}% ± {ci * 100:.1f} ({dt:.0f}s)'
        if score > self.best.get('score', -1.0):
            self.best = {'score': score, 'ci95': ci, 'iter': it, 'heuristic': rec['heuristic'],
                         'incumbent': rec['incumbent']}
            save_model(self.model, self._path('best.pt'), {'iter': it, 'eval': rec,
                                                            'win_rate_vs_heuristic': heur.win_rate})
            msg += ' -> new best.pt'
        self.log(msg)
        self._jsonl({'eval': rec, 'time': time.time()})
        return rec

    def _vec_eval(self, pairs: list, opponents: list) -> dict:
        """Fixed-set evaluation with vectorised greedy play: every job plays a contiguous chunk of the pairs
        against every opponent (so the policy's batches stay full).  Results do not depend on the split."""
        ref = self._publish_weights()
        try:
            n_jobs = max(1, min(len(pairs), self.workers * self.cfg.jobs_per_worker))
            bounds = [len(pairs) * j // n_jobs for j in range(n_jobs + 1)]
            jobs = [(self.wcfg, ref, [(opp, i, pairs[i]) for i in range(a, b) for opp in opponents],
                     self.cfg.seed + 999, self._vec_opts()) for a, b in zip(bounds, bounds[1:]) if b > a]
            total = {}
            for res, _stats in self._gather(vec_rollout.eval_job, jobs):
                for opp, r in res.items():
                    total[opp] = total.get(opp, MatchResult()) + r
            return total
        finally:
            self._remove_weights(ref)

    # -- checkpoints ------------------------------------------------------------------------------------------
    def state_dict(self) -> dict:
        self.counters['wall_seconds'] = self.wall_base + (time.time() - self.session_start)
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
        if self.device.type == 'cuda':
            rng['cuda'] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
        return {
            'version': STATE_VERSION, 'it': self.it,
            'model': checkpoint_dict(self.model, {'iter': self.it}),
            'optimizer': _to_cpu(self.opt.state_dict()),
            'counters': dict(self.counters), 'best': self.best, 'snapshots': self.snapshots,
            'eval_set': self.eval_set, 'eval_history': self.eval_history, 'last_eval_it': self.last_eval_it,
            'incumbent': self.incumbent, 'rng': rng, 'cfg': asdict(self.cfg),
            'timing': {'iter_seconds': self.iter_seconds, 'eval_seconds': self.eval_seconds,
                       'games_per_s': self.games_per_s},
            'time': time.time(),
        }

    def checkpoint(self):
        save_model(self.model, self._path('latest.pt'), {'iter': self.it})
        atomic_save(self.state_dict(), self._path('state.pt'))
        self.last_ckpt = time.time()
        self.sync()

    def sync(self):
        sync = self.cfg.sync_dir
        if not sync:
            return
        try:
            os.makedirs(sync, exist_ok=True)
            names = [s['path'] for s in self.snapshots]
            names += [n for n in (self.incumbent, 'best.pt', 'latest.pt', 'final.pt', 'log.jsonl',
                                  self.cfg.log_file, 'config.json') if n]
            for name in names + ['state.pt']:  # state.pt last: it only refers to files already mirrored
                src = self._path(name)
                if not os.path.exists(src):
                    continue
                st = os.stat(src)
                sig = (st.st_size, st.st_mtime_ns)
                dst = os.path.join(sync, name)
                if self._synced.get(name) == sig and os.path.exists(dst):
                    continue
                copy_atomic(src, dst, only_if_changed=False)
                self._synced[name] = sig
            keep = set(names)
            for name in os.listdir(sync):
                if _SNAPSHOT_RE.match(name) and name not in keep:
                    os.remove(os.path.join(sync, name))
        except OSError as exc:  # a flaky network drive must not stop training
            self.log(f'[warn] sync to {sync} failed: {exc}')

    # -- control ----------------------------------------------------------------------------------------------
    def _eval_estimate(self) -> float:
        if self.eval_seconds:
            return self.eval_seconds
        n = 2 * len((self.eval_set or {}).get('pairs') or []) * (2 if self.incumbent else 1)
        return n / self.games_per_s if self.games_per_s else 0.0

    def _budget_allows(self) -> bool:
        """Is there time for one more iteration plus one evaluation (the scheduled one or the final one) and
        the final save?  Uses the measured iteration / evaluation durations."""
        if not self.deadline:
            return True
        margin = min(60.0, 0.05 * self.cfg.time_budget_h * 3600)  # final save, checkpoint and sync
        it_s = max(self.iter_seconds or 0.0, self.last_iter_seconds or 0.0)
        ev_s = self._eval_estimate() if self.cfg.eval_games > 0 else 0.0
        need = 1.1 * (it_s + ev_s) + margin
        return time.time() + need <= self.deadline

    def run(self) -> str:
        cfg = self.cfg
        self._install_signals()
        reason = None
        try:
            resumed = self.setup()
            self._start_pool()
            if self.need_bc:
                self.behaviour_cloning()
            elif not resumed:
                self.checkpoint()
            while True:
                it = self.it + 1
                if self.stop_flag:
                    reason = f'signal {self.stop_signal}'
                    break
                if it > cfg.iters:
                    reason = 'iteration limit'
                    break
                if not self._budget_allows():
                    reason = 'time budget'
                    break
                self.iteration(it)
                if cfg.eval_every and it % cfg.eval_every == 0 and not self.stop_flag:
                    self.evaluate(it)
                    self.checkpoint()
                elif time.time() - self.last_ckpt >= cfg.checkpoint_minutes * 60:
                    self.checkpoint()
            return self._finish(reason)
        finally:
            self._shutdown_pool()
            self._restore_signals()

    def _finish(self, reason: str) -> str:
        self.log(f'[stop] {reason} after iteration {self.it}')
        if not reason.startswith('signal') and self.it > 0 and self.last_eval_it != self.it:
            self.evaluate(self.it, final=True)
        save_model(self.model, self._path('final.pt'), {'iter': self.it})
        best = self._path('best.pt')
        path = best if os.path.exists(best) else self._path('latest.pt')
        b = self.best
        total = _fmt_s(self.wall_base + time.time() - self.session_start)
        if b.get('iter') is not None:
            self.log(f"done: {self.counters['games']} games, {total} trained; best score {b['score'] * 100:.1f}% "
                     f"± {b.get('ci95', 0) * 100:.1f} at iteration {b['iter']} -> {path}")
        else:
            self.log(f"done: {self.counters['games']} games, {total} trained -> {path}")
        self.checkpoint()
        return path


def train(cfg: TrainConfig, init: str | None = None, log=print) -> str:
    """Run (or resume) training; returns the path of the best model (``best.pt``, else ``latest.pt``)."""
    if init is not None:
        cfg = replace(cfg, init=init)
    return Trainer(cfg, log).run()


def _eval_vs_heuristic(model_path: str, cfg: TrainConfig, seed: int, executor=None):
    from .evaluate import evaluate
    return evaluate(_NNFactory(model_path), HeuristicAgent, cfg.eval_games, formatid=cfg.formatid,
                    seed=seed, workers=resolve_workers(cfg.workers), executor=executor)


class _NNFactory:
    """Picklable factory creating a greedy NNAgent from a checkpoint."""

    def __init__(self, path: str, sample: bool = False):
        self.path = path
        self.sample = sample
        self.name = os.path.basename(path)

    def __call__(self, seed=None):
        torch.set_num_threads(1)
        agent = NNAgent(load_model(self.path), sample=self.sample)
        if seed is not None:
            agent.seed(seed)
        return agent
