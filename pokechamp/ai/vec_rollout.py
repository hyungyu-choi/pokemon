"""Vectorised battles: many games per process, one batched network forward per model per step.

Playing one game at a time spends most of the network's time on batch-1 forward passes (a forward of 32
decisions costs about as much as three single ones).  :func:`run_games` keeps ``n_envs`` battles alive at
once.  In each round it collects the decisions every live game is waiting for. Decisions of network players
(:class:`PolicyPlayer`) are queued per model and decided with one :func:`nn_agent.batch_evaluate` call per
model; any other agent (heuristic, recording teacher, ...) decides inline exactly as in
:func:`pokechamp.env.runner.play_battle`.  A finished game is replaced by the next one from the game source
until it is exhausted.

The per-game semantics are those of ``play_battle``: simultaneous choices of every pending side, a rejected
option falls back to the other legal options in order and finally to ``'default'`` (not recorded), the turn
limit, the loop guard, engine errors captured per game (the other games go on), opponent inference trackers
on both sides (``infer``), team preview, and ``start_battle`` / ``end_battle`` hooks.  Each game produces the
same :class:`BattleResult` (with recorded :class:`Decision` objects) ``play_battle`` would, so the training code
turns it into trajectories with the same helper.

Training jobs (:func:`rollout_job`, :func:`eval_job`) run in the trainer's worker processes.  The learner
publishes its weights once per iteration (:func:`publish_weights`, an atomically written file) and every job
carries only a small :class:`WeightsRef`; workers keep the last weights and a few snapshot models cached.
Trajectories travel back compressed (:func:`compress_trajectory`: float16 features, small integer ids,
observations stacked per trajectory, about half the size); the learner restores the documented record format
with :func:`decompress_trajectory`.  Network players created with ``quantize=True`` act on the float16-rounded
observation, so the learner re-evaluates exactly the inputs the actor saw.
"""
from __future__ import annotations

import os
import random
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch

from ..env.runner import AgentChoice, BattleResult, BattleSession, Decision, FORMAT_SINGLES, gen5_seed
from .features import encode
from .model import PolicyValueNet, atomic_save, cpu_state_dict, load_model
from .nn_agent import batch_evaluate
from .shaping import potential

# ---------------------------------------------------------------------------
# Observation compression

FULL_DTYPES = {'ids': np.int64, 'mon': np.float32, 'move_ids': np.int64, 'move': np.float32,
               'glob': np.float32, 'active_idx': np.int64}
COMPACT_DTYPES = {'ids': np.int16, 'mon': np.float16, 'move_ids': np.int16, 'move': np.float16,
                  'glob': np.float16, 'active_idx': np.int8}
_F16_MAX = 60000.0


def quantize_obs(obs: dict) -> dict:
    """Compact copy of an observation: float16 features, int16 ids, int8 active indices.

    Lossless for the ids; the float features are rounded to float16 (relative error <= 2**-11).  Raises
    ValueError when a value does not fit (an id >= 32768 or a feature beyond the float16 range)."""
    out = {}
    for k, dtype in COMPACT_DTYPES.items():
        a = obs[k]
        if a.dtype == dtype:
            out[k] = a
            continue
        if dtype == np.float16:
            if a.size and float(np.abs(a).max()) > _F16_MAX:
                raise ValueError(f'observation {k!r} exceeds the float16 range')
        elif a.size and (int(a.max()) > np.iinfo(dtype).max or int(a.min()) < np.iinfo(dtype).min):
            raise ValueError(f'observation {k!r} does not fit {np.dtype(dtype).name}')
        out[k] = a.astype(dtype)
    return out


def full_obs(obs: dict) -> dict:
    """Observation with the dtypes ``features.encode`` produces (inverse of :func:`quantize_obs`)."""
    return {k: (obs[k] if obs[k].dtype == d else obs[k].astype(d)) for k, d in FULL_DTYPES.items()}


def compress_trajectory(traj: dict) -> dict:
    """Pack a trajectory for the trip from a worker to the learner: the per-step observations become one
    compact array per key (``traj['packed']``, ``[T, ...]``) and are removed from the steps.  Returns the
    same dict (modified in place)."""
    steps = traj['steps']
    if 'packed' in traj or not steps:
        return traj
    q = [quantize_obs(s['obs']) for s in steps]
    traj['packed'] = {k: np.stack([o[k] for o in q]) for k in COMPACT_DTYPES}
    for s in steps:
        del s['obs']
    return traj


def decompress_trajectory(traj: dict) -> dict:
    """Inverse of :func:`compress_trajectory` (in place): every step gets back ``obs`` with the dtypes of
    ``features.encode`` (views into one array per key).  Uncompressed trajectories are returned unchanged."""
    packed = traj.pop('packed', None)
    if packed is None:
        return traj
    full = {k: packed[k].astype(d) for k, d in FULL_DTYPES.items()}
    for t, s in enumerate(traj['steps']):
        s['obs'] = {k: full[k][t] for k in FULL_DTYPES}
    return traj


# ---------------------------------------------------------------------------
# Players and games

@dataclass
class PolicyPlayer:
    """A network player whose decisions are batched by :func:`run_games`.

    ``model``: key into the ``models`` dict of ``run_games``.  ``sample``: sample from the policy (else the
    most likely option).  ``record``: attach the observation and the PPO data (``idx``, ``logp``, ``value``,
    ``phi``, ``kind``, ``n_lead``) to every decision, as :class:`NNAgent` with ``record=True`` does.
    ``rng``: numpy Generator used for sampling (one draw per decision, as ``NNAgent``).  ``quantize``: act on
    the float16-rounded observation (and record it in compact form).
    """
    model: str
    sample: bool = True
    temperature: float = 1.0
    record: bool = False
    rng: object = None
    quantize: bool = False

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng()
        elif not isinstance(self.rng, np.random.Generator):
            self.rng = np.random.default_rng(self.rng)


@dataclass
class GameSpec:
    """One game for :func:`run_games`.  ``p1`` / ``p2``: a :class:`PolicyPlayer` or any agent with
    ``choose(session, sid, view, legal)``.  ``tag``: anything the caller wants back with the result."""
    team1: list
    team2: list
    p1: object
    p2: object
    seed: list | None = None
    formatid: str = FORMAT_SINGLES
    max_turns: int = 200
    record: bool = False
    infer: bool = True
    keep_log: bool = False
    names: tuple = ('p1', 'p2')
    tag: object = None


@dataclass
class RunStats:
    """Counters of :func:`run_games` (summed over calls)."""
    games: int = 0
    rounds: int = 0
    decisions: int = 0          # all decisions (network and inline agents)
    nn_decisions: int = 0       # decisions made by batched network players
    forwards: int = 0           # batched forward passes
    nn_seconds: float = 0.0     # time in batched forwards (incl. collation and sampling)
    errors: int = 0
    by_model: dict = field(default_factory=dict)   # model key -> [forwards, rows]

    def as_dict(self) -> dict:
        return {'games': self.games, 'rounds': self.rounds, 'decisions': self.decisions,
                'nn_decisions': self.nn_decisions, 'forwards': self.forwards, 'nn_seconds': self.nn_seconds,
                'errors': self.errors}


class _Pending:
    """A network decision waiting for its batched forward pass."""
    __slots__ = ('game', 'sid', 'player', 'obs', 'legal', 'kind', 'n_lead', 'phi', 'option', 'info', 'error')

    def __init__(self, game, sid, player, obs, legal, kind, n_lead, phi):
        self.game, self.sid, self.player, self.obs, self.legal = game, sid, player, obs, legal
        self.kind, self.n_lead, self.phi = kind, n_lead, phi
        self.option, self.info, self.error = None, None, None


def _unpack(out):
    if isinstance(out, AgentChoice):
        return out.option, out.obs, out.info
    return out, None, {}


class _Game:
    """One battle stepped by :func:`run_games` with the control flow of ``play_battle``."""

    def __init__(self, spec: GameSpec):
        self.spec = spec
        self.session = BattleSession(spec.formatid, spec.team1, spec.team2, seed=spec.seed, names=spec.names,
                                     infer=spec.infer)
        self.agents = {'p1': spec.p1, 'p2': spec.p2}
        for sid, agent in self.agents.items():
            if hasattr(agent, 'start_battle'):
                agent.start_battle(self.session, sid)
        self.decisions = []
        self.brought = {}
        self.guard = 0
        self.error = None
        self.pending = ()
        self.choices = {}
        self.result = None

    @property
    def done(self) -> bool:
        return self.result is not None

    def _fail(self, exc):
        self.error = f'{type(exc).__name__}: {exc}'
        self._finish()

    def _finish(self):
        if self.result is not None:
            return
        session, battle = self.session, self.session.battle
        for sid, agent in self.agents.items():
            if hasattr(agent, 'end_battle'):
                agent.end_battle(session, sid)
        self.result = BattleResult(
            winner=session.winner if battle.ended else None, turns=battle.turn,
            log=list(battle.log) if self.spec.keep_log else None, decisions=self.decisions,
            teams=(self.spec.team1, self.spec.team2), brought=self.brought, error=self.error)

    def collect(self, queues: dict) -> int:
        """Decide inline agents and queue network decisions (per model key).  Returns the number of decisions
        this game needs this round (0 when it has just finished)."""
        session, battle = self.session, self.session.battle
        try:
            if battle.ended or battle.turn > self.spec.max_turns:
                self._finish()
                return 0
            self.guard += 1
            if self.guard > 5000:
                raise RuntimeError('battle loop guard exceeded')
            pending = session.pending()
            if not pending:
                self._finish()
                return 0
            choices, queued = {}, []
            for sid in pending:
                view = session.view(sid)
                legal = session.legal(sid)
                agent = self.agents[sid]
                if isinstance(agent, PolicyPlayer):
                    req = session.request(sid)
                    obs = encode(view, req)
                    if agent.quantize:
                        obs = quantize_obs(obs)
                    preview = bool(req.get('teamPreview'))
                    item = _Pending(self, sid, agent, obs, legal, 1 if preview else 0, view.n_active,
                                    potential(view) if agent.record else None)
                    choices[sid] = item
                    queued.append(item)
                else:
                    choices[sid] = (agent.choose(session, sid, view, legal), legal)
        except Exception as exc:  # engine / agent error: this game ends, the others go on
            self._fail(exc)
            return 0
        self.pending, self.choices = pending, choices
        for item in queued:
            queues.setdefault(item.player.model, []).append(item)
        return len(pending)

    def apply(self):
        """Apply this round's choices (after the batched decisions are filled in)."""
        session, battle = self.session, self.session.battle
        try:
            for sid in self.pending:
                c = self.choices[sid]
                if isinstance(c, _Pending):
                    if c.error is not None:  # the agent failed to decide (as an exception in choose())
                        raise c.error
                    option, obs, info, legal = c.option, c.obs, c.info, c.legal
                else:
                    (option, obs, info), legal = _unpack(c[0]), c[1]
                req = session.request(sid)
                kind = 'preview' if req.get('teamPreview') else 'move'
                if kind == 'preview':
                    self.brought[sid] = list(option[:session.picked])
                ok = session.apply(sid, option)
                if not ok:
                    remaining = [o for o in legal if o != option]
                    while remaining and not ok:
                        option = remaining.pop(0)
                        ok = session.apply(sid, option)
                    if not ok:
                        session.apply_default(sid)
                        option = None
                if self.spec.record and option is not None:
                    self.decisions.append(Decision(sid, kind, obs, option, legal, info))
                if battle.ended:
                    break
        except Exception as exc:
            self._fail(exc)
            return
        finally:
            self.pending, self.choices = (), {}
        if battle.ended or battle.turn > self.spec.max_turns:
            self._finish()


def _decide(model: PolicyValueNet, items: list):
    """Fill ``option`` / ``info`` of queued network decisions with one batched forward pass."""
    n_lead = [it.n_lead if it.kind == 1 else None for it in items]
    temps = [it.player.temperature for it in items]
    probs, logps, values, _crit = batch_evaluate(model, [it.obs for it in items], [it.legal for it in items],
                                                n_lead, temps)
    for j, it in enumerate(items):
        p = probs[j]
        pl = it.player
        try:
            if pl.sample:
                idx = int(pl.rng.choice(len(it.legal), p=p / p.sum()))
            else:
                idx = int(np.argmax(p))
        except Exception as exc:  # e.g. NaN probabilities: only this game ends (with the error)
            it.error = exc
            continue
        it.option = it.legal[idx]
        if pl.record:
            it.info = {'idx': idx, 'logp': float(logps[j][idx]), 'value': float(values[j]), 'phi': it.phi,
                       'kind': it.kind, 'n_lead': it.n_lead}
        else:
            it.info = {}
            it.obs = None  # nothing to record for this player


def run_games(games, models: dict, n_envs: int = 48, on_result=None, stats: RunStats | None = None) -> list:
    """Play every game of ``games`` (an iterable of :class:`GameSpec`), ``n_envs`` at a time.

    ``models``: model key -> :class:`PolicyValueNet` for the :class:`PolicyPlayer` keys.  ``on_result(spec,
    result)`` is called as each game finishes (in finishing order); without it the results are returned as a
    list of ``(spec, result)``.  ``stats``: optional :class:`RunStats` to add counters to.
    """
    source = iter(games)
    stats = stats if stats is not None else RunStats()
    out = []
    live: list[_Game] = []
    exhausted = False
    n_envs = max(1, int(n_envs))

    def finished(g: _Game):
        stats.games += 1
        stats.errors += int(bool(g.result.error))
        if on_result is not None:
            on_result(g.spec, g.result)
        else:
            out.append((g.spec, g.result))

    while True:
        while not exhausted and len(live) < n_envs:
            spec = next(source, None)
            if spec is None:
                exhausted = True
                break
            live.append(_Game(spec))
        if not live:
            break
        stats.rounds += 1
        queues: dict = {}
        waiting = []
        for g in live:
            n = g.collect(queues)
            if n:
                waiting.append(g)
                stats.decisions += n
            else:
                finished(g)
        if queues:
            t = time.perf_counter()
            for key, items in queues.items():
                model = models[key]
                _decide(model, items)
                stats.forwards += 1
                stats.nn_decisions += len(items)
                fr = stats.by_model.setdefault(key, [0, 0])
                fr[0] += 1
                fr[1] += len(items)
            stats.nn_seconds += time.perf_counter() - t
        live = []
        for g in waiting:
            g.apply()
            if g.done:
                finished(g)
            else:
                live.append(g)
    return out


# ---------------------------------------------------------------------------
# Weights shipped to the workers once per iteration

@dataclass(frozen=True)
class WeightsRef:
    """What a job carries instead of the weights: the file and a unique key (workers cache by key)."""
    path: str
    key: str


def publish_weights(model: PolicyValueNet, path: str) -> WeightsRef:
    """Write ``{'config', 'state_dict'}`` (CPU tensors) atomically to ``path`` -> a :class:`WeightsRef`."""
    atomic_save({'config': dict(model.config), 'state_dict': cpu_state_dict(model)}, path)
    return WeightsRef(os.path.abspath(path), uuid.uuid4().hex)


_CACHE: dict = {'weights_key': None, 'model': None, 'arch': None, 'snapshots': OrderedDict(), 'loads': 0}
SNAPSHOT_CACHE = 8


def worker_model(ref: WeightsRef) -> PolicyValueNet:
    """The model for ``ref`` in this worker: loaded once per key (CPU), the module reused across keys with
    the same architecture."""
    if _CACHE['weights_key'] == ref.key and _CACHE['model'] is not None:
        return _CACHE['model']
    data = torch.load(ref.path, map_location='cpu', weights_only=False)
    conf, sd = data['config'], data['state_dict']
    arch = (conf['d'], conf['layers'], conf['heads'], bool(conf.get('critic', False)),
            tuple(sorted((conf.get('sizes') or {}).items())))
    model = _CACHE['model']
    if model is None or _CACHE['arch'] != arch:
        model = PolicyValueNet(d=conf['d'], layers=conf['layers'], heads=conf['heads'], sizes=conf.get('sizes'),
                               critic=bool(conf.get('critic', False)),
                               feature_version=int(conf.get('feature_version', 1)))
        _CACHE['arch'] = arch
    model.load_state_dict(sd)
    model.eval()
    _CACHE['model'], _CACHE['weights_key'] = model, ref.key
    _CACHE['loads'] += 1
    return model


def snapshot_model(path: str) -> PolicyValueNet:
    """A checkpoint loaded with ``load_model`` (CPU), cached per worker (least recently used, bounded)."""
    cache = _CACHE['snapshots']
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    model = cache.get(key)
    if model is None:
        model = load_model(path, map_location='cpu')
        cache[key] = model
        while len(cache) > SNAPSHOT_CACHE:
            cache.popitem(last=False)
    else:
        cache.move_to_end(key)
    return model


def clear_worker_cache():
    _CACHE.update({'weights_key': None, 'model': None, 'arch': None, 'loads': 0})
    _CACHE['snapshots'].clear()


# ---------------------------------------------------------------------------
# Training jobs (run in the trainer's worker processes)

def _score(winner, me) -> float:
    return 1.0 if winner == me else (0.5 if winner is None else 0.0)


def rollout_job(args):
    """Vectorised version of ``train._rollout_job``: play ``n_games`` and return ``(trajectories, stats)`` in
    the documented record format, trajectories compressed (see :func:`decompress_trajectory`).

    ``args = (cfg, weights, seed, n_games, snapshots, mode, opts)``: ``weights`` a :class:`WeightsRef` (None
    for behaviour cloning), ``snapshots`` ``[(path, pfsp weight)]``, ``mode`` 'ppo' or 'bc', ``opts``
    ``{'envs': games played at once, 'compress': bool, 'snapshots_per_job': int}``.

    Every game draws its opponent, teams and battle seed from the job's seed in order and gets its own
    sampling generators, so the games do not depend on how many are played at once.  Snapshot opponents of
    one job come from ``snapshots_per_job`` snapshots drawn by their PFSP weights (fewer models, bigger
    batches; across jobs each snapshot is still drawn in proportion to its weight).
    """
    from .heuristic import HeuristicAgent
    from .train import RecordingAgent, _team_sampler, _trajectories
    cfg, weights, seed, n_games, snapshots, mode, opts = args
    opts = opts or {}
    t_start = time.perf_counter()
    envs = int(opts.get('envs', 48))
    compress = bool(opts.get('compress', True))
    rng = random.Random(seed)
    torch.manual_seed(seed)
    teams = _team_sampler(cfg, rng.randrange(1 << 30))
    max_turns = cfg.get('max_turns', 150)
    formatid = cfg['formatid']
    trajs = []
    stats = {'games': 0, 'turns': 0, 'errors': 0, 'by_opp': {}, 'by_snap': {}, 'teams': {}}
    run = RunStats()

    def keep(result, sides):
        for tr in _trajectories(result, sides):
            trajs.append(compress_trajectory(tr) if compress else tr)
        stats['games'] += 1
        stats['turns'] += result.turns
        stats['errors'] += int(bool(result.error))

    if mode == 'bc':
        def bc_games():
            for _ in range(n_games):
                a1 = RecordingAgent(HeuristicAgent(rng.randrange(1 << 30), randomness=0.05))
                a2 = RecordingAgent(HeuristicAgent(rng.randrange(1 << 30), 0.05))
                yield GameSpec(teams.team(), teams.team(), a1, a2, seed=gen5_seed(rng), formatid=formatid,
                               max_turns=max_turns, record=True, tag=('p1', 'p2'))
        run_games(bc_games(), {}, envs, lambda spec, r: keep(r, spec.tag), run)
    else:
        models = {'learner': worker_model(weights)}
        snaps = [(s, 1.0) if isinstance(s, str) else (s[0], float(s[1])) for s in snapshots or []]
        if snaps:
            k = max(1, int(opts.get('snapshots_per_job', 1)))
            snaps = [(p, 1.0) for p in rng.choices([p for p, _ in snaps], [max(w, 1e-6) for _, w in snaps], k=k)]
        opp_names = list(cfg['opponents'])
        opp_w = [cfg['opponents'][k] for k in opp_names]

        def player(key, record):
            return PolicyPlayer(key, sample=True, record=record, rng=rng.randrange(1 << 30), quantize=compress)

        def ppo_games():
            for _ in range(n_games):
                kind = rng.choices(opp_names, opp_w)[0]
                if kind == 'snapshot' and not snaps:
                    kind = 'self'
                snap = None
                learner = player('learner', True)
                if kind == 'self':
                    opp = player('learner', True)
                elif kind == 'snapshot':
                    snap = rng.choice([p for p, _ in snaps])
                    if snap not in models:
                        models[snap] = snapshot_model(snap)
                    opp = player(snap, False)
                else:
                    opp = HeuristicAgent(rng.randrange(1 << 30), randomness=cfg.get('heuristic_randomness', 0.05))
                swap = kind != 'self' and rng.random() < 0.5
                t1, t2 = teams.team(), teams.team()
                p1, p2, sides = (opp, learner, ('p2',)) if swap else (learner, opp, ('p1', 'p2') if kind == 'self'
                                                                       else ('p1',))
                yield GameSpec(t1, t2, p1, p2, seed=gen5_seed(rng), formatid=formatid, max_turns=max_turns,
                               record=True, tag=(kind, snap, sides))

        def done(spec, result):
            kind, snap, sides = spec.tag
            keep(result, sides)
            if kind != 'self':
                score = _score(result.winner, sides[0])
                for group, key in (('by_opp', kind), ('by_snap', snap)):
                    if key is not None:
                        cur = stats[group].setdefault(key, [0.0, 0])
                        cur[0] += score
                        cur[1] += 1

        run_games(ppo_games(), models, envs, done, run)
    stats['teams'] = dict(teams.counts)
    stats.update({'decisions': run.decisions, 'nn_decisions': run.nn_decisions, 'forwards': run.forwards,
                  'nn_seconds': run.nn_seconds, 'job_seconds': time.perf_counter() - t_start, 'jobs': 1})
    return trajs, stats


def eval_seed(seed: int, pair_index: int, swap: bool) -> int:
    """Seed of the heuristic opponent in one evaluation game (independent of how the pairs are split)."""
    return (int(seed) * 1_000_003 + 2 * int(pair_index) + int(bool(swap))) % (1 << 31)


def eval_job(args):
    """Greedy policy (always p1) on fixed evaluation pairs against several opponents, vectorised.

    ``args = (cfg, weights, games, seed, opts)``; ``games``: list of ``(opponent, pair_index, pair)`` where
    ``opponent`` is 'heuristic' (``HeuristicAgent(randomness=0)``, seeded per game with :func:`eval_seed`) or a
    checkpoint path (greedy network), and each pair is played with both team assignments and the pair's
    battle seed.  Returns ``({opponent: MatchResult}, stats)``.
    """
    from .evaluate import MatchResult
    from .heuristic import HeuristicAgent
    cfg, weights, games, seed, opts = args
    opts = opts or {}
    t_start = time.perf_counter()
    models = {'policy': worker_model(weights)}
    res = {}
    run = RunStats()
    max_turns = cfg.get('eval_max_turns', 200)

    def specs():
        for opp, i, pair in games:
            res.setdefault(opp, MatchResult())
            if opp != 'heuristic' and opp not in models:
                models[opp] = snapshot_model(opp)
            t1, t2 = pair['teams']
            for swap in (False, True):
                ta, tb = (t2, t1) if swap else (t1, t2)
                me = PolicyPlayer('policy', sample=False)
                other = (HeuristicAgent(eval_seed(seed, i, swap), randomness=0.0) if opp == 'heuristic'
                         else PolicyPlayer(opp, sample=False))
                yield GameSpec(ta, tb, me, other, seed=list(pair['seed']), formatid=cfg['formatid'],
                               max_turns=max_turns, tag=opp)

    def done(spec, r):
        m = res[spec.tag]
        m.turns += r.turns
        if r.error:
            m.errors += 1
        if r.winner == 'p1':
            m.wins += 1
        elif r.winner == 'p2':
            m.losses += 1
        else:
            m.ties += 1

    run_games(specs(), models, int(opts.get('envs', 48)), done, run)
    stats = {'decisions': run.decisions, 'nn_decisions': run.nn_decisions, 'forwards': run.forwards,
             'nn_seconds': run.nn_seconds, 'job_seconds': time.perf_counter() - t_start, 'games': run.games}
    return res, stats
