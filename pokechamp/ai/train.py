"""Training of the battle policy: behaviour cloning from the heuristic agent, then PPO self-play.

Usage (see ``python -m pokechamp train --help``)::

    python -m pokechamp train --format gen9championsbssregmc --out runs/singles --bc-games 3000 --iters 200

Rollouts run in worker processes (one CPU thread each); the learner updates the
network between rollout rounds.  Opponents are sampled from a league: the current
policy (self-play), earlier snapshots and the heuristic agent.  Rewards are +1 / -1
at the end of the battle plus potential-based shaping on the HP balance (which
does not change the optimal policy).
"""
from __future__ import annotations

import io
import json
import multiprocessing
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..env.runner import AgentChoice, FORMAT_SINGLES, gen5_seed, play_battle
from ..teambuilder.random_teams import RandomTeamGenerator
from .features import encode
from .heuristic import HeuristicAgent
from .model import (PolicyValueNet, collate_obs, joint_logits, load_model, pad_options, preview_logits,
                    save_model)
from .nn_agent import NNAgent, potential


@dataclass
class TrainConfig:
    formatid: str = FORMAT_SINGLES
    out: str = 'runs/battle'
    workers: int = 3
    # model
    d: int = 128
    layers: int = 2
    heads: int = 4
    # behaviour cloning
    bc_games: int = 2000
    bc_epochs: int = 4
    # PPO
    iters: int = 100
    games_per_iter: int = 96
    lr: float = 3e-4
    ppo_epochs: int = 3
    minibatch: int = 256
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    shaping: float = 1.0
    opponents: dict = field(default_factory=lambda: {'self': 0.5, 'snapshot': 0.2, 'heuristic': 0.3})
    snapshot_every: int = 5
    eval_every: int = 5
    eval_games: int = 100
    team_pool: str | None = None      # JSON file with teams (e.g. from the evolutionary team builder)
    team_pool_prob: float = 0.5
    seed: int = 0
    max_turns: int = 150


# ---------------------------------------------------------------------------
# Team sources

class TeamSampler:
    def __init__(self, formatid: str, seed: int, pool_path: str | None = None, pool_prob: float = 0.5):
        self.rng = random.Random(seed)
        self.gen = RandomTeamGenerator(self.rng.randrange(1 << 30), formatid=formatid)
        self.pool = []
        self.pool_prob = pool_prob
        if pool_path and os.path.exists(pool_path):
            with open(pool_path, encoding='utf-8') as f:
                data = json.load(f)
            self.pool = data['teams'] if isinstance(data, dict) else data

    def team(self):
        if self.pool and self.rng.random() < self.pool_prob:
            return self.rng.choice(self.pool)
        return self.gen.random_team()


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
# Worker side

_WORKER = {}


def _worker_model(cfg: dict, state: bytes) -> PolicyValueNet:
    torch.set_num_threads(1)
    model = _WORKER.get('model')
    if model is None:
        model = PolicyValueNet(d=cfg['d'], layers=cfg['layers'], heads=cfg['heads'])
        _WORKER['model'] = model
    model.load_state_dict(torch.load(io.BytesIO(state), weights_only=False))
    model.eval()
    return model


def _snapshot_model(path: str) -> PolicyValueNet:
    cache = _WORKER.setdefault('snapshots', {})
    if path not in cache:
        if len(cache) > 8:
            cache.clear()
        cache[path] = load_model(path)
    return cache[path]


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


def _rollout_job(args):
    cfg, state, seed, n_games, snapshots, mode = args
    rng = random.Random(seed)
    torch.manual_seed(seed)
    teams = TeamSampler(cfg['formatid'], rng.randrange(1 << 30), cfg.get('team_pool'), cfg.get('team_pool_prob', 0.5))
    trajs = []
    stats = {'games': 0, 'wins': 0, 'losses': 0, 'by_opp': {}}
    if mode == 'bc':
        teacher = HeuristicAgent(rng.randrange(1 << 30), randomness=0.05)
        for _ in range(n_games):
            a1, a2 = RecordingAgent(teacher), RecordingAgent(HeuristicAgent(rng.randrange(1 << 30), 0.05))
            r = play_battle(teams.team(), teams.team(), a1, a2, cfg['formatid'], seed=gen5_seed(rng),
                            record=True, max_turns=cfg['max_turns'])
            trajs.extend(_trajectories(r, ('p1', 'p2')))
            stats['games'] += 1
        return trajs, stats
    model = _worker_model(cfg, state)
    learner = NNAgent(model, sample=True, record=True)
    learner.seed(rng.randrange(1 << 30))
    opp_names = list(cfg['opponents'])
    opp_w = [cfg['opponents'][k] for k in opp_names]
    for g in range(n_games):
        kind = rng.choices(opp_names, opp_w)[0]
        if kind == 'snapshot' and not snapshots:
            kind = 'self'
        if kind == 'self':
            opp = NNAgent(model, sample=True, record=True)
            opp.seed(rng.randrange(1 << 30))
            sides = ('p1', 'p2')
        elif kind == 'snapshot':
            opp = NNAgent(_snapshot_model(rng.choice(snapshots)), sample=True)
            opp.seed(rng.randrange(1 << 30))
            sides = ('p1',)
        else:
            opp = HeuristicAgent(rng.randrange(1 << 30), randomness=0.05)
            sides = ('p1',)
        swap = rng.random() < 0.5
        t1, t2 = teams.team(), teams.team()
        if swap and kind != 'self':
            r = play_battle(t1, t2, opp, learner, cfg['formatid'], seed=gen5_seed(rng), record=True,
                            max_turns=cfg['max_turns'])
            sides = ('p2',)
        else:
            r = play_battle(t1, t2, learner, opp, cfg['formatid'], seed=gen5_seed(rng), record=True,
                            max_turns=cfg['max_turns'])
        trajs.extend(_trajectories(r, sides))
        stats['games'] += 1
        if kind != 'self':
            me = sides[0]
            won = r.winner == me
            s = stats['by_opp'].setdefault(kind, [0, 0])
            s[0] += int(won)
            s[1] += 1
    return trajs, stats


# ---------------------------------------------------------------------------
# Learner side

def _gae(traj, gamma, lam, shaping):
    steps = traj['steps']
    n = len(steps)
    values = [s.get('value', 0.0) for s in steps]
    phis = [s.get('phi', 0.0) for s in steps]
    adv = [0.0] * n
    last = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            r = traj['z'] + shaping * (0.0 - phis[t])
            next_v = 0.0
        else:
            r = shaping * (gamma * phis[t + 1] - phis[t])
            next_v = values[t + 1]
        delta = r + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    for t in range(n):
        steps[t]['adv'] = adv[t]
        steps[t]['ret'] = adv[t] + values[t]


def _group_stats(logits, idx):
    """Vectorised log-prob of the chosen option, entropy and greedy agreement for padded logits."""
    lp = F.log_softmax(logits, dim=-1)
    p = lp.exp()
    chosen = lp.gather(1, idx.view(-1, 1)).squeeze(1)
    ent = -(p * torch.where(p > 0, lp, torch.zeros_like(lp))).sum(-1)
    agree = (logits.argmax(-1) == idx).float()
    return chosen, ent, agree


def _policy_eval(model, batch_steps, device='cpu'):
    """-> (logp of the chosen options [B], entropy [B], value [B], greedy agreement [B])."""
    obs = collate_obs([s['obs'] for s in batch_steps], device)
    slot_logits, value, x = model(obs)
    B = len(batch_steps)
    logp = torch.zeros(B)
    ent = torch.zeros(B)
    agree = torch.zeros(B)
    move_idx = [i for i, s in enumerate(batch_steps) if s['kind'] == 0]
    prev_idx = [i for i, s in enumerate(batch_steps) if s['kind'] == 1]
    if move_idx:
        combos, mask = pad_options([batch_steps[i]['legal'] for i in move_idx], 2)
        lg = joint_logits(slot_logits[move_idx], combos.to(device), mask.to(device))
        idx = torch.tensor([batch_steps[i]['idx'] for i in move_idx])
        a, b, c = _group_stats(lg, idx)
        mi = torch.tensor(move_idx)
        logp = logp.index_put((mi,), a)
        ent = ent.index_put((mi,), b)
        agree = agree.index_put((mi,), c)
    if prev_idx:
        pick, lead = model.preview_scores(x[prev_idx])
        options, mask = pad_options([batch_steps[i]['legal'] for i in prev_idx], 4)
        n_lead = torch.tensor([batch_steps[i].get('n_lead', 1) for i in prev_idx])
        lg = preview_logits(pick, lead, options.to(device), n_lead.to(device), mask.to(device))
        idx = torch.tensor([batch_steps[i]['idx'] for i in prev_idx])
        a, b, c = _group_stats(lg, idx)
        pi = torch.tensor(prev_idx)
        logp = logp.index_put((pi,), a)
        ent = ent.index_put((pi,), b)
        agree = agree.index_put((pi,), c)
    return logp, ent, value, agree


def bc_train(model, trajs, epochs, lr, minibatch, log=print):
    steps = []
    for tr in trajs:
        for s in tr['steps']:
            s['ret'] = tr['z']
            steps.append(s)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        random.shuffle(steps)
        tot, acc, n = 0.0, 0.0, 0
        for b in range(0, len(steps), minibatch):
            batch = steps[b:b + minibatch]
            logp, _, value, agree = _policy_eval(model, batch)
            ret = torch.tensor([s['ret'] for s in batch], dtype=torch.float32)
            loss = -logp.mean() + 0.5 * F.mse_loss(value, ret)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(batch)
            acc += float(agree.sum())
            n += len(batch)
        log(f"[bc] epoch {ep + 1}/{epochs} loss {tot / n:.4f} teacher-agreement {acc / n * 100:.1f}% "
            f"({n} samples)")
    model.eval()


def ppo_update(model, opt, steps, cfg: TrainConfig, log=print):
    model.train()
    advs = np.array([s['adv'] for s in steps], dtype=np.float32)
    mean, std = advs.mean(), advs.std() + 1e-8
    stats = {'pl': 0.0, 'vl': 0.0, 'ent': 0.0, 'kl': 0.0, 'n': 0}
    for _ in range(cfg.ppo_epochs):
        random.shuffle(steps)
        for b in range(0, len(steps), cfg.minibatch):
            batch = steps[b:b + cfg.minibatch]
            logp, ent, value, _ = _policy_eval(model, batch)
            old = torch.tensor([s['logp'] for s in batch], dtype=torch.float32)
            adv = (torch.tensor([s['adv'] for s in batch], dtype=torch.float32) - mean) / std
            ret = torch.tensor([s['ret'] for s in batch], dtype=torch.float32)
            ratio = torch.exp(logp - old)
            pl = -torch.min(ratio * adv, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
            vl = F.mse_loss(value, ret)
            loss = pl + cfg.vf_coef * vl - cfg.ent_coef * ent.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            k = len(batch)
            stats['pl'] += float(pl.detach()) * k
            stats['vl'] += float(vl.detach()) * k
            stats['ent'] += float(ent.mean().detach()) * k
            stats['kl'] += float((old - logp).mean().detach()) * k
            stats['n'] += k
    model.eval()
    n = max(1, stats['n'])
    return {k: v / n for k, v in stats.items() if k != 'n'}


def _state_bytes(model) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def _run_jobs(ex, cfg_d, state, n_games, workers, seed, snapshots, mode):
    per = max(1, n_games // workers)
    jobs = [(cfg_d, state, seed * 1000 + w, per, snapshots, mode) for w in range(workers)]
    trajs, stats = [], {'games': 0, 'by_opp': {}}
    for t, s in ex.map(_rollout_job, jobs):
        trajs.extend(t)
        stats['games'] += s['games']
        for k, (w, n) in s.get('by_opp', {}).items():
            cur = stats['by_opp'].setdefault(k, [0, 0])
            cur[0] += w
            cur[1] += n
    return trajs, stats


def _eval_vs_heuristic(model_path: str, cfg: TrainConfig, seed: int):
    from .evaluate import evaluate
    return evaluate(_NNFactory(model_path), HeuristicAgent, cfg.eval_games, formatid=cfg.formatid,
                    seed=seed, workers=cfg.workers)


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


def train(cfg: TrainConfig, init: str | None = None, log=print) -> str:
    os.makedirs(cfg.out, exist_ok=True)
    with open(os.path.join(cfg.out, 'config.json'), 'w') as f:
        json.dump(asdict(cfg), f, indent=2)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model = load_model(init) if init else PolicyValueNet(d=cfg.d, layers=cfg.layers, heads=cfg.heads)
    cfg_d = asdict(cfg)
    log_path = os.path.join(cfg.out, 'log.jsonl')
    snapshots = []
    best_path = os.path.join(cfg.out, 'best.pt')
    best_wr = -1.0
    with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=multiprocessing.get_context('spawn')) as ex:
        if cfg.bc_games and not init:
            t = time.time()
            trajs, _ = _run_jobs(ex, cfg_d, b'', cfg.bc_games, cfg.workers, cfg.seed + 7, [], 'bc')
            log(f"[bc] {sum(len(t['steps']) for t in trajs)} samples from {cfg.bc_games} heuristic games "
                f"in {time.time() - t:.0f}s")
            bc_train(model, trajs, cfg.bc_epochs, 1e-3, cfg.minibatch, log)
            path = os.path.join(cfg.out, 'bc.pt')
            save_model(model, path)
            r = _eval_vs_heuristic(path, cfg, cfg.seed + 11)
            log(f"[bc] greedy policy vs heuristic: {r}")
            best_wr = r.win_rate
            save_model(model, best_path, {'win_rate_vs_heuristic': r.win_rate, 'iter': 0})
            snapshots.append(path)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        for it in range(1, cfg.iters + 1):
            t0 = time.time()
            trajs, stats = _run_jobs(ex, cfg_d, _state_bytes(model), cfg.games_per_iter, cfg.workers,
                                     cfg.seed + 100 + it, snapshots, 'ppo')
            for tr in trajs:
                _gae(tr, cfg.gamma, cfg.lam, cfg.shaping)
            steps = [s for tr in trajs for s in tr['steps']]
            t1 = time.time()
            upd = ppo_update(model, opt, steps, cfg, log)
            t2 = time.time()
            opp = {k: f"{w / max(1, n) * 100:.0f}% of {n}" for k, (w, n) in stats['by_opp'].items()}
            log(f"[ppo {it}] games {stats['games']} samples {len(steps)} | win vs {opp} | "
                f"pl {upd['pl']:.3f} vl {upd['vl']:.3f} ent {upd['ent']:.3f} kl {upd['kl']:.4f} | "
                f"rollout {t1 - t0:.0f}s update {t2 - t1:.0f}s")
            entry = {'iter': it, 'samples': len(steps), 'by_opp': stats['by_opp'], **upd}
            if it % cfg.snapshot_every == 0:
                path = os.path.join(cfg.out, f'iter{it:04d}.pt')
                save_model(model, path, {'iter': it})
                snapshots.append(path)
                snapshots = snapshots[-10:]
            if it % cfg.eval_every == 0:
                path = os.path.join(cfg.out, 'latest.pt')
                save_model(model, path, {'iter': it})
                r = _eval_vs_heuristic(path, cfg, cfg.seed + 1000 + it)
                log(f"[eval {it}] greedy policy vs heuristic: {r}")
                entry['eval_vs_heuristic'] = r.win_rate
                if r.win_rate > best_wr:
                    best_wr = r.win_rate
                    save_model(model, best_path, {'win_rate_vs_heuristic': r.win_rate, 'iter': it})
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
    save_model(model, os.path.join(cfg.out, 'final.pt'))
    log(f"done; best greedy win rate vs heuristic {best_wr * 100:.1f}% -> {best_path}")
    return best_path
