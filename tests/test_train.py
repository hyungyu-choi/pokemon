"""Tests of the training pipeline: value / critic targets, checkpoints, resume, time budget, signals, CLI."""
import json
import os
import random
import signal
import time

import numpy as np
import pytest
import torch

from pokechamp.ai import model as model_mod
from pokechamp.ai import train as train_mod
from pokechamp.ai.model import PolicyValueNet, load_model, save_model
from pokechamp.ai.nn_agent import NNAgent
from pokechamp.ai.train import (StepBatch, TrainConfig, _forward_stats, _trajectories, compute_gae,
                                compute_targets, pfsp_weight, ppo_update, predict_values, train)
from pokechamp.env.runner import play_battle
from pokechamp.teambuilder import RandomTeamGenerator, TeamValidator

ROOT = os.path.dirname(os.path.dirname(__file__))
SHIPPED = os.path.join(ROOT, 'models', 'battle_singles.pt')


def _tiny(critic=True, seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet(d=32, layers=1, heads=2, critic=critic)


def _self_play_trajs(model, n_games=2, seed=0, max_turns=30):
    gen = RandomTeamGenerator(seed)
    trajs = []
    for g in range(n_games):
        a, b = NNAgent(model, sample=True, record=True), NNAgent(model, sample=True, record=True)
        a.seed(seed * 100 + g)
        b.seed(seed * 100 + g + 50)
        r = play_battle(gen.random_team(), gen.random_team(), a, b, seed=[1, 2, 3, g], record=True,
                        max_turns=max_turns)
        trajs += _trajectories(r, ('p1', 'p2'))
    return trajs


def _tiny_cfg(out, **kw):
    base = dict(out=str(out), workers=1, device='cpu', d=32, layers=1, heads=2, bc_games=0, iters=2,
                games_per_iter=4, eval_every=2, eval_games=2, snapshot_every=1, max_turns=30, incumbent=None,
                log_file='train.log', seed=3)
    base.update(kw)
    return TrainConfig(**base)


def _jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# targets

def test_gae_returns_are_shaped_outcome():
    phis = [0.0, 0.2, -0.1, 0.4]
    z, gamma, shaping = 1.0, 0.9, 0.5
    adv, ret = compute_gae(phis, np.zeros(4), z, gamma, 1.0, shaping)
    T = len(phis)
    expect = [gamma ** (T - 1 - t) * z - shaping * phis[t] for t in range(T)]
    assert np.allclose(ret, expect)
    assert np.allclose(adv, ret)  # zero baseline
    # with a perfect critic every advantage is zero
    adv2, ret2 = compute_gae(phis, np.array(expect), z, gamma, 0.95, shaping)
    assert np.allclose(adv2, 0.0, atol=1e-12) and np.allclose(ret2, expect)


def test_collated_batch_matches_actor_and_targets():
    """The learner re-evaluates the recorded trajectories exactly as the actor played them; the value head's
    target is the unshaped outcome, the critic's the shaped return; without a critic the baseline falls back
    to value - shaping * phi."""
    for critic in (True, False):
        model = _tiny(critic)
        trajs = _self_play_trajs(model)
        b = StepBatch(trajs, 'cpu')
        assert int((b.kind == 1).sum()) == len(trajs)  # one team preview per side
        with torch.no_grad():
            logp, ent, value, crit, _ = _forward_stats(model, b, b.rows(0, b.n), 0.7)
        assert torch.allclose(logp, b.logp_old, atol=1e-5)
        assert torch.isfinite(ent).all()
        recorded = torch.tensor([s['value'] for tr in trajs for s in tr['steps']])
        assert torch.allclose(value, recorded, atol=1e-5)
        zs = np.concatenate([[tr['z']] * len(tr['steps']) for tr in trajs])
        assert np.allclose(b.z.numpy(), zs)
        if not critic:
            assert torch.allclose(crit, value - 0.7 * b.phi, atol=1e-6)
        c, v = predict_values(model, b, 0.7)
        assert np.allclose(c, crit.numpy(), atol=1e-5)
        compute_targets(b, c, 0.99, 0.95, 0.7)
        # one PPO pass with a zero learning rate reports the losses against exactly these targets
        cfg = TrainConfig(ppo_epochs=1, target_kl=None, ent_coef=0.0, shaping=0.7)
        opt = torch.optim.Adam(model.parameters(), lr=0.0)
        stats = ppo_update(model, opt, b, cfg, minibatch=b.n, generator=torch.Generator().manual_seed(0))
        assert stats['vl'] == pytest.approx(float(((value - b.z) ** 2).mean()), rel=1e-4)
        if critic:
            assert stats['cl'] == pytest.approx(float(((crit - b.ret) ** 2).mean()), rel=1e-4)
        else:
            assert stats['cl'] == 0.0
        assert stats['kl'] == pytest.approx(0.0, abs=1e-6)


def test_ppo_target_kl_stops_early():
    model = _tiny(True)
    b = StepBatch(_self_play_trajs(model), 'cpu')
    compute_targets(b, predict_values(model, b, 1.0)[0], 0.99, 0.95, 1.0)
    cfg = TrainConfig(ppo_epochs=50, target_kl=1e-4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    stats = ppo_update(model, opt, b, cfg, minibatch=64, generator=torch.Generator().manual_seed(0))
    assert stats['early_stop'] and stats['epochs'] < 50 and stats['last_kl'] > 1.5e-4


# ---------------------------------------------------------------------------
# checkpoints

def test_shipped_and_old_checkpoints_load_strictly(tmp_path):
    m = load_model(SHIPPED)
    assert not m.has_critic and m.config['feature_version'] == 1
    # an old-format checkpoint (no 'critic' / 'feature_version' in its config)
    old = _tiny(critic=False)
    cfg = {k: v for k, v in old.config.items() if k not in ('critic', 'feature_version')}
    path = tmp_path / 'old.pt'
    torch.save({'config': cfg, 'state_dict': old.state_dict(), 'vocab': None, 'extra': {}}, path)
    loaded = load_model(str(path))
    assert not loaded.has_critic
    # a checkpoint made for other features is refused
    bad = dict(cfg, feature_version=99)
    torch.save({'config': bad, 'state_dict': old.state_dict(), 'vocab': None, 'extra': {}}, tmp_path / 'bad.pt')
    with pytest.raises(ValueError, match='feature version'):
        load_model(str(tmp_path / 'bad.pt'))


def test_warm_start_adds_critic(tmp_path):
    old = _tiny(critic=False)
    model = load_model(_save(old, tmp_path / 'old.pt'))
    trajs = _self_play_trajs(old, n_games=1)
    obs = model_mod.collate_obs([s['obs'] for s in trajs[0]['steps']])
    with torch.no_grad():
        before = model(obs)
        assert model.enable_critic() and not model.enable_critic()
        after = model(obs)
        crit = model.critic(after[2])
    assert torch.allclose(before[0], after[0]) and torch.allclose(before[1], after[1])
    assert torch.allclose(torch.tanh(crit), after[1], atol=1e-6)  # starts as the value head before tanh
    reloaded = load_model(_save(model, tmp_path / 'new.pt'))
    assert reloaded.has_critic and reloaded.config['critic']


def _save(model, path):
    save_model(model, str(path))
    return str(path)


def test_save_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / 'm.pt'
    m = _tiny()
    save_model(m, str(path))
    good = path.read_bytes()

    def broken_save(obj, f, *a, **k):
        f.write(b'partial')
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', broken_save)
    with pytest.raises(OSError):
        save_model(_tiny(seed=1), str(path))
    assert path.read_bytes() == good
    assert os.listdir(tmp_path) == ['m.pt']  # no temporary file left behind
    sd = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
    assert all(v.device.type == 'cpu' for v in sd.values())


# ---------------------------------------------------------------------------
# runs: resume, time budget, signals

def test_resume_round_trip(tmp_path):
    out = tmp_path / 'run'
    lines = []
    path = train(_tiny_cfg(out, iters=2), log=lines.append)
    assert path.endswith('best.pt') and os.path.exists(path)
    st1 = torch.load(out / 'state.pt', map_location='cpu', weights_only=False)
    assert st1['it'] == 2 and st1['counters']['games'] == 8
    steps1 = max(int(s['step']) for s in st1['optimizer']['state'].values())
    assert steps1 > 0
    assert [s['path'] for s in st1['snapshots']] == ['iter0001.pt', 'iter0002.pt']
    assert st1['eval_history'] and st1['eval_history'][-1]['heuristic']['games'] == 2

    lines2 = []
    train(_tiny_cfg(out, iters=3), log=lines2.append)
    assert any(line.startswith('[resume]') for line in lines2)
    st2 = torch.load(out / 'state.pt', map_location='cpu', weights_only=False)
    assert st2['it'] == 3 and st2['counters']['games'] == 12 and st2['counters']['sessions'] == 2
    assert max(int(s['step']) for s in st2['optimizer']['state'].values()) > steps1
    assert [s['path'] for s in st2['snapshots']] == ['iter0001.pt', 'iter0002.pt', 'iter0003.pt']
    assert st2['eval_set'] == st1['eval_set']  # the same evaluation games for the whole run
    iters = [e for e in _jsonl(out / 'log.jsonl') if 'iter' in e]
    assert [e['iter'] for e in iters] == [1, 2, 3]
    assert len({e['seed'] for e in iters}) == 3  # no rollout seed repeated after resuming
    assert load_model(str(out / 'latest.pt')).has_critic


def test_resume_from_sync_dir_and_rolling_snapshots(tmp_path):
    out, sync = tmp_path / 'run', tmp_path / 'drive'
    train(_tiny_cfg(out, iters=3, snapshot_keep=2, sync_dir=str(sync), eval_every=0), log=lambda m: None)
    assert sorted(f for f in os.listdir(out) if f.startswith('iter')) == ['iter0002.pt', 'iter0003.pt']
    assert sorted(f for f in os.listdir(sync) if f.startswith('iter')) == ['iter0002.pt', 'iter0003.pt']
    for name in ('state.pt', 'latest.pt', 'log.jsonl', 'train.log'):
        assert (sync / name).exists()
    out2 = tmp_path / 'fresh_vm'
    lines = []
    train(_tiny_cfg(out2, iters=4, snapshot_keep=2, sync_dir=str(sync), eval_every=0), log=lines.append)
    assert any('restored' in line for line in lines)
    st = torch.load(out2 / 'state.pt', map_location='cpu', weights_only=False)
    assert st['it'] == 4 and st['counters']['games'] == 16


def test_time_budget_stops_cleanly(tmp_path):
    out = tmp_path / 'run'
    lines = []
    budget = 20.0
    t0 = time.time()
    train(_tiny_cfg(out, iters=1000, eval_every=1000, time_budget_h=budget / 3600), log=lines.append)
    took = time.time() - t0
    st = torch.load(out / 'state.pt', map_location='cpu', weights_only=False)
    assert 1 <= st['it'] < 1000
    assert any('[stop] time budget' in line for line in lines)
    assert st['eval_history'][-1]['final'] and st['eval_history'][-1]['iter'] == st['it']
    assert took < budget + 15


def test_sigterm_finishes_iteration_and_saves(tmp_path):
    out = tmp_path / 'run'
    lines = []

    def log(msg):
        lines.append(msg)
        if msg.startswith('[it 1]'):
            os.kill(os.getpid(), signal.SIGTERM)
    before = signal.getsignal(signal.SIGTERM)
    train(_tiny_cfg(out, iters=50, eval_every=1), log=log)
    assert signal.getsignal(signal.SIGTERM) == before  # handlers restored
    st = torch.load(out / 'state.pt', map_location='cpu', weights_only=False)
    assert st['it'] == 1
    assert any('[stop] signal SIGTERM' in line for line in lines)
    assert not any(line.startswith('[eval') for line in lines)  # no evaluation after the stop request


def test_device_resolution():
    assert train_mod.resolve_device('cpu').type == 'cpu'
    assert train_mod.resolve_device('auto').type == ('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match='CUDA'):
            train_mod.resolve_device('cuda')
    assert train_mod.resolve_workers('auto') == (os.cpu_count() or 1)
    assert train_mod.resolve_workers(3) == 3


# ---------------------------------------------------------------------------
# league, teams, configuration

def test_pfsp_weights_prefer_hard_opponents():
    assert pfsp_weight(2, 20) > pfsp_weight(10, 20) > pfsp_weight(18, 20)
    assert pfsp_weight(20, 20, floor=0.05) == 0.05
    assert pfsp_weight(0, 0) == pytest.approx(0.25)


def test_team_sampler_sources_are_legal():
    pool = os.path.join(ROOT, 'models', 'teams_singles.json')
    usage = os.path.join(ROOT, 'models', 'usage_singles.json')
    ts = train_mod.TeamSampler('gen9championsbssregmc', 1, pool, mix={'random': 1, 'pool': 1, 'usage': 1},
                               usage_path=usage)
    val = TeamValidator()
    for _ in range(30):
        assert val.validate_team(ts.team()) == []
    assert ts.counts['pool'] and ts.counts['usage'] and ts.counts['random']
    # same seed, same stream after reseeding a cached sampler
    ts.reseed(5)
    a = [json.dumps(ts.team(), sort_keys=True) for _ in range(5)]
    b = train_mod.TeamSampler('gen9championsbssregmc', 5, pool, mix={'random': 1, 'pool': 1, 'usage': 1},
                              usage_path=usage)
    assert a == [json.dumps(b.team(), sort_keys=True) for _ in range(5)]
    # missing sources are skipped
    only = train_mod.TeamSampler('gen9championsbssregmc', 1, None, mix={'pool': 1, 'usage': 1}, usage_path=None)
    assert only.mix == {'random': 1.0}
    assert val.validate_team(only.team()) == []


def test_cli_presets_and_config(tmp_path):
    from pokechamp.__main__ import build_parser, train_config_from_args

    def cfg_of(*argv):
        return train_config_from_args(build_parser().parse_args(['train', *argv]))
    d = cfg_of()
    assert d.init is None and d.bc_games == 3000 and d.workers == 'auto' and d.resume == 'auto'
    c = cfg_of('--preset', 'cloud', '--config', '{"lr": 1e-4, "games_per_iter": 64}', '--games-per-iter', '32',
               '--workers', '2', '--device', 'cpu', '--time-budget-h', '11.5', '--sync-dir', 'drive')
    assert c.init == 'models/battle_singles.pt' and c.bc_games == 0 and c.critic and c.iters == 100000
    assert c.lr == 1e-4 and c.games_per_iter == 32 and c.workers == 2 and c.device == 'cpu'
    assert c.time_budget_h == 11.5 and c.sync_dir == 'drive' and c.eval_every == 10 and c.snapshot_every == 5
    f = tmp_path / 'cfg.json'
    f.write_text(json.dumps({'opponents': {'self': 1.0}, 'target_kl': None, 'resume': 'never'}))
    c2 = cfg_of('--preset', 'cloud', '--config', str(f), '--init', 'none')
    assert c2.opponents == {'self': 1.0} and c2.target_kl is None and c2.resume == 'never' and c2.init is None
    with pytest.raises(SystemExit):
        cfg_of('--config', '{"no_such_option": 1}')
    with pytest.raises(SystemExit):
        build_parser().parse_args(['train', '--workers', '0'])
