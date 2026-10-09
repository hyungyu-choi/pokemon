"""Tests of the vectorised actors: same games as play_battle, record format, compression, worker caches."""
import json
import os

import numpy as np
import pytest
import torch

from pokechamp.ai import train as train_mod
from pokechamp.ai import vec_rollout as V
from pokechamp.ai.evaluate import make_eval_pairs, play_pairs
from pokechamp.ai.heuristic import HeuristicAgent
from pokechamp.ai.model import PolicyValueNet, load_model, save_model
from pokechamp.ai.nn_agent import NNAgent, batch_evaluate
from pokechamp.ai.train import StepBatch, TrainConfig, _forward_stats, _trajectories, check_trajectory, train
from pokechamp.env.runner import BattleSession, play_battle
from pokechamp.teambuilder import RandomTeamGenerator

FMT = 'gen9championsbssregmc'


def _tiny(critic=True, seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet(d=32, layers=1, heads=2, critic=critic)


def _wcfg(**kw):
    cfg = {'formatid': FMT, 'max_turns': 30, 'opponents': {'self': 1.0, 'snapshot': 1.0, 'heuristic': 1.0},
           'heuristic_randomness': 0.05, 'team_pool': None, 'team_pool_prob': 0.5, 'usage_stats': None,
           'team_mix': {'random': 1.0}, 'd': 32, 'layers': 1, 'heads': 2}
    cfg.update(kw)
    return cfg


def _decisions(result):
    return [(d.side, d.kind, d.action) for d in result.decisions]


# ---------------------------------------------------------------------------
# the runner plays the same games as play_battle

def test_vectorised_games_match_play_battle():
    """Seeded games played 4 at a time (network players batched) reach exactly the outcome of play_battle with
    the same agents and seeds: same decisions, observations, log-probabilities and values."""
    model = _tiny()
    gen = RandomTeamGenerator(11)
    setups = []
    for g in range(4):
        t1, t2 = gen.random_team(), gen.random_team()
        setups.append((t1, t2, [5, 6, 7, g], g))

    def seq_agents(g):
        nn_a = NNAgent(model, sample=True, record=True)
        nn_a.seed(100 + g)
        nn_b = NNAgent(model, sample=True, record=True)
        nn_b.seed(200 + g)
        return [(nn_a, HeuristicAgent(300 + g, 0.05)), (nn_a, nn_b), (HeuristicAgent(300 + g, 0.05), nn_b),
                (HeuristicAgent(300 + g, 0.05), HeuristicAgent(400 + g, 0.05))][g]

    def vec_agents(g):
        a = V.PolicyPlayer('m', sample=True, record=True, rng=100 + g)
        b = V.PolicyPlayer('m', sample=True, record=True, rng=200 + g)
        return [(a, HeuristicAgent(300 + g, 0.05)), (a, b), (HeuristicAgent(300 + g, 0.05), b),
                (HeuristicAgent(300 + g, 0.05), HeuristicAgent(400 + g, 0.05))][g]

    expected = []
    for t1, t2, seed, g in setups:
        a1, a2 = seq_agents(g)
        expected.append(play_battle(t1, t2, a1, a2, FMT, seed=list(seed), record=True, max_turns=40))
    for n_envs in (4, 1):
        specs = []
        for t1, t2, seed, g in setups:
            p1, p2 = vec_agents(g)
            specs.append(V.GameSpec(t1, t2, p1, p2, seed=list(seed), formatid=FMT, max_turns=40, record=True, tag=g))
        stats = V.RunStats()
        got = {spec.tag: r for spec, r in V.run_games(specs, {'m': model}, n_envs, stats=stats)}
        assert stats.games == 4 and stats.errors == 0
        if n_envs == 4:
            assert stats.nn_decisions / stats.forwards > 1.0  # decisions of several games share a forward
        for g, exp in enumerate(expected):
            r = got[g]
            assert r.error is None and exp.error is None
            assert (r.winner, r.turns, r.brought) == (exp.winner, exp.turns, exp.brought)
            assert _decisions(r) == _decisions(exp)
            for d, e in zip(r.decisions, exp.decisions):
                assert d.action in d.legal and d.legal == e.legal
                assert (d.obs is None) == (e.obs is None)
                if d.obs is None:
                    continue
                for k in e.obs:
                    assert np.array_equal(d.obs[k], e.obs[k])
                assert d.info['idx'] == e.info['idx'] and d.info['kind'] == e.info['kind']
                assert d.info['phi'] == e.info['phi'] and d.info['n_lead'] == e.info['n_lead']
                assert d.info['logp'] == pytest.approx(e.info['logp'], abs=1e-4)
                assert d.info['value'] == pytest.approx(e.info['value'], abs=1e-5)


def test_engine_error_ends_only_that_game(monkeypatch):
    """An exception inside one game is captured in its result (as play_battle does); the others finish."""
    model = _tiny()
    gen = RandomTeamGenerator(3)

    class Exploding:
        def choose(self, session, sid, view, legal):
            if session.battle.turn >= 2:
                raise RuntimeError('boom')
            return legal[0]

    specs = [V.GameSpec(gen.random_team(), gen.random_team(), V.PolicyPlayer('m', rng=g), Exploding() if g == 1 else
                        HeuristicAgent(g), seed=[1, 2, 3, g], formatid=FMT, max_turns=30, tag=g) for g in range(3)]
    res = {s.tag: r for s, r in V.run_games(specs, {'m': model}, 3)}
    assert res[1].error == 'RuntimeError: boom' and res[1].winner is None
    assert res[0].error is None and res[2].error is None
    assert res[0].winner is not None or res[0].turns > 30


def test_failed_network_decision_ends_only_that_game():
    """A network that outputs NaN cannot be sampled from: like an exception in NNAgent.choose under play_battle,
    the game ends with the error recorded, the job goes on."""
    model = _tiny()
    with torch.no_grad():
        model.move_head[-1].bias.fill_(float('nan'))
    gen = RandomTeamGenerator(8)
    t1, t2 = gen.random_team(), gen.random_team()
    nn = NNAgent(model, sample=True)
    nn.seed(1)
    exp = play_battle(t1, t2, nn, HeuristicAgent(2), FMT, seed=[3, 3, 3, 3], max_turns=30)
    specs = [V.GameSpec(t1, t2, V.PolicyPlayer('m', rng=1), HeuristicAgent(2), seed=[3, 3, 3, 3], formatid=FMT,
                        max_turns=30), V.GameSpec(t1, t2, HeuristicAgent(4), HeuristicAgent(5), seed=[1, 1, 1, 1],
                                                  formatid=FMT, max_turns=30)]
    (s1, r1), (s2, r2) = sorted(V.run_games(specs, {'m': model}, 2), key=lambda x: x[0] is specs[1])
    assert exp.error and r1.error == exp.error and r1.turns == exp.turns and r1.winner is None
    assert r2.error is None


def test_batch_evaluate_matches_single_decisions():
    """One forward over mixed team-preview and move decisions gives each decision's single-forward result."""
    model = _tiny()
    agent = NNAgent(model)
    gen = RandomTeamGenerator(5)
    s = BattleSession(FMT, gen.random_team(), gen.random_team(), seed=[1, 1, 1, 1])
    items = []  # (obs, legal, n_lead or None, single-forward probs, value)
    for _round in range(2):
        pending = s.pending()
        for sid in pending:
            view, req, legal = s.view(sid), s.request(sid), s.legal(sid)
            p, value, obs = agent.evaluate(view, req, legal)
            items.append((obs, legal, view.n_active if req.get('teamPreview') else None, p, value))
        for sid in pending:
            s.apply(sid, s.legal(sid)[0])
    n_lead = [it[2] for it in items]
    assert any(n is None for n in n_lead) and any(n is not None for n in n_lead)
    probs, logps, values, crit = batch_evaluate(model, [it[0] for it in items], [it[1] for it in items], n_lead,
                                                critic=True)
    assert crit is not None and crit.shape == (len(items),)
    for i, (_obs, _legal, _n, p, value) in enumerate(items):
        assert np.allclose(probs[i], p, atol=1e-6) and np.allclose(np.exp(logps[i]), p, atol=1e-6)
        assert values[i] == pytest.approx(value, abs=1e-6)


# ---------------------------------------------------------------------------
# training jobs, record format, compression

def test_rollout_job_records_compression_and_k_independence(tmp_path):
    model = _tiny()
    ref = V.publish_weights(model, str(tmp_path / '_weights_1.pt'))
    snap = str(tmp_path / 'snap.pt')
    save_model(_tiny(seed=1), snap)
    V.clear_worker_cache()
    job = (_wcfg(), ref, 7, 6, [(snap, 1.0)], 'ppo', {'envs': 3, 'compress': True})
    trajs, stats = V.rollout_job(job)
    assert stats['games'] == 6 and stats['errors'] == 0
    assert stats['decisions'] > 0 and stats['forwards'] > 0 and stats['nn_decisions'] <= stats['decisions']
    assert sum(n for _s, n in stats['by_opp'].values()) <= 6
    assert set(stats['by_snap']) <= {snap}
    for tr in trajs:
        assert 'packed' in tr and all('obs' not in s for s in tr['steps'])
        assert tr['packed']['mon'].dtype == np.float16 and tr['packed']['ids'].dtype == np.int16
    raw = sum(sum(a.nbytes for a in tr['packed'].values()) for tr in trajs)
    n_steps = sum(len(tr['steps']) for tr in trajs)
    assert raw < 0.55 * n_steps * 13_600
    for tr in trajs:
        V.decompress_trajectory(tr)
        check_trajectory(tr)
    # the learner re-evaluates exactly what the actors saw (quantised observations): same log-probabilities
    b = StepBatch(trajs, 'cpu')
    with torch.no_grad():
        logp, *_ = _forward_stats(model, b, b.rows(0, b.n), 1.0)
    assert torch.allclose(logp, b.logp_old, atol=1e-4)
    # the games do not depend on how many are played at once
    trajs1, stats1 = V.rollout_job((_wcfg(), ref, 7, 6, [(snap, 1.0)], 'ppo', {'envs': 1, 'compress': True}))
    for tr in trajs1:
        V.decompress_trajectory(tr)

    def key(trs):
        return sorted((tr['z'], tuple(s['idx'] for s in tr['steps'])) for tr in trs)
    assert key(trajs1) == key(trajs) and stats1['by_opp'] == stats['by_opp']


def test_bc_job_matches_recording_agent_records():
    trajs, stats = V.rollout_job((_wcfg(), None, 3, 2, [], 'bc', {'envs': 2, 'compress': False}))
    assert stats['games'] == 2 and len(trajs) == 4
    for tr in trajs:
        check_trajectory(tr, need_logp=False)


def test_compression_round_trip():
    gen = RandomTeamGenerator(9)
    r = play_battle(gen.random_team(), gen.random_team(), train_mod.RecordingAgent(HeuristicAgent(1)),
                    train_mod.RecordingAgent(HeuristicAgent(2)), FMT, seed=[9, 9, 9, 9], record=True, max_turns=20)
    trajs = _trajectories(r, ('p1', 'p2'))
    originals = [[{k: a.copy() for k, a in s['obs'].items()} for s in tr['steps']] for tr in trajs]
    for tr, orig in zip([V.compress_trajectory(tr) for tr in trajs], originals):
        V.decompress_trajectory(tr)
        check_trajectory(tr, need_logp=False)
        for s, o in zip(tr['steps'], orig):
            for k, a in o.items():
                got = s['obs'][k]
                assert got.dtype == a.dtype and got.shape == a.shape
                if a.dtype == np.float32:
                    assert np.array_equal(got, a.astype(np.float16).astype(np.float32))
                    assert np.allclose(got, a, rtol=1e-3, atol=1e-4)
                else:
                    assert np.array_equal(got, a)
    # quantising twice changes nothing; values that do not fit are refused
    q = V.quantize_obs(originals[0][0])
    assert all(V.quantize_obs(q)[k] is q[k] for k in q)
    bad = dict(originals[0][0], ids=originals[0][0]['ids'] + 40000)
    with pytest.raises(ValueError):
        V.quantize_obs(bad)
    assert V.decompress_trajectory({'z': 0.0, 'steps': []}) == {'z': 0.0, 'steps': []}


def test_worker_weight_and_snapshot_caches(tmp_path, monkeypatch):
    V.clear_worker_cache()
    m1, m2 = _tiny(seed=1), _tiny(seed=2)
    ref1 = V.publish_weights(m1, str(tmp_path / '_weights_1.pt'))
    a = V.worker_model(ref1)
    assert V._CACHE['loads'] == 1
    os.remove(ref1.path)  # cached by key: the file is not read again
    assert V.worker_model(ref1) is a and V._CACHE['loads'] == 1
    ref2 = V.publish_weights(m2, str(tmp_path / '_weights_2.pt'))
    b = V.worker_model(ref2)
    assert V._CACHE['loads'] == 2 and b is a  # same architecture: the module is reused, weights replaced
    for p, q in zip(b.parameters(), m2.parameters()):
        assert torch.equal(p, q)
    assert ref1.key != ref2.key and ref2.path == os.path.abspath(str(tmp_path / '_weights_2.pt'))
    # snapshots: loaded once per file, least recently used dropped beyond SNAPSHOT_CACHE
    monkeypatch.setattr(V, 'SNAPSHOT_CACHE', 2)
    paths = []
    for i in range(3):
        p = str(tmp_path / f'snap{i}.pt')
        save_model(_tiny(seed=i), p)
        paths.append(p)
    s0 = V.snapshot_model(paths[0])
    assert V.snapshot_model(paths[0]) is s0
    V.snapshot_model(paths[1])
    V.snapshot_model(paths[2])
    assert len(V._CACHE['snapshots']) == 2 and V.snapshot_model(paths[0]) is not s0
    V.clear_worker_cache()


def test_vectorised_eval_independent_of_split(tmp_path):
    model = _tiny()
    ref = V.publish_weights(model, str(tmp_path / '_weights_1.pt'))
    inc = str(tmp_path / 'inc.pt')
    save_model(_tiny(seed=4), inc)
    pairs = make_eval_pairs(3, FMT, seed=2)
    games = [(opp, i, pairs[i]) for i in range(3) for opp in ('heuristic', inc)]
    cfg = _wcfg(eval_max_turns=40)
    whole, st = V.eval_job((cfg, ref, games, 5, {'envs': 8}))
    assert st['games'] == 12 and whole['heuristic'].games == 6 and whole[inc].games == 6
    parts = [V.eval_job((cfg, ref, games[:2], 5, {'envs': 2}))[0], V.eval_job((cfg, ref, games[2:], 5, {'envs': 3}))[0]]
    for opp in ('heuristic', inc):
        total = parts[0][opp] + parts[1][opp]
        assert total.to_dict() == whole[opp].to_dict()
    # against a network opponent both greedy policies are deterministic: same as the sequential evaluation
    seq = play_pairs(NNAgent(model), NNAgent(load_model(inc)), pairs, FMT, max_turns=40)
    assert seq.to_dict() == whole[inc].to_dict()


# ---------------------------------------------------------------------------
# trainer integration

def _tiny_cfg(out, **kw):
    base = dict(out=str(out), workers=1, device='cpu', d=32, layers=1, heads=2, bc_games=0, iters=1,
                games_per_iter=4, eval_every=1, eval_games=2, snapshot_every=1, max_turns=30, incumbent=None,
                log_file='train.log', seed=3, team_mix={'random': 1.0})
    base.update(kw)
    return TrainConfig(**base)


@pytest.mark.parametrize('vectorized', [True, False])
def test_trainer_runs_both_actor_paths(tmp_path, vectorized):
    out = tmp_path / 'run'
    lines = []
    train(_tiny_cfg(out, vectorized=vectorized, envs_per_worker=3), log=lines.append)
    entries = [json.loads(line) for line in open(out / 'log.jsonl')]
    it = [e for e in entries if 'iter' in e][0]
    assert it['games'] == 4 and it['rollout_games_per_s'] > 0 and it['decisions'] > 0
    assert ('mean_batch' in it) == vectorized
    assert any(line.startswith('[it 1]') and 'rollout' in line for line in lines)
    ev = [e['eval'] for e in entries if 'eval' in e]
    assert ev and ev[-1]['heuristic']['games'] == 2
    assert not [n for n in os.listdir(out) if n.startswith('_weights_')]  # published weights are cleaned up


def test_job_split(tmp_path):
    t = train_mod.Trainer(_tiny_cfg(tmp_path / 'r', workers=4, envs_per_worker=32, jobs_per_worker=3),
                          log=lambda m: None)
    for n, n_jobs in ((512, 12), (256, 8), (100, 4), (3, 3), (1, 1)):
        sizes = t.job_sizes(n)
        assert sum(sizes) == n and len(sizes) == n_jobs and max(sizes) - min(sizes) <= 1
    with pytest.raises(ValueError):
        train_mod.build_config('default', {'envs_per_worker': 0})


# ---------------------------------------------------------------------------
# cached feature helpers give exactly the values of the original formulas

def test_encode_caches_match_reference_formulas():
    from pokechamp.ai import features as F
    from pokechamp.ai import inference as I
    from pokechamp.ai.damage import move_info
    from pokechamp.env import view as view_mod
    from pokechamp.sim.dex import get_dex
    dex = get_dex()
    n = 0
    for mid in sorted(dex.move_data):
        move = move_info(mid)
        if move is None:
            assert F._move_static(mid) is None
            continue
        acc = move.accuracy
        head = [min(move.basePower or 0, 250) / 150.0, 1.0 if acc is True else acc / 100.0,
                (move.priority or 0) / 5.0] + [1.0 if move.category == c else 0.0 for c in F.CATEGORIES]
        head += [1.0 if move.type == t else 0.0 for t in F.TYPES]
        tail = (1.0 if move.flags.get('contact') else 0.0, 1.0 if (move.self or {}).get('boosts') or move.boosts else 0.0,
                1.0 if move.heal or move.flags.get('heal') else 0.0, 1.0 if move.status or move.volatileStatus else 0.0,
                1.0 if move.sideCondition else 0.0, 1.0 if move.selfSwitch else 0.0, 1.0 if move.drain else 0.0,
                1.0 if move.recoil else 0.0)
        spread = 1.0 if move.target in ('allAdjacentFoes', 'allAdjacent', 'all') else 0.0
        assert F._move_static(mid) == (tuple(head), spread, tail), mid
        assert len(head) + 4 + 4 + len(tail) == F.MOVE_F
        n += 1
    assert n > 100
    # speed quantiles from the shared sort order == a fresh argsort of this belief's speeds
    rng = np.random.default_rng(0)
    for species in ('Garchomp', 'Dragonite', 'Gengar'):
        b = I.StatBelief()
        b._update(rng.uniform(0.05, 1.0, size=len(b.w)))
        spe = b._stats(species)[:, 5] * 1.0 * np.where(b.scarf, 1.5, 1.0)
        order = np.argsort(spe)
        cw = np.cumsum(b.w[order])
        expect = (float(spe[order][np.searchsorted(cw, 0.1)]),
                  float(spe[order][min(len(cw) - 1, np.searchsorted(cw, 0.9))]))
        assert b.speed_quantiles(species) == expect
    for table, name in ((dex.items, 'choicescarf'), (dex.abilities, 'Intimidate'), (dex.moves, 'U-turn'),
                        (dex.abilities, ''), (dex.moves, 'notamove')):
        assert view_mod._dex_name(table, name) == table.get(name).name
    assert view_mod._species_types(dex.species, 'Garchomp') == tuple(dex.species.get('Garchomp').types)
    assert view_mod._species_types(dex.species, 'Notamon') is None


def test_cli_flag_and_benchmark_table():
    from pokechamp.__main__ import build_parser, train_config_from_args
    from pokechamp.tools import bench_rollout
    cfg = train_config_from_args(build_parser().parse_args(['train', '--preset', 'cloud', '--envs-per-worker', '8']))
    assert cfg.envs_per_worker == 8 and cfg.vectorized and cfg.jobs_per_worker == 2
    assert TrainConfig().envs_per_worker == 48
    with pytest.raises(SystemExit):
        train_config_from_args(build_parser().parse_args(['train', '--envs-per-worker', '0']))
    assert bench_rollout._parse_list('1, 2,4') == [1, 2, 4]
    rows = [{'workers': 2, 'mode': 'old', 'envs': None, 'games_per_s': 10.0, 'decisions_per_s': 250.0,
             'ms_per_decision': 8.0, 'speedup': 1.0},
            {'workers': 2, 'mode': 'vec', 'envs': 48, 'games_per_s': 16.0, 'decisions_per_s': 400.0,
             'ms_per_decision': 5.0, 'speedup': 1.6, 'mean_batch': 20.0, 'nn_share': 0.12}]
    table = bench_rollout.format_table(rows)
    assert 'vec K=48' in table and '1.60x' in table and '20.0' in table
