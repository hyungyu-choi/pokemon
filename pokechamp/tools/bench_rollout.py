"""Benchmark the training actors: one game at a time (``vectorized=False``) vs vectorised workers.

Plays the same PPO rollout rounds the trainer plays (cloud preset: warm start from the shipped model, opponent
mix self / snapshot / heuristic, random / pool / usage teams) on a worker pool, for each worker count, and
prints games/s, decisions/s (both players), worker milliseconds per decision and the mean batch per forward::

    python -m pokechamp.tools.bench_rollout                       # 1, 2 and 4 workers, 512 games per round
    python -m pokechamp.tools.bench_rollout --workers 2 --games 256 --envs 32,48 --repeats 2

Spawning the pool and the first job of each worker (imports, team sampler, model) are excluded by a warm-up
round.  Rounds of the two modes alternate, so slow drifts of the machine affect both alike.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time


def _parse_list(text: str) -> list:
    return [int(x) for x in str(text).split(',') if x.strip()]


def bench(workers: list, games: int, envs: list, repeats: int = 1, model: str = 'models/battle_singles.pt',
          snapshots: int = 3, out: str | None = None, log=print) -> list:
    """Returns one dict per (workers, mode) with the measured throughput."""
    import torch
    from ..ai.train import Trainer, build_config, rollout_throughput
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    rows = []
    base = out or tempfile.mkdtemp(prefix='bench_rollout_')
    try:
        for W in workers:
            run_dir = os.path.join(base, f'w{W}')
            cfg = build_config('cloud', overrides={
                'out': run_dir, 'workers': W, 'device': 'cpu', 'init': model, 'eval_games': 0, 'incumbent': None,
                'log_file': None, 'resume': 'never', 'games_per_iter': games, 'envs_per_worker': envs[0]})
            t = Trainer(cfg, log=lambda m: None)
            t.setup()
            for i in range(1, snapshots + 1):
                t._add_snapshot(i)
            t._start_pool()
            try:
                modes = [('old', None)] + [('vec', k) for k in envs]
                # warm-up: spawn the workers, import, build team samplers and caches
                for _name, k in modes:
                    cfg.vectorized = k is not None
                    if k:
                        cfg.envs_per_worker = k
                    t.rollout(2 * W, 12345, t._snapshot_weights(), 'ppo')
                totals = {m: {'games': 0, 'decisions': 0, 'seconds': 0.0, 'nn_decisions': 0, 'forwards': 0,
                              'job_seconds': 0.0, 'nn_seconds': 0.0} for m in modes}
                for r in range(repeats):
                    for mode in modes:
                        cfg.vectorized = mode[1] is not None
                        if mode[1]:
                            cfg.envs_per_worker = mode[1]
                        t0 = time.perf_counter()
                        trajs, stats = t.rollout(games, 1000 + r, t._snapshot_weights(), 'ppo')
                        dt = time.perf_counter() - t0
                        del trajs
                        tot = totals[mode]
                        tot['seconds'] += dt
                        for k in ('games', 'decisions', 'nn_decisions', 'forwards', 'job_seconds', 'nn_seconds'):
                            tot[k] += stats.get(k, 0)
                        tp = rollout_throughput(stats, dt, W)
                        log(f"  workers {W} {mode[0]}{'' if mode[1] is None else f' K={mode[1]}'} round {r + 1}: "
                            f"{tp['rollout_games_per_s']:.2f} games/s, {tp.get('decisions_per_s', 0):.0f} dec/s"
                            + (f", batch {tp['mean_batch']:.1f}" if 'mean_batch' in tp else ''))
                old_gps = totals[modes[0]]['games'] / max(1e-9, totals[modes[0]]['seconds'])
                for mode in modes:
                    tot = totals[mode]
                    gps = tot['games'] / max(1e-9, tot['seconds'])
                    row = {'workers': W, 'mode': mode[0], 'envs': mode[1], 'games': tot['games'],
                           'seconds': round(tot['seconds'], 2), 'games_per_s': round(gps, 3),
                           'decisions_per_s': round(tot['decisions'] / max(1e-9, tot['seconds']), 1),
                           'ms_per_decision': round(1000 * W * tot['seconds'] / max(1, tot['decisions']), 3),
                           'speedup': round(gps / max(1e-9, old_gps), 3)}
                    if tot['forwards']:
                        row['mean_batch'] = round(tot['nn_decisions'] / tot['forwards'], 2)
                        row['nn_share'] = round(tot['nn_seconds'] / max(1e-9, tot['job_seconds']), 3)
                    rows.append(row)
            finally:
                t._shutdown_pool()
    finally:
        if out is None:
            shutil.rmtree(base, ignore_errors=True)
    return rows


def format_table(rows: list) -> str:
    head = f"{'workers':>7} {'mode':>9} {'games/s':>8} {'dec/s':>7} {'ms/dec':>7} {'batch':>6} {'nn%':>5} {'speedup':>7}"
    lines = [head, '-' * len(head)]
    for r in rows:
        mode = r['mode'] if r['envs'] is None else f"vec K={r['envs']}"
        lines.append(f"{r['workers']:>7} {mode:>9} {r['games_per_s']:>8.2f} {r['decisions_per_s']:>7.0f} "
                     f"{r['ms_per_decision']:>7.2f} {r.get('mean_batch', 1.0):>6.1f} "
                     f"{100 * r.get('nn_share', 0):>5.0f} {r['speedup']:>6.2f}x")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m pokechamp.tools.bench_rollout', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', default='1,2,4', help='comma-separated worker counts (default 1,2,4)')
    ap.add_argument('--games', type=int, default=512, help='games per measured round (default 512, as the cloud preset)')
    ap.add_argument('--envs', default='48', help='games per worker at once for the vectorised mode, e.g. 16,32')
    ap.add_argument('--repeats', type=int, default=1, help='measured rounds per mode (default 1)')
    ap.add_argument('--model', default='models/battle_singles.pt', help='warm-start model')
    ap.add_argument('--snapshots', type=int, default=3, help='snapshot opponents in the pool (default 3)')
    ap.add_argument('--json', help='also write the results to this JSON file')
    args = ap.parse_args(argv)
    rows = bench(_parse_list(args.workers), args.games, _parse_list(args.envs), args.repeats, args.model,
                 args.snapshots, log=lambda m: print(m, flush=True))
    print(format_table(rows))
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
