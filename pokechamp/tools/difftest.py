"""Differential testing: Python engine vs. the reference Pokemon Showdown simulator.

For each random battle (random legal teams, random seeds, random choices):
  1. the reference simulator (Node.js, ``tools/showdown_runner.js``) plays it and
     records its input log and battle log;
  2. the Python engine replays the same input log;
  3. both logs must be identical line by line (timestamps excluded).

Usage:
    python -m pokechamp.tools.difftest --showdown /path/to/pokemon-showdown -n 200
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback

from ..teambuilder import RandomTeamGenerator
from .replay import diff_report, first_divergence, replay_result

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(ROOT, 'tools', 'showdown_runner.js')
FORMATS = ['gen9championsbssregmc', 'gen9championsvgc2026regmc']


def make_jobs(n: int, seed: int, formats=None, switch_chance=0.15, mega_chance=0.5):
    rng = random.Random(seed)
    gen = RandomTeamGenerator(seed)
    formats = formats or FORMATS
    jobs = []
    for i in range(n):
        s = [rng.randrange(65536) for _ in range(4)]
        c = [rng.randrange(65536) for _ in range(4)]
        jobs.append({
            'formatid': formats[i % len(formats)],
            'seed': ','.join(map(str, s)),
            'chooserSeed': ','.join(map(str, c)),
            'teams': [gen.random_team(), gen.random_team()],
            'switchChance': switch_chance,
            'megaChance': mega_chance,
            'maxTurns': 150,
        })
    return jobs


def run_reference(jobs, showdown_path: str):
    data = '\n'.join(json.dumps(j) for j in jobs) + '\n'
    proc = subprocess.run(['node', RUNNER, showdown_path], input=data, capture_output=True, text=True, check=True)
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def compare(results, jobs, verbose=3, stop_on_first=False):
    ok = 0
    failures = []
    for job, res in zip(jobs, results):
        if res.get('error'):
            failures.append((job, res, 'reference error: ' + res['error'][:500]))
            continue
        try:
            battle = replay_result(res)
            d = first_divergence(res['log'], battle.log)
            if d is None:
                ok += 1
                continue
            failures.append((job, res, diff_report(res['log'], battle.log)))
        except Exception:
            failures.append((job, res, traceback.format_exc()))
        if stop_on_first:
            break
    return ok, failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--showdown', required=True, help='path to a built pokemon-showdown checkout')
    ap.add_argument('-n', type=int, default=50, help='number of battles')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--format', action='append', help='format id (default: BSS + VGC Reg M-C)')
    ap.add_argument('--show', type=int, default=3, help='number of failures to print')
    ap.add_argument('--save-failures', help='write failing jobs (jsonl) to this file')
    args = ap.parse_args(argv)

    t0 = time.time()
    jobs = make_jobs(args.n, args.seed, args.format)
    results = run_reference(jobs, args.showdown)
    t1 = time.time()
    ok, failures = compare(results, jobs)
    t2 = time.time()
    for job, res, report in failures[:args.show]:
        print('=' * 100)
        print(f"format={job['formatid']} seed={job['seed']}")
        print(report)
    if args.save_failures and failures:
        with open(args.save_failures, 'w') as f:
            for job, res, report in failures:
                f.write(json.dumps(job) + '\n')
    total_turns = sum(r.get('turns', 0) for r in results)
    print(f"identical: {ok}/{len(jobs)}  diverged: {len(failures)}  "
          f"(reference {t1 - t0:.1f}s, python {t2 - t1:.1f}s, {total_turns} turns)")
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
