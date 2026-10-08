"""Command line interface: ``python -m pokechamp <command> ...``

Commands
  battle     play battles between agents and report the results (optionally save logs)
  train      train the battle AI (behaviour cloning + PPO self-play)
  evolve     evolve teams starting from random teams
  advise     recommend the best action for a live battle situation (JSON state file)
  validate   check a team (Showdown text format with SPs) against the Champions rules
  randomteam print a random legal team
  difftest   compare the engine against Pokemon Showdown (needs Node.js + a built Showdown)
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _agent_factory(spec: str):
    """'random' | 'maxdamage' | 'heuristic' | path/to/model.pt -> picklable factory(seed)."""
    if spec == 'random':
        from .ai.agents_basic import RandomAgent
        return RandomAgent
    if spec == 'maxdamage':
        from .ai.heuristic import MaxDamageAgent
        return MaxDamageAgent
    if spec == 'heuristic':
        from .ai.heuristic import HeuristicAgent
        return HeuristicAgent
    if spec.endswith('.pt') and os.path.exists(spec):
        from .ai.train import _NNFactory
        return _NNFactory(spec)
    raise SystemExit(f"unknown agent {spec!r} (random, maxdamage, heuristic or a model .pt file)")


def _load_team(path: str):
    from .sim.teams import import_team
    with open(path, encoding='utf-8') as f:
        text = f.read()
    if text.lstrip().startswith('[') or text.lstrip().startswith('{'):
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get('teams', [data])[0]
        return data
    return import_team(text)


def cmd_battle(args):
    from .ai.evaluate import evaluate
    from .env.runner import gen5_seed, play_battle
    from .teambuilder import RandomTeamGenerator
    fa, fb = _agent_factory(args.p1), _agent_factory(args.p2)
    if args.team1 or args.team2 or args.log_dir:
        import random
        rng = random.Random(args.seed)
        gen = RandomTeamGenerator(args.seed, formatid=args.format)
        t1 = _load_team(args.team1) if args.team1 else None
        t2 = _load_team(args.team2) if args.team2 else None
        wins = {'p1': 0, 'p2': 0, None: 0}
        if args.log_dir:
            os.makedirs(args.log_dir, exist_ok=True)
        for i in range(args.n):
            a, b = fa(rng.randrange(1 << 30)), fb(rng.randrange(1 << 30))
            r = play_battle(t1 or gen.random_team(), t2 or gen.random_team(), a, b, args.format,
                            seed=gen5_seed(rng), keep_log=bool(args.log_dir))
            wins[r.winner] += 1
            if args.log_dir:
                with open(os.path.join(args.log_dir, f'battle_{i:04d}.log'), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(line for line in r.log if not line.startswith('|t:')) + '\n')
            if r.error:
                print(f"battle {i}: {r.error}", file=sys.stderr)
        print(f"{args.p1} vs {args.p2}: p1 won {wins['p1']}, p2 won {wins['p2']}, unfinished {wins[None]}")
        return
    r = evaluate(fa, fb, args.n, formatid=args.format, seed=args.seed, workers=args.workers)
    print(f"{args.p1} vs {args.p2} ({args.format}): {r}")


def cmd_train(args):
    from .ai.train import TrainConfig, train
    cfg = TrainConfig(formatid=args.format, out=args.out, workers=args.workers, bc_games=args.bc_games,
                      bc_epochs=args.bc_epochs, iters=args.iters, games_per_iter=args.games_per_iter,
                      lr=args.lr, eval_every=args.eval_every, eval_games=args.eval_games, team_pool=args.team_pool,
                      d=args.width, layers=args.layers, seed=args.seed)
    train(cfg, init=args.init)


def cmd_evolve(args):
    from .teambuilder.evolve import EvolveConfig, evolve
    cfg = EvolveConfig(formatid=args.format, out=args.out, population=args.population,
                       generations=args.generations, games_per_team=args.games_per_team, workers=args.workers,
                       agent=args.agent, seed=args.seed)
    evolve(cfg, resume=args.resume)
    print(f"population, best team and usage statistics written to {args.out}/")


def cmd_advise(args):
    from .ai.advisor import Advisor, format_recommendations, parse_input
    with open(args.state, encoding='utf-8') as f:
        inp = parse_input(json.load(f))
    model = args.model
    if model is None and os.path.exists(os.path.join('models', 'battle_singles.pt')) and inp.formatid.endswith('bssregmc'):
        model = os.path.join('models', 'battle_singles.pt')
    library = args.library
    if library is None and os.path.exists(os.path.join('models', 'teams_singles.json')):
        library = os.path.join('models', 'teams_singles.json')
    adv = Advisor(model_path=model, library=library, seed=args.seed)
    recs = adv.recommend(inp, determinizations=args.samples, depth=args.depth,
                         log=(print if args.verbose else None))
    print(format_recommendations(recs, args.top))


def cmd_validate(args):
    from .teambuilder import TeamValidator
    team = _load_team(args.team)
    problems = TeamValidator(args.format).validate_team(team)
    if problems:
        print('\n'.join(problems))
        raise SystemExit(1)
    print('team is legal')


def cmd_randomteam(args):
    from .sim.teams import export_team
    from .teambuilder import RandomTeamGenerator
    print(export_team(RandomTeamGenerator(args.seed, formatid=args.format).random_team()), end='')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m pokechamp', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    fmt_help = 'gen9championsbssregmc (singles), gen9championsvgc2026regmc (doubles) or gen9championsou'

    p = sub.add_parser('battle', help='play battles between agents')
    p.add_argument('--p1', default='heuristic', help='random | maxdamage | heuristic | model.pt')
    p.add_argument('--p2', default='random')
    p.add_argument('-n', type=int, default=100)
    p.add_argument('--format', default='gen9championsbssregmc', help=fmt_help)
    p.add_argument('--team1', help='team file for p1 (default: random team every battle)')
    p.add_argument('--team2', help='team file for p2')
    p.add_argument('--log-dir', help='save every battle log (Showdown protocol) here')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--seed', type=int, default=0)
    p.set_defaults(func=cmd_battle)

    p = sub.add_parser('train', help='train the battle AI')
    p.add_argument('--format', default='gen9championsbssregmc', help=fmt_help)
    p.add_argument('--out', default='runs/battle')
    p.add_argument('--init', help='continue from this checkpoint (skips behaviour cloning)')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--bc-games', type=int, default=3000)
    p.add_argument('--bc-epochs', type=int, default=4)
    p.add_argument('--iters', type=int, default=150)
    p.add_argument('--games-per-iter', type=int, default=96)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--eval-games', type=int, default=200)
    p.add_argument('--team-pool', help='JSON of teams to train with (e.g. evolved population.json)')
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser('evolve', help='evolve teams from random teams')
    p.add_argument('--format', default='gen9championsbssregmc', help=fmt_help)
    p.add_argument('--out', default='runs/teams')
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--generations', type=int, default=50)
    p.add_argument('--games-per-team', type=int, default=16)
    p.add_argument('--agent', default='heuristic', help='battle agent used for fitness: heuristic or model.pt')
    p.add_argument('--resume', help='population.json to continue from')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--seed', type=int, default=0)
    p.set_defaults(func=cmd_evolve)

    p = sub.add_parser('advise', help='rank actions for a live battle situation')
    p.add_argument('state', help='JSON file describing the situation (see examples/advisor_state.json)')
    p.add_argument('--model', help='policy/value checkpoint (default: models/battle_singles.pt if present)')
    p.add_argument('--library', help='set library for the opponent prior (default: models/teams_singles.json)')
    p.add_argument('--samples', type=int, default=16, help='number of sampled opponent sets (determinizations)')
    p.add_argument('--depth', type=int, default=3, help='turns simulated after the decision')
    p.add_argument('--top', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('-v', '--verbose', action='store_true')
    p.set_defaults(func=cmd_advise)

    p = sub.add_parser('validate', help='validate a team file')
    p.add_argument('team')
    p.add_argument('--format', default='gen9championsbssregmc', help=fmt_help)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser('randomteam', help='print a random legal team')
    p.add_argument('--format', default='gen9championsbssregmc', help=fmt_help)
    p.add_argument('--seed', type=int)
    p.set_defaults(func=cmd_randomteam)

    p = sub.add_parser('difftest', help='compare the engine with Pokemon Showdown', add_help=False)
    p.set_defaults(func=None)

    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == 'difftest':
        from .tools.difftest import main as difftest_main
        return difftest_main(argv[1:])
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    main()
