"""Command line interface: ``python -m pokechamp <command> ...``

Commands
  battle     play battles between agents and report the results (optionally save logs)
  train      train the battle AI (behaviour cloning + PPO self-play; --preset cloud for long GPU runs)
  evolve     evolve teams starting from random teams
  coevolve   alternate team evolution and policy training (teams and battle AI improve together)
  advise     recommend the best action for a live battle situation (JSON state file)
  ui         battle assistant web UI: recommended team, team preview, per-turn advice by menu selection
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
    with open(path, encoding='utf-8-sig') as f:  # -sig: also files saved with a BOM (Windows Notepad)
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


def _workers_arg(text: str):
    """--workers: 'auto' (= number of CPU threads) or a positive number."""
    if str(text).lower() == 'auto':
        return 'auto'
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'auto' or a number, got {text!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError('need at least 1 worker')
    return n


# train flag -> TrainConfig field (flags left out on the command line keep the preset / --config value)
TRAIN_FLAGS = {'format': 'formatid', 'out': 'out', 'init': 'init', 'workers': 'workers', 'bc_games': 'bc_games',
               'bc_epochs': 'bc_epochs', 'iters': 'iters', 'games_per_iter': 'games_per_iter', 'lr': 'lr',
               'eval_every': 'eval_every', 'eval_games': 'eval_games', 'team_pool': 'team_pool', 'width': 'd',
               'layers': 'layers', 'seed': 'seed', 'resume': 'resume', 'time_budget_h': 'time_budget_h',
               'sync_dir': 'sync_dir', 'device': 'device', 'envs_per_worker': 'envs_per_worker'}


def train_config_from_args(args):
    """TrainConfig from parsed ``train`` arguments: defaults < --preset < --config < explicit flags."""
    from .ai.train import build_config
    overrides = {}
    for flag, name in TRAIN_FLAGS.items():
        value = getattr(args, flag, None)
        if value is not None:
            overrides[name] = value
    if isinstance(overrides.get('init'), str) and overrides['init'].lower() in ('none', ''):
        overrides['init'] = None
    try:
        return build_config(args.preset, args.config, overrides)
    except (OSError, ValueError) as exc:
        raise SystemExit(f'train: {exc}') from None


def cmd_train(args):
    from .ai.train import train
    train(train_config_from_args(args))


def cmd_evolve(args):
    from .teambuilder.evolve import EvolveConfig, evolve
    cfg = EvolveConfig(formatid=args.format, out=args.out, population=args.population,
                       generations=args.generations, games_per_team=args.games_per_team, workers=args.workers,
                       agent=args.agent, seed=args.seed)
    evolve(cfg, resume=args.resume)
    print(f"population, best team and usage statistics written to {args.out}/")


def cmd_coevolve(args):
    """Alternate team evolution (with the current policy) and policy training (on the evolved teams)."""
    from .ai.train import TrainConfig, train
    from .teambuilder.evolve import EvolveConfig, evolve
    os.makedirs(args.out, exist_ok=True)
    model = args.init
    teams = None
    for r in range(1, args.rounds + 1):
        tdir = os.path.join(args.out, f'round{r}_teams')
        evolve(EvolveConfig(formatid=args.format, out=tdir, population=args.population,
                            generations=args.generations, workers=args.workers, agent=model or 'heuristic',
                            seed=args.seed + r), resume=teams)
        teams = os.path.join(tdir, 'population.json')
        mdir = os.path.join(args.out, f'round{r}_policy')
        model = train(TrainConfig(formatid=args.format, out=mdir, workers=args.workers, iters=args.iters,
                                  bc_games=0 if model else args.bc_games, team_pool=teams, team_pool_prob=0.6,
                                  seed=args.seed + r), init=model)
        print(f"[coevolve] round {r}: teams -> {teams}, policy -> {model}")


def cmd_advise(args):
    from .ai.advisor import Advisor, format_recommendations, parse_input
    with open(args.state, encoding='utf-8-sig') as f:
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


def cmd_ui(args):
    from .ui import console
    console.prepare_console()
    print('배틀 도우미를 시작하는 중입니다...', flush=True)
    from .ui.server import STATIC_DIR, serve
    from .ui.service import AssistantService
    if not os.path.isfile(os.path.join(STATIC_DIR, 'index.html')):
        print(f'\n[오류] 화면 파일이 없습니다: {os.path.join(STATIC_DIR, "index.html")}\n'
              '  다운로드한 폴더(ZIP을 푼 폴더 전체)를 그대로 두고 그 폴더에서 run_ui.bat / run_ui.sh 를 실행하세요.\n',
              file=sys.stderr, flush=True)
        raise SystemExit(1)
    for flag, path in (('--model', args.model), ('--library', args.library)):
        if path and not os.path.isfile(path):
            print(f'\n[오류] {flag} 로 지정한 파일을 찾을 수 없습니다: {path}\n', file=sys.stderr, flush=True)
            raise SystemExit(1)
    service = AssistantService(model_path='' if args.no_model else args.model, library_path=args.library)
    # numpy is needed by every AI calculation, torch by the neural network: check before starting, so the
    # problem is explained here instead of failing later in the browser
    missing = console.missing_dependencies(need_torch=bool(service.model_path))
    if missing:
        print(console.dependency_message(missing), file=sys.stderr, flush=True)
        raise SystemExit(1)
    notes = []
    if args.no_model:
        notes.append('신경망 없이 (--no-model) 휴리스틱 AI로 계산합니다.')
    elif not service.model_path:
        notes += ['[주의] AI 모델 파일(models/battle_singles.pt)을 찾지 못해 더 약한 휴리스틱 AI로 계산합니다.',
                  '   다운로드한 폴더에서 실행하거나 --model 로 파일을 지정하세요.']
    if not service.library_path:
        notes += ['[주의] 팀 라이브러리(models/teams_singles.json)를 찾지 못해 추천 파티가 비어 있습니다.',
                  '   다운로드한 폴더에서 실행하거나 --library 로 파일을 지정하세요.']
    serve(args.host, args.port, service, open_browser=args.open, notes=notes)


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


TRAIN_HELP = """Train the battle AI: behaviour cloning + PPO self-play (default preset) or a long resumable run
that continues from the shipped model (--preset cloud).

Settings are combined as: built-in defaults < --preset < --config JSON < the flags given here.
Ctrl-C (or SIGTERM) finishes the current iteration, saves and exits; a second Ctrl-C exits at once.
Running the same command again continues where it stopped (--resume auto).

Examples
  python -m pokechamp train --preset cloud --out runs/cloud --time-budget-h 11.5
  python -m pokechamp train --preset cloud --out runs/cloud --sync-dir /content/drive/MyDrive/pokechamp_run
  python -m pokechamp train --out runs/fresh --bc-games 3000 --iters 200 --workers 4
"""


def build_parser():
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

    p = sub.add_parser('train', help='train the battle AI', formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=TRAIN_HELP)
    p.add_argument('--preset', choices=('default', 'cloud'), default='default',
                   help='default: fresh model (behaviour cloning, then PPO); cloud: long GPU/cloud run continuing '
                        'from models/battle_singles.pt (no BC, 512 games per iteration, runs until the time budget)')
    p.add_argument('--config', help='TrainConfig fields as a JSON file or inline JSON object, '
                                    'e.g. \'{"games_per_iter": 256, "lr": 2e-4}\'')
    p.add_argument('--format', help=fmt_help)
    p.add_argument('--out', help='output folder (default runs/battle)')
    p.add_argument('--init', help='warm start from this checkpoint, skipping behaviour cloning '
                                  '(cloud preset: models/battle_singles.pt; "none" = fresh model)')
    p.add_argument('--resume', choices=('auto', 'never'),
                   help='auto (default): continue from OUT/state.pt, or from SYNC_DIR/state.pt when OUT has none')
    p.add_argument('--time-budget-h', type=float, help='stop in time (final evaluation + save) after this many hours')
    p.add_argument('--sync-dir', help='mirror checkpoints and logs here after every checkpoint (e.g. Google Drive)')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), help='learner device (auto: GPU if available)')
    p.add_argument('--workers', type=_workers_arg, help="rollout processes: 'auto' (= CPU threads) or a number")
    p.add_argument('--envs-per-worker', type=int,
                   help='games each worker plays at once, sharing batched network calls (default 48)')
    p.add_argument('--bc-games', type=int)
    p.add_argument('--bc-epochs', type=int)
    p.add_argument('--iters', type=int)
    p.add_argument('--games-per-iter', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--eval-every', type=int)
    p.add_argument('--eval-games', type=int, help='games per evaluation opponent (heuristic and incumbent)')
    p.add_argument('--team-pool', help='JSON of teams to train with (e.g. evolved population.json)')
    p.add_argument('--width', type=int)
    p.add_argument('--layers', type=int)
    p.add_argument('--seed', type=int)
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

    p = sub.add_parser('coevolve', help='alternate team evolution and policy training')
    p.add_argument('--format', default='gen9championsbssregmc', help=fmt_help)
    p.add_argument('--out', default='runs/coevolve')
    p.add_argument('--init', help='starting policy checkpoint (default: behaviour cloning first)')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--generations', type=int, default=30)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--bc-games', type=int, default=3000)
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--seed', type=int, default=0)
    p.set_defaults(func=cmd_coevolve)

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

    p = sub.add_parser('ui', help='battle assistant web UI (singles): team, team preview and turn advice')
    p.add_argument('--host', default='127.0.0.1', help='use 0.0.0.0 to open it from a phone on the same Wi-Fi')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--model', help='policy/value checkpoint (default: models/battle_singles.pt)')
    p.add_argument('--library', help='evolved team library (default: models/teams_singles.json)')
    p.add_argument('--no-model', action='store_true',
                   help='do not use the neural network (weaker heuristic AI; PyTorch is not needed)')
    p.add_argument('--open', action='store_true', help='open the browser')
    p.set_defaults(func=cmd_ui)

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
    return ap


def main(argv=None):
    ap = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == 'difftest':
        from .tools.difftest import main as difftest_main
        return difftest_main(argv[1:])
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    if not __package__:
        # started as a file ("python pokechamp/__main__.py ui"): import the package so its relative imports work
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from pokechamp.__main__ import main as _package_main
        _package_main()
    else:
        main()
