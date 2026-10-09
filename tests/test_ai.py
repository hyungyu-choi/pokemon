"""Tests for the battle environment, validator, cloning, agents and the advisor."""
import json
import os
import random

from pokechamp.env.actions import SLOT_ACTIONS, decode, legal_joint_actions, team_preview_options
from pokechamp.env.runner import BattleSession, gen5_seed, play_battle
from pokechamp.sim.clone import clone_battle
from pokechamp.teambuilder import RandomTeamGenerator, TeamValidator

ROOT = os.path.dirname(os.path.dirname(__file__))


def test_random_teams_are_legal():
    for fmt in ('gen9championsbssregmc', 'gen9championsvgc2026regmc', 'gen9championsou'):
        gen = RandomTeamGenerator(5, formatid=fmt)
        val = TeamValidator(fmt)
        for _ in range(30):
            assert val.validate_team(gen.random_team()) == []


def test_validator_rejects_illegal_sets():
    from pokechamp.sim.teams import import_team
    team = import_team('''Garchomp @ Choice Scarf
Ability: Rough Skin
SPs: 32 Atk / 34 Spe
- Earthquake
- Earthquake
- Spore
''')
    problems = TeamValidator().validate_team(team)
    assert any('Spore' in p for p in problems)
    assert any('more than once' in p for p in problems)
    assert any('34 Stat Points' in p for p in problems)
    assert any('at least 6' in p for p in problems)


def test_team_preview_options():
    assert len(team_preview_options(6, 3, 1)) == 60
    assert len(team_preview_options(6, 4, 2)) == 90


def test_legal_actions_and_masks():
    gen = RandomTeamGenerator(9)
    rng = random.Random(9)
    for fmt in ('gen9championsbssregmc', 'gen9championsvgc2026regmc'):
        s = BattleSession(fmt, gen.random_team(), gen.random_team(), seed=gen5_seed(rng))
        for sid in s.pending():
            assert s.apply(sid, s.legal(sid)[0])
        for sid in s.pending():
            legal = s.legal(sid)
            assert legal
            for combo in legal:
                for a in combo:
                    assert a == -1 or 0 <= a < SLOT_ACTIONS
                    decode(a)
            assert legal_joint_actions(s.request(sid), s.n_active) == legal or s.n_active == 2


def test_clone_is_independent_and_deterministic():
    gen = RandomTeamGenerator(2)
    rng = random.Random(2)
    s = BattleSession('gen9championsbssregmc', gen.random_team(), gen.random_team(), seed=gen5_seed(rng))
    for _ in range(3):
        for sid in s.pending():
            s.apply(sid, s.legal(sid)[0])
    c = clone_battle(s.battle)
    assert c.sides[0].pokemon[0] is not s.battle.sides[0].pokemon[0]
    for _ in range(5):
        if s.battle.ended:
            break
        for side in s.battle.sides:
            if side.activeRequest and not side.activeRequest.get('wait') and not side.isChoiceDone():
                s.battle.choose(side.id, 'default')
                c.choose(side.id, 'default')
    strip = lambda log: [x for x in log if not x.startswith('|t:')]  # noqa: E731
    assert strip(s.battle.log) == strip(c.log)


def test_tracker_follows_battle():
    gen = RandomTeamGenerator(4)
    rng = random.Random(4)
    s = BattleSession('gen9championsbssregmc', gen.random_team(), gen.random_team(), seed=gen5_seed(rng))
    while not s.ended and s.battle.turn < 10:
        for sid in s.pending():
            view = s.view(sid)
            if not s.request(sid).get('teamPreview'):
                me = s.battle.getSide(sid).active[0]
                mine = view.my_side.active()[0]
                assert mine.species == me.species.name
                assert abs(mine.hp - me.hp / me.maxhp) < 1e-6
            if not s.apply(sid, rng.choice(s.legal(sid))):
                s.apply_default(sid)


def test_heuristic_beats_random():
    from pokechamp.ai.agents_basic import RandomAgent
    from pokechamp.ai.evaluate import evaluate
    from pokechamp.ai.heuristic import HeuristicAgent
    r = evaluate(HeuristicAgent, RandomAgent, 20, seed=1)
    assert r.win_rate >= 0.8


def test_network_forward_and_agent():
    import torch
    from pokechamp.ai.model import PolicyValueNet
    from pokechamp.ai.nn_agent import NNAgent
    torch.manual_seed(0)
    model = PolicyValueNet(d=32, layers=1, heads=2)
    gen = RandomTeamGenerator(1)
    r = play_battle(gen.random_team(), gen.random_team(), NNAgent(model, sample=True), NNAgent(model),
                    'gen9championsbssregmc', seed=[1, 2, 3, 4], max_turns=30)
    assert r.error is None


def test_advisor_example():
    from pokechamp.ai.advisor import Advisor, parse_input
    with open(os.path.join(ROOT, 'examples', 'advisor_state.json'), encoding='utf-8') as f:
        inp = parse_input(json.load(f))
    recs = Advisor(seed=0).recommend(inp, determinizations=2, depth=1)
    labels = [r.label for r in recs]
    assert labels
    # Garchomp is choice-locked into Earthquake: no other move may be suggested
    assert all('Dragon Claw' not in x and 'Stone Edge' not in x for x in labels)
    assert all(0.0 <= r.win_rate <= 1.0 for r in recs)


def test_team_evolution_step():
    from pokechamp.teambuilder.evolve import EvolveConfig, TeamEvolver
    cfg = EvolveConfig(population=6, games_per_team=2, hall_of_fame=2, elite=2, random_opponents=1, workers=1,
                       max_turns=40)
    evo = TeamEvolver(cfg, log=lambda *a: None)
    evo.step(None)
    val = TeamValidator()
    for team in evo.population:
        from pokechamp.teambuilder.evolve import Obj_team
        assert val.validate_team(Obj_team(team)) == []
