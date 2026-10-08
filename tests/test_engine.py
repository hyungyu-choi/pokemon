"""Engine tests: replay recorded Pokemon Showdown battles and compare the full protocol logs."""
import gzip
import json
import os
import random

import pytest

from pokechamp.sim.battle import Battle
from pokechamp.sim.prng import PRNG
from pokechamp.sim.teams import calc_stats, export_team, import_team, pack_team, unpack_team
from pokechamp.tools.replay import first_divergence, replay_result

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'showdown_reference.json.gz')


def _battles():
    with gzip.open(FIXTURE, 'rt', encoding='utf-8') as f:
        return json.load(f)['battles']


@pytest.mark.parametrize('index', range(40))
def test_replay_matches_showdown(index):
    ref = _battles()[index]
    battle = replay_result(ref)
    assert first_divergence(ref['log'], battle.log) is None


def test_prng_matches_showdown_sequence():
    # Gen5 LCG seeded like Showdown; values checked against the reference implementation
    prng = PRNG([1, 2, 3, 4])
    seq = [prng.random(100) for _ in range(5)]
    again = PRNG([1, 2, 3, 4])
    assert seq == [again.random(100) for _ in range(5)]
    assert all(0 <= x < 100 for x in seq)


def test_champions_stats():
    team = import_team('''Garchomp @ Choice Scarf
Ability: Rough Skin
SPs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Earthquake
''')
    stats = calc_stats(team[0])
    # Champions: HP = base + SP + 75, others = (base + SP + 20) * nature
    assert stats['hp'] == 108 + 2 + 75
    assert stats['atk'] == 130 + 32 + 20
    assert stats['spe'] == int((102 + 32 + 20) * 1.1)
    assert stats['spa'] == int((80 + 0 + 20) * 0.9)


def test_pack_roundtrip():
    team = import_team('''Rotom-Wash @ Sitrus Berry
Ability: Levitate
SPs: 32 HP / 32 Def / 2 SpA
Bold Nature
- Hydro Pump
- Volt Switch
''')
    packed = pack_team(team)
    again = unpack_team(packed)
    assert again[0].species == 'Rotom-Wash'
    assert again[0].evs['def'] == 32
    assert 'Hydro Pump' in export_team(again)


def test_random_battles_finish():
    from pokechamp.env.runner import gen5_seed, play_battle
    from pokechamp.ai.agents_basic import RandomAgent
    from pokechamp.teambuilder import RandomTeamGenerator
    gen = RandomTeamGenerator(3)
    rng = random.Random(3)
    for fmt in ('gen9championsbssregmc', 'gen9championsvgc2026regmc'):
        for i in range(5):
            r = play_battle(gen.random_team(), gen.random_team(), RandomAgent(i), RandomAgent(i + 1), fmt,
                            seed=gen5_seed(rng))
            assert r.error is None
            assert r.winner in ('p1', 'p2', None)


def test_all_callbacks_ported():
    from pokechamp.sim.dex import get_dex
    assert get_dex().missing == []
