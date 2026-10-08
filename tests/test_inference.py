"""Tests for the inference of hidden opponent information."""
import random

import numpy as np

from pokechamp.ai.inference import StatBelief, _stats_for
from pokechamp.env.runner import BattleSession, gen5_seed
from pokechamp.teambuilder import RandomTeamGenerator


def test_speed_observations_concentrate_belief():
    species = 'Garchomp'
    true_speed = int((102 + 32 + 20) * 1.1)  # Jolly, 32 Speed SP
    belief = StatBelief()
    belief.set_item('Life Orb')               # known non-Scarf item
    prior_mean = belief.mean_stats(species)['spe']
    for mine in (120, 140, 150, 160, 165, 170, 180):
        belief.observe_speed(species, 1.0, mine, foe_first=true_speed > mine, trick_room=False)
    post = belief.mean_stats(species)['spe']
    assert abs(post - true_speed) < abs(prior_mean - true_speed)
    assert belief.p_faster(species, 1.0, 165, False) > 0.8
    assert belief.p_faster(species, 1.0, 175, False) < 0.2


def test_scarf_is_inferred_from_speed():
    species = 'Garchomp'
    belief = StatBelief()
    # outspeeds something far faster than any non-Scarf Garchomp could be
    max_plain = _stats_for(species)[:, 5].max()
    belief.observe_speed(species, 1.0, max_plain + 20, foe_first=True, trick_room=False)
    assert belief.p_scarf() > 0.7


def test_tracker_builds_beliefs_in_battle():
    gen = RandomTeamGenerator(6)
    rng = random.Random(6)
    from pokechamp.ai.heuristic import HeuristicAgent
    s = BattleSession('gen9championsbssregmc', gen.random_team(), gen.random_team(), seed=gen5_seed(rng))
    agent = HeuristicAgent(0)
    while not s.ended and s.battle.turn < 40:
        for sid in s.pending():
            v = s.view(sid)
            if not s.apply(sid, agent.choose(s, sid, v, s.legal(sid))):
                s.apply_default(sid)
    view = s.view('p1')
    beliefs = [m.belief for m in view.foe_side.pokemon if m.revealed]
    assert beliefs and all(b is not None for b in beliefs)
    assert sum(b.observations for b in beliefs) > 0
    for b in beliefs:
        assert np.isclose(b.w.sum(), 1.0)
