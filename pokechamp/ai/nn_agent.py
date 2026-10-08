"""Agent driven by the policy/value network."""
from __future__ import annotations

import numpy as np
import torch

from ..env.runner import AgentChoice
from ..env.view import BattleView
from .features import encode
from .model import PolicyValueNet, collate_obs, joint_logits, pad_options, preview_logits


def potential(view: BattleView) -> float:
    """Material balance used for reward shaping: (our HP - their HP) / Pokemon brought, in [-1, 1]."""
    def side_hp(side):
        hp = sum(p.hp for p in side.pokemon if p.revealed and not p.fainted)
        revealed = sum(1 for p in side.pokemon if p.revealed)
        hp += max(0, side.team_size - revealed)  # unseen Pokemon are at full HP
        return hp / max(1, side.team_size)
    return side_hp(view.my_side) - side_hp(view.foe_side)


class NNAgent:
    """Chooses actions with a :class:`PolicyValueNet`.

    ``sample=True`` samples from the policy (training / exploration); otherwise it picks the
    most likely legal option.  With ``record=True`` the observation and the data PPO needs
    (log-probability, value, shaping potential) are attached to every decision.
    """

    name = 'nn'

    def __init__(self, model: PolicyValueNet, sample: bool = False, temperature: float = 1.0,
                 record: bool = False):
        self.model = model
        self.sample = sample
        self.temperature = temperature
        self.record = record
        self.rng = np.random.default_rng()

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def evaluate(self, view: BattleView, request: dict, legal: list):
        """-> (probabilities over ``legal``, value estimate, obs)."""
        obs = encode(view, request)
        batch = collate_obs([obs])
        slot_logits, value, x = self.model(batch)
        if request.get('teamPreview'):
            pick, lead = self.model.preview_scores(x)
            options, mask = pad_options([legal], 4)
            n_lead = torch.tensor([view.n_active])
            logits = preview_logits(pick, lead, options, n_lead, mask)[0]
        else:
            combos, mask = pad_options([legal], 2)
            logits = joint_logits(slot_logits, combos, mask)[0]
        logits = logits[:len(legal)] / max(1e-3, self.temperature)
        probs = torch.softmax(logits, dim=-1).numpy()
        return probs, float(value[0]), obs

    def choose(self, session, sid, view, legal):
        request = session.request(sid)
        probs, value, obs = self.evaluate(view, request, legal)
        if self.sample:
            idx = int(self.rng.choice(len(legal), p=probs / probs.sum()))
        else:
            idx = int(np.argmax(probs))
        if not self.record:
            return legal[idx]
        info = {'idx': idx, 'logp': float(np.log(max(probs[idx], 1e-12))), 'value': value,
                'phi': potential(view), 'kind': 1 if request.get('teamPreview') else 0,
                'n_lead': view.n_active}
        return AgentChoice(legal[idx], obs=obs, info=info)
