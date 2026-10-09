"""Agent driven by the policy/value network."""
from __future__ import annotations

import numpy as np
import torch

from ..env.runner import AgentChoice
from ..env.view import BattleView
from .features import encode
from .model import PolicyValueNet, collate_obs, joint_logits, pad_options, preview_logits
from .shaping import potential  # noqa: F401  (re-exported: training imports it from here)


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


@torch.no_grad()
def batch_policy(model: PolicyValueNet, items: list):
    """Policy probabilities and values for many move decisions at once.

    ``items``: list of ``(view, request, legal)`` (no team-preview decisions).  Returns
    ``(probs_list, values)`` where ``probs_list[i]`` is a numpy array over ``legal``.
    """
    if not items:
        return [], []
    obs = [encode(view, request) for view, request, _legal in items]
    slot_logits, values, _x = model(collate_obs(obs))
    combos, mask = pad_options([legal for _v, _r, legal in items], 2)
    logits = joint_logits(slot_logits, combos, mask)
    probs = torch.softmax(logits, dim=-1).numpy()
    out = [probs[i, :len(items[i][2])] for i in range(len(items))]
    return out, values.tolist()
