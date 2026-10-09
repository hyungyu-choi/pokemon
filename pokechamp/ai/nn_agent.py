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


_OBS_DTYPES = {'ids': np.int64, 'mon': np.float32, 'move_ids': np.int64, 'move': np.float32,
               'glob': np.float32, 'active_idx': np.int64}


def collate_any(obs_list: list, device='cpu') -> dict:
    """Like :func:`model.collate_obs`, but also accepts compact observations (float16 / small ints, see
    :mod:`pokechamp.ai.vec_rollout`): every array is converted to the dtype the network expects."""
    out = {}
    for k, dtype in _OBS_DTYPES.items():
        arr = np.stack([o[k] for o in obs_list])
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
        out[k] = torch.from_numpy(arr).to(device)
    return out


@torch.no_grad()
def batch_evaluate(model: PolicyValueNet, obs_list: list, legal_list: list, preview_n_lead: list,
                   temperatures=None, critic: bool = False):
    """One forward pass for many decisions, move decisions and team previews mixed.

    ``obs_list[i]``: ``features.encode`` output (or its compact form); ``legal_list[i]``: the options;
    ``preview_n_lead[i]``: ``None`` for a move decision, else the number of leads (``view.n_active``) of a
    team-preview decision; ``temperatures[i]`` (default 1) divides that row's logits, as in
    :meth:`NNAgent.evaluate`.  Returns ``(probs, logps, values, critics)``: per-decision numpy arrays over the
    legal options (softmax probabilities and log-softmax), the value head ``[B]`` (``2 * P(win) - 1``) and the
    critic ``[B]`` (or ``None`` when the model has none or ``critic`` is False).  All tensors live on the
    model's device.
    """
    n = len(obs_list)
    if n == 0:
        return [], [], np.zeros(0, np.float32), None
    dev = next(model.parameters()).device
    slot_logits, value, x = model(collate_any(obs_list, dev))
    crit = model.critic(x).float().cpu().numpy() if critic and model.has_critic else None
    temps = np.ones(n, dtype=np.float32) if temperatures is None else \
        np.maximum(1e-3, np.asarray(temperatures, dtype=np.float32))
    probs, logps = [None] * n, [None] * n
    mv = [i for i in range(n) if preview_n_lead[i] is None]
    pv = [i for i in range(n) if preview_n_lead[i] is not None]
    for rows, width in ((mv, 2), (pv, 4)):
        if not rows:
            continue
        sel = torch.tensor(rows, dtype=torch.long, device=dev)
        options, mask = pad_options([legal_list[i] for i in rows], width)
        options, mask = options.to(dev), mask.to(dev)
        if width == 2:
            logits = joint_logits(slot_logits[sel], options, mask)
        else:
            pick, lead = model.preview_scores(x[sel])
            n_lead = torch.tensor([int(preview_n_lead[i]) for i in rows], dtype=torch.long, device=dev)
            logits = preview_logits(pick, lead, options, n_lead, mask)
        t = torch.from_numpy(temps[rows]).to(dev).unsqueeze(1)
        logits = logits / t
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
        for j, i in enumerate(rows):
            k = len(legal_list[i])
            probs[i], logps[i] = p[j, :k], lp[j, :k]
    return probs, logps, value.float().cpu().numpy(), crit


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
