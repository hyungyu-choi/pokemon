"""Policy / value network (PyTorch).

Each of the 12 Pokemon (6 ours, 6 opponent's) becomes a token built from learned
species / item / ability / move embeddings plus numeric features; a global field
token is added and a small Transformer mixes them.  Heads:

* per active slot: 38 action logits (4 moves x 4 targets x Mega on/off, 6 switches),
  combined over the legal joint actions (doubles: two slots),
* team preview: per-Pokemon "bring" and "lead" scores summed over each option,
* value: probability-like estimate of winning (tanh, -1..1).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..env.actions import N_MOVES, SLOT_ACTIONS
from .features import GLOB_F, MON_F, MOVE_F, N_MONS, vocab, vocab_sizes

PASS_COL = SLOT_ACTIONS  # extra logit column (always 0) used for 'pass'


def mlp(i, h, o, n=2):
    layers = []
    d = i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class PolicyValueNet(nn.Module):
    def __init__(self, d: int = 128, layers: int = 2, heads: int = 4, sizes: dict | None = None):
        super().__init__()
        sizes = sizes or vocab_sizes()
        self.config = {'d': d, 'layers': layers, 'heads': heads, 'sizes': dict(sizes)}
        self.species_emb = nn.Embedding(sizes['species'], 48)
        self.item_emb = nn.Embedding(sizes['items'], 16)
        self.ability_emb = nn.Embedding(sizes['abilities'], 16)
        self.move_emb = nn.Embedding(sizes['moves'], 32)
        self.move_mlp = mlp(32 + MOVE_F, 96, 64)
        self.mon_mlp = mlp(48 + 16 + 16 + MON_F + 128, 192, d)
        self.glob_mlp = mlp(GLOB_F, 128, d)
        self.token_type = nn.Embedding(3, d)  # ours / theirs / global
        enc_layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0,
                                               batch_first=True, activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.move_head = mlp(2 * d + 64, 128, 8)
        self.switch_head = mlp(3 * d, 128, 1)
        self.value_head = mlp(2 * d, 128, 1)
        self.pick_head = mlp(2 * d, 64, 1)
        self.lead_head = mlp(2 * d, 64, 1)
        self.register_buffer('_type_ids', torch.tensor([0] * 6 + [1] * 6 + [2], dtype=torch.long),
                             persistent=False)

    # ------------------------------------------------------------------
    def encode(self, obs: dict):
        ids, mon, move_ids, move = obs['ids'], obs['mon'], obs['move_ids'], obs['move']
        mv = torch.cat([self.move_emb(move_ids), move], dim=-1)          # [B,12,4,32+F]
        mv = self.move_mlp(mv)                                           # [B,12,4,64]
        known = (move_ids > 0).float().unsqueeze(-1)
        mv_mean = (mv * known).sum(2) / known.sum(2).clamp(min=1.0)
        mv_max = (mv * known - (1 - known) * 1e4).max(2).values
        mv_max = torch.where(known.sum(2) > 0, mv_max, torch.zeros_like(mv_max))
        tok = torch.cat([self.species_emb(ids[..., 0]), self.item_emb(ids[..., 1]),
                         self.ability_emb(ids[..., 2]), mon, mv_mean, mv_max], dim=-1)
        tok = self.mon_mlp(tok)                                          # [B,12,d]
        g = self.glob_mlp(obs['glob']).unsqueeze(1)                      # [B,1,d]
        x = torch.cat([tok, g], dim=1) + self.token_type(self._type_ids).unsqueeze(0)
        x = self.norm(self.encoder(x))
        return x, mv

    def forward(self, obs: dict):
        """Returns slot logits [B, 2, SLOT_ACTIONS + 1] (last column = pass), value [B], tokens."""
        x, mv = self.encode(obs)
        B = x.shape[0]
        ctx = x[:, N_MONS]                                               # global token
        act = obs['active_idx']                                          # [B,2] (-1 = none)
        logits = []
        bidx = torch.arange(B, device=x.device)
        mine = x[:, :6]                                                  # [B,6,d]
        for s in range(2):
            a = act[:, s].clamp(min=0)
            a_tok = x[bidx, a]                                           # [B,d]
            a_mv = mv[bidx, a]                                           # [B,4,64]
            inp = torch.cat([a_tok.unsqueeze(1).expand(-1, N_MOVES, -1), a_mv,
                             ctx.unsqueeze(1).expand(-1, N_MOVES, -1)], dim=-1)
            move_logits = self.move_head(inp).reshape(B, N_MOVES * 8)    # move i -> cols i*8..i*8+7
            sw_in = torch.cat([mine, a_tok.unsqueeze(1).expand(-1, 6, -1), ctx.unsqueeze(1).expand(-1, 6, -1)],
                              dim=-1)
            sw_logits = self.switch_head(sw_in).squeeze(-1)              # [B,6]
            pass_col = torch.zeros(B, 1, device=x.device)
            logits.append(torch.cat([move_logits, sw_logits, pass_col], dim=-1))
        slot_logits = torch.stack(logits, dim=1)                         # [B,2,39]
        pooled = torch.cat([ctx, x[:, :N_MONS].mean(1)], dim=-1)
        value = torch.tanh(self.value_head(pooled).squeeze(-1))
        return slot_logits, value, x

    def preview_scores(self, x):
        ctx = x[:, N_MONS].unsqueeze(1).expand(-1, 6, -1)
        mine = x[:, :6]
        pick = self.pick_head(torch.cat([mine, ctx], -1)).squeeze(-1)   # [B,6]
        lead = self.lead_head(torch.cat([mine, ctx], -1)).squeeze(-1)   # [B,6]
        return pick, lead


# ---------------------------------------------------------------------------
# Option scoring helpers (shared by acting and training)

def joint_logits(slot_logits: torch.Tensor, combos: torch.Tensor, combo_mask: torch.Tensor) -> torch.Tensor:
    """slot_logits [B,2,39], combos [B,L,2] (action ids, -1 = pass), combo_mask [B,L] -> [B,L] logits."""
    c = combos.clone()
    c[c < 0] = PASS_COL
    l0 = torch.gather(slot_logits[:, 0], 1, c[..., 0])
    l1 = torch.gather(slot_logits[:, 1], 1, c[..., 1])
    out = l0 + l1
    return out.masked_fill(~combo_mask, -1e9)


def preview_logits(pick: torch.Tensor, lead: torch.Tensor, options: torch.Tensor, n_lead: torch.Tensor,
                   option_mask: torch.Tensor) -> torch.Tensor:
    """options [B,O,K] team indices (-1 pad), n_lead [B] -> [B,O] logits."""
    valid = options >= 0
    idx = options.clamp(min=0)
    p = torch.gather(pick.unsqueeze(1).expand(-1, options.shape[1], -1), 2, idx) * valid
    lpos = torch.arange(options.shape[2], device=options.device).view(1, 1, -1)
    is_lead = (lpos < n_lead.view(-1, 1, 1)) & valid
    q = torch.gather(lead.unsqueeze(1).expand(-1, options.shape[1], -1), 2, idx) * is_lead
    out = p.sum(-1) + q.sum(-1)
    return out.masked_fill(~option_mask, -1e9)


# ---------------------------------------------------------------------------
# Batching

OBS_KEYS = ('ids', 'mon', 'move_ids', 'move', 'glob', 'active_idx')


def collate_obs(obs_list: list, device='cpu') -> dict:
    out = {}
    for k in OBS_KEYS:
        arr = np.stack([o[k] for o in obs_list])
        out[k] = torch.from_numpy(arr).to(device)
    return out


def pad_options(option_lists: list, width: int) -> tuple:
    """List of option lists (tuples) -> int tensor [B,L,width] (-1 pad) and bool mask [B,L]."""
    L = max(len(o) for o in option_lists)
    arr = np.full((len(option_lists), L, width), -1, dtype=np.int64)
    mask = np.zeros((len(option_lists), L), dtype=bool)
    for b, opts in enumerate(option_lists):
        for i, o in enumerate(opts):
            arr[b, i, :len(o)] = o
            mask[b, i] = True
    return torch.from_numpy(arr), torch.from_numpy(mask)


def save_model(model: PolicyValueNet, path: str, extra: dict | None = None):
    torch.save({'config': model.config, 'state_dict': model.state_dict(), 'vocab': vocab(),
                'extra': extra or {}}, path)


def load_model(path: str, map_location='cpu') -> PolicyValueNet:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ckpt['config']
    model = PolicyValueNet(d=cfg['d'], layers=cfg['layers'], heads=cfg['heads'], sizes=cfg['sizes'])
    if ckpt.get('vocab') and ckpt['vocab'] != vocab():
        raise ValueError('checkpoint vocabulary differs from the current data export')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model
