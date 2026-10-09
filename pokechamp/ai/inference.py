"""Bayesian inference of an opponent's hidden Stat Points, nature and Choice Scarf.

Players never see the opponent's nature or Stat Point (SP) spread, but the battle
leaks information about them:

* **speed order** - who moved first with equal priority bounds their Speed,
* **damage they take** from our (fully known) attacks bounds their HP and defences,
* **damage they deal** to us (we know our exact HP) bounds their attacking stat.

Each opponent Pokemon gets a :class:`StatBelief`: a weighted set of hypotheses
(SP spread x nature x Choice Scarf or not).  The prior mixes competitive spreads
(two stats at 32) with arbitrary random spreads; every observation multiplies the
weights by a likelihood with a small floor, so effects the model does not know
about (an unrevealed Assault Vest, an unknown ability, ...) cannot make the
belief collapse.  The posterior feeds the damage estimates, the speed
comparisons and the network features, and the advisor samples spreads from it.
"""
from __future__ import annotations

import copy
import itertools
import random
from functools import lru_cache

import numpy as np

from ..sim.dex import get_dex

STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
STAT_INDEX = {s: i for i, s in enumerate(STATS)}
EPS = 0.05            # likelihood of an observation that contradicts a hypothesis
SPEED_EPS = 0.03
SCARF_PRIOR = 0.12    # chance an unknown item is a Choice Scarf
TOL = 0.013           # HP-percentage rounding tolerance


@lru_cache(maxsize=1)
def _natures():
    """Distinct nature effects: 20 non-neutral natures + one neutral one."""
    dex = get_dex()
    out = []
    seen_neutral = False
    for n in sorted(dex.natures.all(), key=lambda n: n.name):
        if not n.plus:
            if seen_neutral:
                continue
            seen_neutral = True
            out.append(('Serious', None, None, 5.0))  # five neutral natures share one hypothesis
        else:
            out.append((n.name, n.plus, n.minus, 1.0))
    return out


@lru_cache(maxsize=1)
def _spreads():
    """(SP spreads [S,6], prior weight [S]): competitive templates + random spreads."""
    spreads, weights = [], []
    idx = range(6)
    for a, b in itertools.combinations(idx, 2):
        for c in idx:
            if c in (a, b):
                continue
            sp = [0] * 6
            sp[a] = sp[b] = 32
            sp[c] = 2
            spreads.append(sp)
            weights.append(3.0)
    for a in idx:
        for b, c in itertools.combinations([i for i in idx if i != a], 2):
            sp = [0] * 6
            sp[a] = 32
            sp[b] = sp[c] = 17
            spreads.append(sp)
            weights.append(1.0)
    rng = random.Random(1234)
    for _ in range(160):
        sp = [0] * 6
        budget = 66
        order = list(idx)
        rng.shuffle(order)
        for i in order:
            amount = rng.randint(0, min(32, budget))
            sp[i] = amount
            budget -= amount
        while budget > 0:
            cands = [i for i in idx if sp[i] < 32]
            if not cands:
                break
            i = rng.choice(cands)
            add = min(budget, 32 - sp[i], rng.randint(1, 8))
            sp[i] += add
            budget -= add
        spreads.append(sp)
        weights.append(1.2)
    return np.array(spreads, dtype=np.int32), np.array(weights, dtype=np.float64)


@lru_cache(maxsize=1)
def _hypotheses():
    """Flattened hypotheses: SP [H,6], nature multipliers [H,6], nature names, prior [H]."""
    spreads, sw = _spreads()
    natures = _natures()
    H = len(spreads) * len(natures)
    sp = np.repeat(spreads, len(natures), axis=0)
    mult = np.ones((H, 6), dtype=np.float64)
    names = []
    prior = np.repeat(sw, len(natures)).astype(np.float64)
    for k in range(len(spreads)):
        for j, (name, plus, minus, nw) in enumerate(natures):
            h = k * len(natures) + j
            names.append(name)
            if plus:
                mult[h, STAT_INDEX[plus]] = 1.1
                mult[h, STAT_INDEX[minus]] = 0.9
                # competitive spreads usually boost an invested stat and lower an unused one
                if sp[h, STAT_INDEX[plus]] >= 17 and sp[h, STAT_INDEX[minus]] == 0 and sw[k] >= 3.0:
                    prior[h] *= 3.0
            prior[h] *= nw
    return sp, mult, names, prior / prior.sum()


@lru_cache(maxsize=512)
def _stats_for(species: str) -> np.ndarray:
    """Raw stats [H,6] of every hypothesis for ``species`` (level 50 Champions formula)."""
    sp, mult, _names, _prior = _hypotheses()
    s = get_dex().species.get(species)
    base = np.array([s.baseStats[k] for k in STATS], dtype=np.float64) if s.exists else np.full(6, 80.0)
    raw = base[None, :] + sp
    stats = np.floor((raw + 20) * mult)
    stats[:, 0] = raw[:, 0] + 75
    return stats


@lru_cache(maxsize=512)
def _stats_doubled(species: str) -> np.ndarray:
    st = _stats_for(species)
    return np.concatenate([st, st], axis=0)


@lru_cache(maxsize=512)
def _speed_order(species: str, mult: float) -> tuple:
    """Sort order of the effective speeds of every (doubled, Scarf) hypothesis and the sorted speeds; they do
    not depend on the weights, so beliefs share them (read-only arrays)."""
    stats = _stats_doubled(species)
    n = stats.shape[0] // 2
    spe = stats[:, 5] * mult * np.where(np.concatenate([np.zeros(n, dtype=bool), np.ones(n, dtype=bool)]), 1.5, 1.0)
    order = np.argsort(spe)
    spe_sorted = spe[order]
    order = order.astype(np.int32)
    order.setflags(write=False)
    spe_sorted.setflags(write=False)
    return order, spe_sorted


@lru_cache(maxsize=1)
def _scarf_mult() -> np.ndarray:
    n = len(_hypotheses()[3])
    return np.concatenate([np.ones(n), np.full(n, 1.5)])


class StatBelief:
    """Posterior over (Stat Points, nature, Choice Scarf) of one opponent Pokemon."""

    def __init__(self):
        sp, _mult, _names, prior = _hypotheses()
        n = len(prior)
        # hypotheses are duplicated: first half without, second half with a Choice Scarf
        self.w = np.concatenate([prior * (1 - SCARF_PRIOR), prior * SCARF_PRIOR])
        self.scarf = np.concatenate([np.zeros(n, dtype=bool), np.ones(n, dtype=bool)])
        self.n = n
        self.observations = 0
        self._version = 0
        self._cache = {}

    # -- helpers -----------------------------------------------------------------
    def _stats(self, species: str) -> np.ndarray:
        return _stats_doubled(species)

    def _update(self, lik: np.ndarray):
        w = self.w * lik
        total = w.sum()
        if total <= 0 or not np.isfinite(total):
            return
        self.w = w / total
        self.observations += 1
        self._version += 1
        self._cache.clear()

    def set_item(self, item: str | None):
        """Known item: a Choice Scarf (or anything else) removes the other half of the hypotheses."""
        if item is None:
            return
        keep = self.scarf if item == 'Choice Scarf' else ~self.scarf
        w = np.where(keep, self.w, 0.0)
        if w.sum() > 0 and not np.allclose(w, self.w):
            self.w = w / w.sum()
            self._version += 1
            self._cache.clear()

    # -- observations --------------------------------------------------------------
    def observe_damage_taken(self, species: str, parts, attacker_stat: float, before: float, after: float,
                             crit: bool, fainted: bool):
        """We hit this Pokemon: HP fraction went from ``before`` to ``after``."""
        stats = self._stats(species)
        maxhp = stats[:, 0]
        lo, hi = parts.roll_range(attacker_stat, stats[:, STAT_INDEX[parts.def_key]], crit)
        lo, hi = lo * 0.98 / maxhp, hi * 1.02 / maxhp   # slack for rounding in the real formula
        lost = before - after
        if fainted:
            ok = hi >= before - TOL
        else:
            ok = (hi >= lost - TOL) & (lo <= lost + TOL)
        self._update(np.where(ok, 1.0, EPS))

    def observe_damage_dealt(self, species: str, parts, defender_stat: float, defender_maxhp: int,
                             dmg_hp: int, crit: bool, fainted: bool, item_known: bool):
        """This Pokemon hit us for ``dmg_hp`` HP (exact, since we know our own HP)."""
        stats = self._stats(species)
        lo, hi = parts.roll_range(stats[:, STAT_INDEX[parts.atk_key]], defender_stat, crit)
        lo, hi = lo * 0.98 - 1, hi * 1.02 + 1   # slack for rounding in the real formula
        if fainted:
            ok = hi >= dmg_hp - 1
            boosted = hi * 1.5 >= dmg_hp - 1
        else:
            ok = (hi >= dmg_hp - 1) & (lo <= dmg_hp + 1)
            boosted = (hi * 1.5 >= dmg_hp - 1) & (lo * 1.3 <= dmg_hp + 1)
        lik = np.where(ok, 1.0, EPS)
        if not item_known:
            # an unrevealed Choice Band / Specs could explain more damage than the spread alone
            lik = np.where(~ok & boosted, 0.35, lik)
        self._update(lik)

    def observe_speed(self, species: str, foe_mult: float, my_speed: float, foe_first: bool,
                      trick_room: bool, scarf_active: bool = True):
        """Same priority, both moved: did this Pokemon act before ours?"""
        stats = self._stats(species)
        spe = np.floor(stats[:, 5] * foe_mult * np.where(self.scarf & scarf_active, 1.5, 1.0))
        if trick_room:
            faster = spe < my_speed
        else:
            faster = spe > my_speed
        tie = spe == my_speed
        lik = np.where(tie, 0.5, np.where(faster == foe_first, 1.0, SPEED_EPS))
        self._update(lik)

    # -- queries -------------------------------------------------------------------
    def mean_stats(self, species: str) -> dict:
        means = self._cache.get('mean')
        if means is None:
            means = self._cache['mean'] = {}
        m = means.get(species)
        if m is None:
            v = self.w @ self._stats(species)
            m = means[species] = {s: float(v[i]) for i, s in enumerate(STATS)}
        return m

    def p_scarf(self) -> float:
        p = self._cache.get('scarf')
        if p is None:
            p = self._cache['scarf'] = float(self.w[self.scarf].sum())
        return p

    def speed_quantiles(self, species: str, mult: float = 1.0) -> tuple[float, float]:
        """(10%, 90%) quantiles of the effective speed (incl. a possible Scarf)."""
        key = ('q', species, mult)
        if key not in self._cache:
            order, spe_sorted = _speed_order(species, mult)
            cw = np.cumsum(self.w[order])
            q10 = spe_sorted[np.searchsorted(cw, 0.1)]
            q90 = spe_sorted[min(len(cw) - 1, np.searchsorted(cw, 0.9))]
            self._cache[key] = (float(q10), float(q90))
        return self._cache[key]

    def p_faster(self, species: str, foe_mult: float, my_speed: float, trick_room: bool,
                 scarf_active: bool = True) -> float:
        """Probability that this Pokemon moves before a Pokemon with effective speed ``my_speed``."""
        key = ('pf', species, foe_mult, my_speed, trick_room, scarf_active, self._version)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        stats = self._stats(species)
        spe = stats[:, 5] * foe_mult
        if scarf_active:
            spe = spe * _scarf_mult()
        spe = np.floor(spe)
        if trick_room:
            p = self.w[spe < my_speed].sum() + 0.5 * self.w[spe == my_speed].sum()
        else:
            p = self.w[spe > my_speed].sum() + 0.5 * self.w[spe == my_speed].sum()
        self._cache[key] = float(p)
        return float(p)

    def effective_samples(self) -> float:
        return float(1.0 / np.square(self.w).sum())

    def certainty(self) -> float:
        """0 = prior, -> 1 as the posterior concentrates (log effective sample size)."""
        c = self._cache.get('certainty')
        if c is None:
            full = np.log(2 * self.n)
            c = self._cache['certainty'] = float(max(0.0, 1.0 - np.log(max(1.0, self.effective_samples())) / full))
        return c

    def sample(self, rng: random.Random) -> tuple[str, dict, bool]:
        """Draw (nature, SP spread, holds a Choice Scarf) from the posterior."""
        sp, _mult, names, _prior = _hypotheses()
        u = rng.random()
        idx = int(np.searchsorted(np.cumsum(self.w), u * self.w.sum()))
        idx = min(idx, len(self.w) - 1)
        h = idx % self.n
        return names[h], {s: int(sp[h, i]) for i, s in enumerate(STATS)}, bool(self.scarf[idx])

    def most_likely(self, species: str) -> dict:
        """Readable summary for the advisor output."""
        stats = self._stats(species)
        m = self.mean_stats(species)
        q10, q90 = self.speed_quantiles(species)
        return {'mean_stats': {k: round(v) for k, v in m.items()}, 'speed_10_90': (round(q10), round(q90)),
                'p_choice_scarf': round(self.p_scarf(), 3), 'observations': self.observations,
                'max_hp_range': (int(stats[:, 0].min()), int(stats[:, 0].max()))}


# ---------------------------------------------------------------------------
# Feeding observations from the battle protocol

SKIP_INFERENCE_MOVES = {'gyroball', 'electroball', 'foulplay', 'beatup', 'present', 'magnitude', 'lastrespects',
                        'ragefist', 'payback', 'avalanche', 'revenge', 'boltbeak', 'fishiousrend', 'storedpower',
                        'powertrip', 'punishment', 'naturalgift', 'fling', 'spitup', 'tripleaxel', 'triplekick'}
ORDER_BREAKERS = ('Quick Claw', 'Custap Berry', 'Quick Draw', 'After You', 'Quash', 'Instruct', 'Shell Trap',
                  'Focus Punch', 'Beak Blast', 'Pursuit')


def make_inference_tracker(me: str, formatid: str = ''):
    from ..env.view import LogTracker, _kwargs, _parse_ident
    from .damage import damage_parts, move_info, move_priority, speed_multiplier

    class InferenceTracker(LogTracker):
        """:class:`LogTracker` that also maintains a :class:`StatBelief` for every opponent Pokemon."""

        def __init__(self):
            super().__init__(me, formatid)
            self._last_move = None
            self._crit = set()
            self._turn_moves = []
            self._turn_snapshot = {}
            self._turn_tainted = False
            self._switched_in = set()

        # -- beliefs -------------------------------------------------------------
        def belief(self, mon):
            if mon.belief is None:
                mon.belief = StatBelief()
            original = mon.last_item or mon.item
            if original is not None and not getattr(mon, '_belief_item_set', False):
                mon.belief.set_item(original)
                mon._belief_item_set = True
            return mon.belief

        def _is_foe(self, ident):
            return _parse_ident(ident)[0] != self.view.me

        # -- protocol hooks ----------------------------------------------------------
        def _switch_in(self, args, is_drag=False):
            super()._switch_in(args, is_drag)
            mon = self._get(args[0])
            self._switched_in.add(id(mon))
            if self._is_foe(args[0]):
                self.belief(mon)

        _on_switch = _switch_in

        def _on_drag(self, args):
            self._switch_in(args, True)

        def _on_move(self, args):
            super()._on_move(args)
            mon = self._get(args[0])
            kw = _kwargs(args[3:])
            src = kw.get('from', '')
            self._last_move = (mon, _parse_ident(args[0])[0], args[1], 'spread' in kw)
            self._crit = set()
            if src and src != 'lockedmove':
                return
            side = _parse_ident(args[0])[0]
            if side != self.view.me and not src:
                # remember the opponent's free choice and what it was facing
                target_ident = args[2] if len(args) > 2 else ''
                target = self._get(target_ident, create=False) if target_ident.startswith(self.view.me) else None
                if target is None:
                    act = self.view.my_side.active()
                    target = act[0] if act else None
                if target is not None:
                    snap = copy.copy(target)
                    snap.boosts = dict(target.boosts)
                    snap.volatiles = set(target.volatiles)
                    self.view.history.append({'foe': mon, 'move': args[1], 'target': snap, 'turn': self.view.turn})
            if not self._turn_moves:
                # speed-relevant state at the moment the turn order was decided
                self._turn_snapshot = {
                    'tr': 'trickroom' in self.view.field.pseudo,
                    'speed': {id(p): (p, s) for s in ('p1', 'p2') for p in self.view.sides[s].active()},
                }
            self._turn_moves.append((mon, _parse_ident(args[0])[0], args[1]))

        def _on__crit(self, args):
            mon = self._get(args[0])
            self._crit.add(id(mon))

        def _on__mega(self, args):
            super()._on__mega(args)
            self._turn_tainted = True

        def _on__activate(self, args):
            super()._on__activate(args)
            if any(b in a for a in args[1:2] for b in ORDER_BREAKERS):
                self._turn_tainted = True

        def _on__item(self, args):
            super()._on__item(args)
            mon = self._get(args[0])
            if mon.belief is not None:
                mon._belief_item_set = False
                self.belief(mon)

        def _on__enditem(self, args):
            super()._on__enditem(args)
            mon = self._get(args[0])
            if mon.belief is not None:
                mon._belief_item_set = False
                self.belief(mon)

        def _on__damage(self, args, heal=False):
            target = self._get(args[0])
            before_frac, before_exact = target.hp, target.hp_exact
            super()._on__damage(args, heal)
            if heal or self._last_move is None:
                return
            kw = _kwargs(args[2:])
            if 'from' in kw:
                return
            attacker, a_side, move_name, spread = self._last_move
            t_side = _parse_ident(args[0])[0]
            if a_side == t_side or attacker is target:
                return
            move = move_info(move_name)
            if move is None or move.multihit or move.id in SKIP_INFERENCE_MOVES:
                return
            try:
                parts = damage_parts(attacker, target, move_name, self.view, spread)
            except Exception:
                return
            if parts is None or parts.kind != 'normal' or parts.atk_from_target or parts.hits != 1:
                return
            crit = id(target) in self._crit
            if t_side != self.view.me:
                # we hit the opponent: information about its HP / defences
                if not attacker.stats:
                    return
                atk_stat = attacker.stats.get(parts.atk_key)
                if atk_stat is None:
                    return
                self.belief(target).observe_damage_taken(target.species, parts, atk_stat, before_frac, target.hp,
                                                         crit, target.fainted)
            else:
                # the opponent hit us: information about its attacking stat
                if not target.stats or before_exact is None or target.hp_exact is None or not target.maxhp:
                    return
                def_stat = target.stats.get(parts.def_key)
                if def_stat is None:
                    return
                self.belief(attacker).observe_damage_dealt(attacker.species, parts, def_stat, target.maxhp,
                                                           before_exact - target.hp_exact, crit, target.fainted,
                                                           attacker.item is not None)

        def _on_turn(self, args):
            self._speed_inference()
            super()._on_turn(args)
            self._turn_moves = []
            self._turn_tainted = False
            self._switched_in = set()
            self._last_move = None

        def _on_upkeep(self, args):
            self._speed_inference()
            self._turn_moves = []
            self._turn_snapshot = {}

        def _speed_inference(self):
            moves = self._turn_moves
            if len(moves) < 2 or self._turn_tainted or not self._turn_snapshot:
                return
            first_mon, first_side, first_move = moves[0]
            if id(first_mon) in self._switched_in and self.view.turn > 0:
                return
            snap = self._turn_snapshot
            if id(first_mon) not in snap['speed']:
                return
            pr = move_priority(first_move, first_mon, self.view)
            for mon, side, move_name in moves[1:]:
                if side == first_side or id(mon) not in snap['speed'] or id(mon) in self._switched_in:
                    continue
                if move_priority(move_name, mon, self.view) != pr:
                    continue
                if first_side == self.view.me:
                    mine, foe, foe_first = first_mon, mon, False
                else:
                    mine, foe, foe_first = mon, first_mon, True
                if not mine.stats:
                    continue
                my_speed = mine.stats['spe'] * speed_multiplier(mine, self.view, self.view.me)
                foe_side = self.view.foe
                mult = speed_multiplier(foe, self.view, foe_side, include_scarf=False)
                self.belief(foe).observe_speed(foe.species, mult, my_speed, foe_first, snap['tr'],
                                               scarf_active=foe.item in (None, 'Choice Scarf'))

        def update_request(self, request):
            super().update_request(request)
            for mon in self.view.foe_side.pokemon:
                if mon.revealed:
                    self.belief(mon)

    return InferenceTracker()
