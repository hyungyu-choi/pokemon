"""Rule-based agents: max-damage and a stronger heuristic player (singles and doubles).

They serve as baselines, as opponents in the training league and as the teacher
for behaviour cloning before reinforcement learning.
"""
from __future__ import annotations

import random

from ..env.actions import PASS, decode
from ..env.view import BattleView, PokemonView
from .damage import (best_damage, estimate_damage, likely_moves, move_info, move_priority, outspeeds,
                     species_info)

SETUP_MOVES = {
    'swordsdance': ('atk', 2), 'dragondance': ('atk', 1), 'nastyplot': ('spa', 2), 'calmmind': ('spa', 1),
    'quiverdance': ('spa', 1), 'bulkup': ('atk', 1), 'shellsmash': ('atk', 2), 'irondefense': ('def', 2),
    'shiftgear': ('atk', 1), 'coil': ('atk', 1), 'victorydance': ('atk', 1), 'tidyup': ('atk', 1),
    'curse': ('atk', 1), 'agility': ('spe', 2), 'rockpolish': ('spe', 2), 'tailglow': ('spa', 3),
    'growth': ('atk', 1), 'workup': ('atk', 1), 'honeclaws': ('atk', 1), 'howl': ('atk', 1),
    'filletaway': ('atk', 2), 'clangoroussoul': ('atk', 1), 'noretreat': ('atk', 1), 'bellydrum': ('atk', 6),
    'amnesia': ('spd', 2), 'cosmicpower': ('def', 1), 'acidarmor': ('def', 2), 'barrier': ('def', 2),
    'cottonguard': ('def', 3), 'takeheart': ('spa', 1), 'geomancy': ('spa', 2),
}
RECOVERY_MOVES = {'recover', 'roost', 'slackoff', 'softboiled', 'moonlight', 'morningsun', 'synthesis',
                  'milkdrink', 'shoreup', 'healorder', 'strengthsap', 'rest', 'lifedew', 'junglehealing'}
HAZARDS = {'stealthrock': 'stealthrock', 'spikes': 'spikes', 'toxicspikes': 'toxicspikes',
           'stickyweb': 'stickyweb', 'stoneaxe': None, 'ceaselessedge': None}
SCREENS = {'reflect': 'reflect', 'lightscreen': 'lightscreen', 'auroraveil': 'auroraveil'}
STATUS_MOVES = {
    'willowisp': 'brn', 'thunderwave': 'par', 'toxic': 'tox', 'spore': 'slp', 'sleeppowder': 'slp',
    'hypnosis': 'slp', 'yawn': 'slp', 'glare': 'par', 'stunspore': 'par', 'nuzzle': 'par', 'poisonpowder': 'psn',
    'darkvoid': 'slp', 'lovelykiss': 'slp', 'sing': 'slp', 'grasswhistle': 'slp',
}
PROTECT_MOVES = {'protect', 'detect', 'kingsshield', 'spikyshield', 'banefulbunker', 'silktrap',
                 'burningbulwark', 'obstruct'}


def _req_moves(request, slot):
    act = (request.get('active') or [None])[slot] if request.get('active') else None
    return act['moves'] if act else []


def _my_mon(view: BattleView, request, position: int) -> PokemonView | None:
    entry = request['side']['pokemon'][position]
    name = entry['ident'].split(': ', 1)[1]
    return view.my_side.find(name)


def matchup(me: PokemonView, foe: PokemonView, view: BattleView) -> float:
    """> 0: ``me`` is favoured against ``foe`` (damage race with speed)."""
    my_dmg, my_ko, _ = best_damage(me, foe, view)
    foe_dmg, foe_ko, _ = best_damage(foe, me, view, likely_moves(foe))
    fast = outspeeds(me, view.me, foe, view.foe, view)
    # turns to KO each other
    my_turns = foe.hp / max(my_dmg, 1e-3)
    foe_turns = me.hp / max(foe_dmg, 1e-3)
    score = (foe_turns - my_turns) / max(1.0, min(foe_turns, my_turns))
    score += 0.3 * (fast - 0.5)
    return max(-3.0, min(3.0, score))


class MaxDamageAgent:
    """Always uses the move with the highest estimated damage (Mega Evolves when possible)."""

    name = 'maxdamage'

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose(self, session, sid, view, legal):
        request = session.request(sid)
        if request.get('teamPreview'):
            return self.rng.choice(legal)
        if 'forceSwitch' in request:
            return self._best_switch(session, view, request, legal)
        best, best_v = None, -1e9
        for option in legal:
            v = 0.0
            for slot, a in enumerate(option):
                v += self._slot_value(view, request, slot, a)
            v += self.rng.random() * 1e-3
            if v > best_v:
                best, best_v = option, v
        return best

    def _slot_value(self, view, request, slot, a):
        if a == PASS:
            return 0.0
        kind = decode(a)
        if kind[0] == 'switch':
            return -1.0
        _, i, target, mega = kind
        moves = _req_moves(request, slot)
        if i >= len(moves):
            return -1.0
        me = view.my_side.active_at(slot)
        foes = [p for p in view.foe_side.active() if p.alive]
        if me is None or not foes:
            return 0.0
        name = moves[i]['move']
        if target in (1, 2):
            foe = view.foe_side.active_at(target - 1)
            foes = [foe] if foe is not None and foe.alive else foes[:1]
        elif target == 3:
            return -0.5
        d = max(estimate_damage(me, f, name, view)[0] for f in foes)
        return d + (0.01 if mega else 0.0)

    def _best_switch(self, session, view, request, legal):
        foes = [p for p in view.foe_side.active() if p.alive]
        best, best_v = legal[0], -1e9
        for option in legal:
            v = 0.0
            for a in option:
                if a == PASS or decode(a)[0] != 'switch':
                    continue
                mon = _my_mon(view, request, decode(a)[1])
                if mon is not None and foes:
                    v += max(best_damage(mon, f, view)[0] for f in foes)
            if v > best_v:
                best, best_v = option, v
        return best


class HeuristicAgent:
    """A reasonably strong rule-based player.

    * Team preview: picks the subset (and lead) with the best estimated matchups.
    * Attacks with the best expected damage, prefers KOs and priority when slower.
    * Switches out of losing matchups, sets up when safe, spreads status, sets hazards
      and screens early, heals when low, Mega Evolves as soon as possible.
    """

    name = 'heuristic'

    def __init__(self, seed=None, randomness: float = 0.0):
        self.rng = random.Random(seed)
        self.randomness = randomness

    # -- team preview --------------------------------------------------------
    def choose(self, session, sid, view, legal):
        request = session.request(sid)
        if request.get('teamPreview'):
            return self.team_preview(view, request, legal)
        if self.randomness and self.rng.random() < self.randomness:
            return self.rng.choice(legal)
        scored = []
        for option in legal:
            v = sum(self.slot_value(view, request, slot, a) for slot, a in enumerate(option))
            if len(option) == 2 and option[0] != PASS and option[1] != PASS:
                v += self._pair_bonus(view, request, option)
            scored.append((v + self.rng.random() * 1e-3, option))
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    def team_preview(self, view, request, legal):
        if not view.foe_side.pokemon:
            return self.rng.choice(legal)
        scores = self.preview_scores(view, legal)
        best, best_v = legal[0], -1e9
        for option, v in zip(legal, scores):
            v += self.rng.random() * 0.05
            if v > best_v:
                best, best_v = option, v
        return best

    def preview_scores(self, view, legal) -> list[float]:
        """Matchup score of every team-preview option (higher = better)."""
        mine = view.my_side.pokemon
        foes = view.foe_side.pokemon
        if not foes:
            return [0.0] * len(legal)
        m = [[matchup(a, b, view) for b in foes] for a in mine]
        n_lead = view.n_active
        out = []
        for option in legal:
            # coverage: for each foe, how well do the brought Pokemon handle it
            cover = sum(max(m[i][j] for i in option) for j in range(len(foes)))
            lead = sum(sum(m[i][j] for j in range(len(foes))) for i in option[:n_lead]) / len(foes)
            out.append(cover + 0.5 * lead)
        return out

    # -- per-slot evaluation ---------------------------------------------------
    def slot_value(self, view: BattleView, request, slot: int, a: int) -> float:
        if a == PASS:
            return 0.0
        kind = decode(a)
        me = view.my_side.active_at(slot)
        foes = [p for p in view.foe_side.active() if p.alive]
        if kind[0] == 'switch':
            mon = _my_mon(view, request, kind[1])
            if mon is None or not foes:
                return 0.0
            forced = 'forceSwitch' in request
            score = max(matchup(mon, f, view) for f in foes) * 0.3
            score += 0.2 * mon.hp
            if forced:
                return score
            cur = max(matchup(me, f, view) for f in foes) * 0.3 if me is not None else -1.0
            # switching costs a turn and takes a hit
            incoming = max(best_damage(f, mon, view, likely_moves(f))[0] for f in foes)
            return score - cur - 0.25 - 0.4 * incoming + self._hazard_cost(view)
        _, i, target, mega = kind
        moves = _req_moves(request, slot)
        if me is None or i >= len(moves) or not foes:
            return 0.0
        mv = moves[i]
        name = mv['move']
        move = move_info(name)
        if move is None:
            return 0.0
        value = 0.05 if mega else 0.0
        if target in (1, 2):
            foe = view.foe_side.active_at(target - 1)
            targets = [foe] if foe is not None and foe.alive else foes[:1]
        elif target == 3:
            return self._ally_move_value(view, me, move, slot)
        elif move.target in ('allAdjacentFoes', 'allAdjacent'):
            targets = foes
        else:
            targets = foes[:1]
        foe = targets[0]
        fast = outspeeds(me, view.me, foe, view.foe, view)
        foe_dmg, foe_ko, _ = best_damage(foe, me, view, likely_moves(foe))
        if move.category != 'Status':
            spread = len(targets) > 1
            total = 0.0
            for t in targets:
                d, ko = estimate_damage(me, t, name, view, spread=spread)
                d = min(d, t.hp)
                pr = move_priority(name, me, view)
                moves_first = 1.0 if pr > 0 else fast if pr == 0 else 0.0
                total += d + ko * (0.6 + 0.4 * moves_first)
                if move.target == 'allAdjacent' and view.n_active > 1:
                    ally = view.my_side.active_at(slot ^ 1)
                    if ally is not None and ally.alive:
                        total -= 0.7 * min(ally.hp, estimate_damage(me, ally, name, view, spread=True)[0])
            # recoil / self-drops
            if move.recoil:
                total -= 0.1
            if move.self and move.self.get('boosts'):
                total -= 0.05 * sum(abs(v) for v in move.self['boosts'].values() if v < 0)
            if move.flags.get('charge') and not me.item == 'Power Herb' and move.id != 'solarbeam':
                total *= 0.5
            if move.id in ('fakeout', 'firstimpression') and me.last_move:
                total = -1.0
            if move.flags.get('recharge'):
                total *= 0.7
            return value + total
        return value + self._status_value(view, me, foe, move, fast, foe_dmg, foe_ko, slot)

    def _status_value(self, view, me, foe, move, fast, foe_dmg, foe_ko, slot) -> float:
        mid = move.id
        threat = foe_dmg  # fraction of our HP the foe takes per turn
        safe = threat < 0.35 and not (foe_ko > 0.5)
        if mid in SETUP_MOVES:
            stat, n = SETUP_MOVES[mid]
            cur = me.boosts.get(stat, 0)
            if cur >= 4 or (mid == 'bellydrum' and me.hp < 0.6):
                return -0.5
            v = 0.45 if safe and me.hp > 0.6 else 0.05
            return v * (1.0 if cur < 2 else 0.4)
        if mid in RECOVERY_MOVES:
            if mid == 'rest' and me.status == 'slp':
                return -1.0
            missing = 1.0 - me.hp
            if missing < 0.35:
                return -0.2
            return missing * (0.9 if threat < 0.5 else 0.3)
        if mid in STATUS_MOVES:
            status = STATUS_MOVES[mid]
            if foe.status or ('substitute' in foe.volatiles):
                return -0.5
            if status == 'brn' and 'Fire' in foe.types:
                return -0.5
            if status == 'par' and ('Electric' in foe.types or (mid == 'thunderwave' and 'Ground' in foe.types)):
                return -0.5
            if status in ('tox', 'psn') and ('Poison' in foe.types or 'Steel' in foe.types):
                return -0.5
            if mid in ('spore', 'sleeppowder', 'stunspore', 'poisonpowder') and 'Grass' in foe.types:
                return -0.5
            if status == 'slp':
                if any(p.status == 'slp' for p in view.foe_side.pokemon if p.alive and p is not foe):
                    return -0.5
                return 0.55 * (move_info(move.name).accuracy if move.accuracy is not True else 100) / 100
            if status == 'brn':
                info = species_info(foe.species)
                physical = info is not None and info[0]['atk'] >= info[0]['spa']
                return 0.4 if physical else 0.15
            if status == 'par':
                return 0.35 if not fast else 0.15
            return 0.3
        if mid in HAZARDS and HAZARDS[mid]:
            cond = HAZARDS[mid]
            layers = view.foe_side.conditions.get(cond, 0)
            maxl = {'spikes': 3, 'toxicspikes': 2}.get(cond, 1)
            if layers >= maxl:
                return -0.5
            remaining = sum(1 for p in view.foe_side.pokemon if not p.fainted) - len(view.foe_side.active())
            remaining = max(remaining, view.foe_side.alive_count() - len(view.foe_side.active()))
            return 0.12 * remaining if safe or view.turn <= 2 else 0.05
        if mid in SCREENS:
            if SCREENS[mid] in view.my_side.conditions:
                return -0.5
            if mid == 'auroraveil' and view.field.weather not in ('snowscape', 'snow', 'hail'):
                return -0.5
            return 0.3
        if mid in PROTECT_MOVES:
            if me.last_move and move_info(me.last_move) and move_info(me.last_move).id in PROTECT_MOVES:
                return -0.6
            return 0.15 if view.n_active > 1 else 0.05
        if mid in ('taunt',):
            return 0.1 if 'taunt' not in foe.volatiles else -0.5
        if mid in ('trickroom',):
            mine_slow = not fast
            if 'trickroom' in view.field.pseudo:
                return 0.2 if not mine_slow else -0.5
            return 0.35 if mine_slow else -0.3
        if mid in ('tailwind',):
            return -0.5 if 'tailwind' in view.my_side.conditions else (0.35 if not fast else 0.1)
        if mid in ('substitute',):
            if 'substitute' in me.volatiles or me.hp < 0.3:
                return -0.5
            return 0.15 if safe else 0.0
        if mid in ('leechseed',):
            return -0.5 if ('Grass' in foe.types or 'leechseed' in foe.volatiles) else 0.2
        if mid in ('haze', 'clearsmog') and sum(max(0, v) for v in foe.boosts.values()) >= 2:
            return 0.5
        if mid in ('roar', 'whirlwind', 'dragontail') and sum(max(0, v) for v in foe.boosts.values()) >= 2:
            return 0.5
        if mid in ('encore',) and foe.last_move:
            return 0.2
        if mid in ('uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport'):
            return 0.1
        if mid in ('sunnyday', 'raindance', 'sandstorm', 'snowscape'):
            weather = {'sunnyday': 'sunnyday', 'raindance': 'raindance', 'sandstorm': 'sandstorm',
                       'snowscape': 'snowscape'}[mid]
            return -0.5 if view.field.weather == weather else 0.1
        return 0.0

    def _ally_move_value(self, view, me, move, slot) -> float:
        ally = view.my_side.active_at(slot ^ 1)
        if ally is None or not ally.alive:
            return -1.0
        if move.id in ('pollenpuff', 'healpulse', 'floralhealing', 'lifedew'):
            return 0.6 * (1 - ally.hp) - 0.1
        if move.category != 'Status':
            return -0.5
        return 0.05

    def _pair_bonus(self, view, request, option) -> float:
        # avoid both slots focusing a foe that the first attack already KOs
        targets = []
        for slot, a in enumerate(option):
            kind = decode(a)
            if kind[0] == 'move' and kind[2] in (1, 2):
                targets.append(kind[2])
        if len(targets) == 2 and targets[0] == targets[1]:
            foe = view.foe_side.active_at(targets[0] - 1)
            if foe is not None:
                me0 = view.my_side.active_at(0)
                moves0 = _req_moves(request, 0)
                k0 = decode(option[0])
                if me0 is not None and k0[1] < len(moves0):
                    _, ko = estimate_damage(me0, foe, moves0[k0[1]]['move'], view)
                    return -0.5 * ko
        return 0.0

    def _hazard_cost(self, view) -> float:
        c = view.my_side.conditions
        cost = 0.0
        if 'stealthrock' in c:
            cost += 0.08
        cost += 0.05 * c.get('spikes', 0)
        return -cost
