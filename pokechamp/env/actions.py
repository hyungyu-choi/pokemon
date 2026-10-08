"""Action space: legal choices for a battle request and their Showdown choice strings.

Every active slot picks one of ``SLOT_ACTIONS`` actions:

* ``0..31`` - use move ``i`` (0-3) at target ``t`` (0-3), with Mega Evolution ``m`` (0/1):
  ``index = i * 8 + t * 2 + m``.  Targets: 0 = automatic (singles, or moves without
  a chosen target), 1 = foe in slot a, 2 = foe in slot b, 3 = ally (or self for
  ``adjacentAllyOrSelf`` moves when there is no ally).
* ``32..37`` - switch to the Pokemon at position ``j`` (0-5) of the request's team list.

In singles only targets 0 are used.  Team preview is a separate decision: an
ordering of the team (``team 312456``) where the first ``pickedTeamSize``
Pokemon are brought and the first one (two in doubles) lead.
"""
from __future__ import annotations

import itertools

N_MOVES = 4
N_TARGETS = 4
SWITCH_BASE = N_MOVES * N_TARGETS * 2
SLOT_ACTIONS = SWITCH_BASE + 6
PASS = -1

TARGETED = {'normal', 'any', 'adjacentFoe'}


def move_action(move_index: int, target: int = 0, mega: bool = False) -> int:
    return move_index * 8 + target * 2 + int(mega)


def switch_action(position: int) -> int:
    return SWITCH_BASE + position


def decode(action: int) -> tuple:
    """-> ('move', i, target, mega) | ('switch', position) | ('pass',)"""
    if action == PASS:
        return ('pass',)
    if action >= SWITCH_BASE:
        return ('switch', action - SWITCH_BASE)
    return ('move', action // 8, (action // 2) % 4, bool(action % 2))


def _fainted(entry) -> bool:
    return entry['condition'].endswith(' fnt')


def slot_legal_actions(request: dict, slot: int, n_active: int, foe_alive=(True, True)) -> list[int]:
    """Legal actions of active slot ``slot`` (ignoring cross-slot constraints)."""
    side_pokemon = request['side']['pokemon']
    if request.get('wait') or request.get('teamPreview'):
        return []
    if 'forceSwitch' in request:
        flags = request['forceSwitch']
        if slot >= len(flags) or not flags[slot]:
            return [PASS]
        reviving = side_pokemon[slot].get('reviving')
        out = []
        for j, p in enumerate(side_pokemon):
            if (j >= len(flags) or reviving) and (not _fainted(p)) != bool(reviving) and not p.get('active'):
                out.append(switch_action(j))
        if reviving:
            out = [switch_action(j) for j, p in enumerate(side_pokemon) if _fainted(p)]
        return out or [PASS]
    actives = request.get('active') or []
    if slot >= len(actives) or actives[slot] is None:
        return [PASS]
    me = side_pokemon[slot]
    if _fainted(me) or me.get('commanding'):
        return [PASS]
    areq = actives[slot]
    has_ally = (n_active > 1 and len(side_pokemon) > (slot ^ 1) and
                not _fainted(side_pokemon[slot ^ 1]) and side_pokemon[slot ^ 1].get('active'))
    out = []
    megas = (False, True) if areq.get('canMegaEvo') else (False,)
    for i, m in enumerate(areq['moves'][:N_MOVES]):
        if m.get('disabled'):
            continue
        tgt = m.get('target')
        if n_active == 1:
            targets = [0]
        elif tgt in TARGETED:
            targets = [t for t, alive in ((1, foe_alive[0]), (2, foe_alive[1])) if alive] or [1]
            if tgt in ('normal', 'any') and has_ally:
                targets.append(3)
        elif tgt == 'adjacentAlly':
            targets = [3] if has_ally else []
        elif tgt == 'adjacentAllyOrSelf':
            targets = [3, 0] if has_ally else [0]
        else:
            targets = [0]
        for t in targets:
            for mg in megas:
                out.append(move_action(i, t, mg))
    if not areq.get('trapped'):
        for j, p in enumerate(side_pokemon):
            if j < n_active or p.get('active') or _fainted(p):
                continue
            out.append(switch_action(j))
    if not out:
        out = [move_action(0)]
    return out


def legal_joint_actions(request: dict, n_active: int, foe_alive=(True, True)) -> list[tuple]:
    """All legal combinations of slot actions (switch targets distinct, at most one Mega Evolution)."""
    per_slot = [slot_legal_actions(request, s, n_active, foe_alive) for s in range(n_active)]
    if n_active == 1:
        return [(a,) for a in per_slot[0]]
    out = []
    for combo in itertools.product(*per_slot):
        if combo_is_legal(combo):
            out.append(combo)
    if not out:
        # e.g. two forced switches but only one Pokemon left: second slot passes
        for combo in itertools.product(*[p + [PASS] for p in per_slot]):
            if combo_is_legal(combo) and any(a != PASS for a in combo):
                out.append(combo)
    return out


def combo_is_legal(combo) -> bool:
    switches = [a for a in combo if a != PASS and a >= SWITCH_BASE]
    if len(switches) != len(set(switches)):
        return False
    megas = [a for a in combo if a != PASS and a < SWITCH_BASE and a % 2 == 1]
    return len(megas) <= 1


def slot_choice(request: dict, slot: int, action: int) -> str:
    kind = decode(action)
    if kind[0] == 'pass':
        return 'pass'
    if kind[0] == 'switch':
        return f"switch {kind[1] + 1}"
    _, i, target, mega = kind
    s = f"move {i + 1}"
    if target in (1, 2):
        s += f" {target}"
    elif target == 3:
        s += f" -{(slot ^ 1) + 1}"
    elif target == 0 and _needs_self_target(request, slot, i):
        s += f" -{slot + 1}"
    if mega:
        s += ' mega'
    return s


def _needs_self_target(request, slot, move_index) -> bool:
    actives = request.get('active') or []
    if len(actives) < 2 or slot >= len(actives) or not actives[slot]:
        return False
    moves = actives[slot]['moves']
    return move_index < len(moves) and moves[move_index].get('target') == 'adjacentAllyOrSelf'


def joint_choice(request: dict, combo) -> str:
    return ', '.join(slot_choice(request, s, a) for s, a in enumerate(combo))


# ---------------------------------------------------------------------------
# Team preview

def team_preview_options(team_size: int, picked: int, n_lead: int) -> list[tuple]:
    """Distinct team-preview decisions as tuples of team indices (leads first).

    Singles, pick 3 of 6: lead + 2 back (order of the back Pokemon irrelevant) -> 6 * C(5,2) = 60.
    Doubles, pick 4 of 6: 2 leads + 2 back -> C(6,2) * C(4,2) = 90.
    """
    idx = range(team_size)
    out = []
    for leads in itertools.combinations(idx, n_lead):
        rest = [i for i in idx if i not in leads]
        for back in itertools.combinations(rest, max(0, picked - n_lead)):
            out.append(tuple(leads) + tuple(back))
    return out


def team_choice(order: tuple, team_size: int) -> str:
    """Showdown 'team' choice; the unpicked Pokemon follow in their original order."""
    full = list(order) + [i for i in range(team_size) if i not in order]
    return 'team ' + ''.join(str(i + 1) for i in full)
