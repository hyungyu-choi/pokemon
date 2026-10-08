"""Replay a Showdown input log with the Python engine and diff the battle logs."""
from __future__ import annotations

import json
from typing import Iterable

from ..sim.battle import Battle


def replay_input_log(input_log: Iterable[str], timestamp: int = 0) -> Battle:
    battle = None
    for line in input_log:
        if line.startswith('>version'):
            continue
        if line.startswith('>start '):
            opts = json.loads(line[len('>start '):])
            battle = Battle(opts['formatid'], seed=opts.get('seed'), timestamp=timestamp)
        elif line.startswith('>player '):
            rest = line[len('>player '):]
            slot, data = rest.split(' ', 1)
            battle.setPlayer(slot, json.loads(data))
        elif line[:3] in ('>p1', '>p2', '>p3', '>p4'):
            sideid, choice = line[1:].split(' ', 1)
            ok = battle.choose(sideid, choice)
            if not ok:
                raise ValueError(f"choice rejected by python engine: {line} :: "
                                 f"{battle.getSide(sideid).choice.error}")
        elif line.startswith('>forcewin') or line.startswith('>forcetie') or line.startswith('>tiebreak'):
            pass
    return battle


def replay_result(result: dict, timestamp: int = 0) -> Battle:
    """Replay a reference-runner result: battle setup from its input log, then its raw choices."""
    log = result['inputLog']
    if result.get('choices') is not None:
        setup = [l for l in log if not l[:3] in ('>p1', '>p2', '>p3', '>p4')]
        log = setup + list(result['choices'])
    return replay_input_log(log, timestamp)


def strip_timestamps(log: Iterable[str]) -> list[str]:
    return [l for l in log if not l.startswith('|t:|')]


def first_divergence(expected: list[str], actual: list[str]):
    exp = strip_timestamps(expected)
    act = strip_timestamps(actual)
    for i, (a, b) in enumerate(zip(exp, act)):
        if a != b:
            return i, exp, act
    if len(exp) != len(act):
        return min(len(exp), len(act)), exp, act
    return None


def diff_report(expected: list[str], actual: list[str], context: int = 8) -> str:
    d = first_divergence(expected, actual)
    if d is None:
        return 'identical'
    i, exp, act = d
    lo = max(0, i - context)
    out = [f"first divergence at line {i}"]
    out.append('--- showdown ---')
    out += [f"{'>' if j == i else ' '} {exp[j]}" for j in range(lo, min(len(exp), i + context))]
    out.append('--- python ---')
    out += [f"{'>' if j == i else ' '} {act[j]}" for j in range(lo, min(len(act), i + context))]
    return '\n'.join(out)
