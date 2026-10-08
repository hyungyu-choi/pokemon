"""Team (de)serialisation: Showdown packed format and the human-readable export format.

In Pokemon Champions, stat investment uses Stat Points (SP): up to 32 per stat
and 66 in total, at level 50 with all IVs fixed at 31. Teams store SP in the
``evs`` field (exactly like Showdown's champions mod) and accept ``SPs:`` (or
``EVs:``) lines in the text format.
"""
from __future__ import annotations

import re

from .js import Obj, to_id

STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
STAT_NAMES = {'HP': 'hp', 'Atk': 'atk', 'Def': 'def', 'SpA': 'spa', 'SpD': 'spd', 'Spe': 'spe'}
STAT_LABELS = {v: k for k, v in STAT_NAMES.items()}


def _dex():
    from .dex import get_dex
    return get_dex()


def pack_name(name) -> str:
    if not name:
        return ''
    return re.sub(r'[^A-Za-z0-9]+', '', name)


def unpack_name(name: str, table=None) -> str:
    if not name:
        return ''
    if table is not None:
        obj = table.get(name)
        if obj.exists:
            return obj.name
    name = re.sub(r'([0-9]+)', r' \1 ', name)
    name = re.sub(r'([A-Z])', r' \1', name)
    name = re.sub(r'[ ][ ]', ' ', name)
    return name.strip()


def pack_team(team) -> str:
    if not team:
        return ''

    def get_iv(ivs, s):
        v = ivs.get(s)
        return '' if v == 31 or v is None else str(v)

    buf = ''
    for set_ in team:
        if buf:
            buf += ']'
        name = set_.get('name') or set_.get('species')
        buf += name
        sid = pack_name(set_.get('species') or set_.get('name'))
        buf += '|' + ('' if pack_name(set_.get('name') or set_.get('species')) == sid else sid)
        buf += '|' + pack_name(set_.get('item'))
        buf += '|' + pack_name(set_.get('ability'))
        buf += '|' + ','.join(pack_name(m) for m in set_.get('moves') or [])
        buf += '|' + (set_.get('nature') or '')
        evs = '|'
        if set_.get('evs'):
            e = set_['evs']
            evs = '|' + ','.join(str(e.get(s) or '') if e.get(s) else '' for s in STATS)
        buf += '|' if evs == '|,,,,,' else evs
        buf += f"|{set_['gender']}" if set_.get('gender') else '|'
        ivs = '|'
        if set_.get('ivs'):
            ivs = '|' + ','.join(get_iv(set_['ivs'], s) for s in STATS)
        buf += '|' if ivs == '|,,,,,' else ivs
        buf += '|S' if set_.get('shiny') else '|'
        level = set_.get('level')
        buf += f"|{level}" if level and level != 100 else '|'
        happiness = set_.get('happiness')
        buf += f"|{happiness}" if happiness is not None and happiness != 255 else '|'
        if set_.get('pokeball') or set_.get('hpType') or set_.get('teraType'):
            buf += f",{set_.get('hpType') or ''}"
            buf += f",{pack_name(set_.get('pokeball') or '')}"
            buf += ","
            buf += ","
            buf += f",{set_.get('teraType') or ''}"
    return buf


def unpack_team(buf: str):
    if not buf:
        return None
    dex = _dex()
    team = []
    i = 0
    for _ in range(24):
        set_ = Obj()
        team.append(set_)
        j = buf.find('|', i)
        if j < 0:
            return None
        set_.name = buf[i:j]
        i = j + 1
        j = buf.find('|', i)
        if j < 0:
            return None
        set_.species = unpack_name(buf[i:j], dex.species) or set_.name
        i = j + 1
        j = buf.find('|', i)
        if j < 0:
            return None
        set_.item = unpack_name(buf[i:j], dex.items)
        i = j + 1
        j = buf.find('|', i)
        if j < 0:
            return None
        ability = buf[i:j]
        species = dex.species.get(set_.species)
        if ability in ('', '0', '1', 'H', 'S'):
            abilities = species.abilities or {}
            set_.ability = abilities.get(ability or '0') or ('' if ability == '' else '!!!ERROR!!!')
        else:
            set_.ability = unpack_name(ability, dex.abilities)
        i = j + 1
        j = buf.find('|', i)
        if j < 0:
            return None
        set_.moves = [unpack_name(n, dex.moves) for n in buf[i:j].split(',')[:24]]
        i = j + 1
        j = buf.find('|', i)
        if j < 0:
            return None
        set_.nature = unpack_name(buf[i:j], dex.natures)
        i = j + 1
        j = buf.find('|', i)
        if j < 0:
            return None
        if j != i:
            evs = buf[i:j].split(',')[:6]
            evs += [''] * (6 - len(evs))
            set_.evs = Obj({s: _num(evs[k]) for k, s in enumerate(STATS)})
        i = j + 1
        j = buf.find('|', i)
        if j < 0:
            return None
        if i != j:
            set_.gender = buf[i:j]
        i = j + 1
        j = buf.find('|', i)
        if j < 0:
            return None
        if j != i:
            ivs = buf[i:j].split(',')[:6]
            ivs += [''] * (6 - len(ivs))
            set_.ivs = Obj({s: (31 if ivs[k] == '' else _num(ivs[k])) for k, s in enumerate(STATS)})
        i = j + 1
        j = buf.find('|', i)
        if j < 0:
            return None
        if i != j:
            set_.shiny = True
        i = j + 1
        j = buf.find('|', i)
        if j < 0:
            return None
        if i != j:
            m = re.match(r'\s*[+-]?\d+', buf[i:j])
            set_.level = int(m.group(0)) if m else None
        i = j + 1
        j = buf.find(']', i)
        misc = None
        if j < 0:
            if i < len(buf):
                misc = buf[i:].split(',')[:6]
        else:
            if i != j:
                misc = buf[i:j].split(',')[:6]
        if misc:
            misc += [''] * (6 - len(misc))
            set_.happiness = _num(misc[0]) if misc[0] else 255
            set_.hpType = misc[1] or ''
            set_.pokeball = unpack_name(misc[2] or '', dex.items)
            set_.gigantamax = bool(misc[3])
            set_.dynamaxLevel = _num(misc[4]) if misc[4] else 10
            set_.teraType = misc[5] or None
        if j < 0:
            break
        i = j + 1
    return team


def _num(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return 0
    if v != v:
        return 0
    return int(v) if v.is_integer() else v


# ---------------------------------------------------------------------------
# Text format (Showdown "export" format, with Champions SPs)

def import_team(text: str) -> list[Obj]:
    """Parse a team in Showdown's text format.

    Stat investment may be given as ``SPs: 32 HP / 32 Atk / 2 Spe`` (Champions Stat
    Points) or ``EVs:`` (interpreted as SP values, like the Showdown champions mod).
    """
    dex = _dex()
    team = []
    cur = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cur = None
            continue
        if line.startswith('===') or line.startswith('//'):
            continue
        if cur is None:
            cur = Obj(name='', species='', item='', ability='', moves=[], nature='',
                      evs=Obj({s: 0 for s in STATS}), level=50)
            team.append(cur)
            item = ''
            if ' @ ' in line:
                line, item = line.rsplit(' @ ', 1)
            cur.item = item.strip()
            line = line.strip()
            gender = ''
            if line.endswith(' (M)') or line.endswith(' (F)'):
                gender = line[-2]
                line = line[:-4]
            m = re.match(r'^(.*?)\s*\(([^()]+)\)$', line)
            if m and dex.species.get(m.group(2)).exists:
                cur.name = m.group(1).strip()
                cur.species = dex.species.get(m.group(2)).name
            else:
                cur.species = dex.species.get(line).name if dex.species.get(line).exists else line
                cur.name = ''
            if gender:
                cur.gender = gender
            continue
        if line.startswith('Ability:'):
            cur.ability = line[len('Ability:'):].strip()
        elif line.startswith('Level:'):
            cur.level = int(line[len('Level:'):].strip())
        elif line.startswith('Shiny:'):
            cur.shiny = line[len('Shiny:'):].strip().lower() == 'yes'
        elif line.startswith('Tera Type:') or line.startswith('Happiness:'):
            continue
        elif line.startswith('EVs:') or line.startswith('SPs:') or line.startswith('SP:'):
            spec = line.split(':', 1)[1]
            for part in spec.split('/'):
                part = part.strip()
                if not part:
                    continue
                amount, stat = part.split(None, 1)
                cur.evs[STAT_NAMES.get(stat.strip(), to_id(stat))] = int(amount)
        elif line.startswith('IVs:'):
            continue
        elif line.endswith(' Nature'):
            cur.nature = line[:-len(' Nature')].strip()
        elif line.startswith('-'):
            move = line[1:].strip()
            if move.startswith('Hidden Power'):
                move = 'Hidden Power'
            cur.moves.append(move)
    return team


def export_team(team) -> str:
    out = []
    for s in team:
        name = s.get('name')
        species = s.get('species')
        head = f"{name} ({species})" if name and name != species else species
        if s.get('gender'):
            head += f" ({s['gender']})"
        if s.get('item'):
            head += f" @ {s['item']}"
        lines = [head]
        if s.get('ability'):
            lines.append(f"Ability: {s['ability']}")
        if s.get('level') and s['level'] != 50:
            lines.append(f"Level: {s['level']}")
        evs = s.get('evs') or {}
        sp = ' / '.join(f"{evs[k]} {STAT_LABELS[k]}" for k in STATS if evs.get(k))
        if sp:
            lines.append(f"SPs: {sp}")
        if s.get('nature'):
            lines.append(f"{s['nature']} Nature")
        for m in s.get('moves') or []:
            lines.append(f"- {m}")
        out.append('\n'.join(lines))
    return '\n\n'.join(out) + '\n'


def calc_stats(set_, species=None) -> dict:
    """Champions stats of a set (level 50, IV 31, SP stored in ``evs``)."""
    dex = _dex()
    species = species or dex.species.get(set_.get('species'))
    nature = dex.natures.get(set_.get('nature'))
    evs = set_.get('evs') or {}
    out = {}
    for s in STATS:
        base = species.baseStats[s]
        sp = evs.get(s) or 0
        if s == 'hp':
            out[s] = base + sp + 75
            continue
        v = base + sp + 20
        if nature.plus == s:
            v = (v * 110) // 100
        elif nature.minus == s:
            v = (v * 90) // 100
        out[s] = v
    return out
