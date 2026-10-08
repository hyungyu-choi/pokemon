"""Build ``pokechamp/data/names_ko.json``: Korean names for the battle assistant UI.

Source: the PokeAPI CSV dump (https://github.com/PokeAPI/pokeapi, ``data/v2/csv``), Korean = language 3.
Every Showdown entry of the Champions dex (species incl. formes and Megas, moves, items, abilities,
natures, types) is looked up and written keyed by its Showdown id (``pokechamp.sim.js.to_id`` of the
English name)::

    {"species": {"garchomp": "한카리아스", "rotomwash": "워시로토무", "charizardmegax": "메가리자몽X", ...},
     "moves": {...}, "items": {...}, "abilities": {...}, "natures": {...}, "types": {...},
     "_meta": {"source": ..., "constructed": {table: {id: ko}}}}

Formes: PokeAPI's form-specific Korean names are used when they are full names (``워시로토무``,
``메가리자몽X``); regional formes become ``<region> <species>`` (``알로라 나인테일``) and other formes
``<species> (<form>)`` (``비비용 (빙설의 모양)``, ``마휘핑 (밀키루비)``, ``냐오닉스 (암컷)``). Megas /
Mega Stones missing from PokeAPI are built from the base species (``메가`` + name + X/Y/Z, written like
PokeAPI's own ``메가리자몽X``; ``<species>나이트``). Those, the few hand-written form labels and the
unofficial translations of the newest Champions abilities are listed under ``_meta.constructed`` and in
the report.

Usage::

    python tools/build_korean_names.py                      # download the CSVs (temp dir), build, report
    python tools/build_korean_names.py --csv-dir DIR        # read CSVs from DIR (missing ones downloaded there)
    python tools/build_korean_names.py --csv-dir DIR --offline --strict
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pokechamp.sim.dex import get_dex  # noqa: E402
from pokechamp.sim.js import to_id  # noqa: E402

BASE_URL = 'https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/'
CSV_FILES = ('languages', 'pokemon_species', 'pokemon_species_names', 'pokemon', 'pokemon_forms',
             'pokemon_form_names', 'moves', 'move_names', 'items', 'item_names', 'abilities', 'ability_names',
             'natures', 'nature_names', 'types', 'type_names')
DEFAULT_OUT = os.path.join(ROOT, 'pokechamp', 'data', 'names_ko.json')
TABLES = ('species', 'moves', 'items', 'abilities', 'natures', 'types')

REGION_KO = {'alola': '알로라', 'galar': '가라르', 'hisui': '히스이', 'paldea': '팔데아'}
GENDER_KO = {'male': '수컷', 'female': '암컷'}
# Form labels PokeAPI has no Korean name for (keyed by the normalised Showdown forme, see _forme_tokens).
FORM_FALLBACK_KO = {
    ('female',): '암컷의 모습',                 # Basculegion-F
    ('male',): '수컷의 모습',
    ('antique',): '진작의 모습',                # Polteageist-Antique (진작 = 真作)
    ('phony',): '위작의 모습',
    ('four',): '네 식구',                       # Maushold-Four
    ('three',): '세 식구',
    ('blue',): '블루 페더',                     # Squawkabilly plumages
    ('yellow',): '옐로 페더',
    ('white',): '화이트 페더',
    ('green',): '그린 페더',
    ('combat',): '컴뱃종',                      # Tauros-Paldea breeds
    ('blaze',): '블레이즈종',
    ('aqua',): '워터종',
    ('hero',): '마이티폼',                      # Palafin-Hero (battle-only)
    ('zero',): '나이브폼',
}
# Abilities introduced by Pokemon Champions that PokeAPI has no Korean name for yet (unofficial translations).
ABILITY_FALLBACK_KO = {
    'auraguard': '아우라가드',     # Lucario-Mega-Z: contact moves deal 1/2 damage
    'eelevate': '부유상승',        # Eelektross-Mega: Ground immunity, +1 highest stat after a KO
    'firemane': '불꽃갈기',        # Pyroar-Mega: 1.5x attacking stat for Fire moves
}
# Corrections of PokeAPI entries that are evidently wrong.
OVERRIDES = {
    # PokeAPI: 플라베베나이트 (Flabébé); the stone is Floette's (JP フラエッテナイト), Floette = 플라엣테.
    'items': {'floettite': '플라엣테나이트'},
}
# Tokens of PokeAPI form identifiers that Showdown formes leave out.
DROP_TOKENS = {'strawberry', 'sweet', 'breed', 'plumage', 'family', 'of', 'mask', 'cap'}
# Official names that legitimately do not contain the species name (후딘 -> 후디나이트).
SUSPICIOUS_OK = {'alakazite'}

SPOT_CHECKS = [
    ('species', 'Garchomp', '한카리아스'), ('species', 'Dragonite', '망나뇽'), ('species', 'Incineroar', '어흥염'),
    ('species', 'Kingambit', '대도각참'), ('species', 'Gholdengo', '타부자고'),
    ('species', 'Rotom-Wash', '워시로토무'), ('species', 'Rotom-Heat', '히트로토무'), ('species', 'Rotom-Mow', '커트로토무'),
    ('species', 'Alcremie-Ruby-Cream', '마휘핑 (밀키루비)'),
    ('species', 'Charizard-Mega-Y', '메가리자몽Y'), ('species', 'Ninetales-Alola', '알로라 나인테일'),
    ('moves', 'Earthquake', '지진'), ('moves', 'Thunderbolt', '10만볼트'), ('moves', 'Protect', '방어'),
    ('moves', 'U-turn', '유턴'), ('moves', 'Sucker Punch', '기습'),
    ('items', 'Choice Scarf', '구애스카프'), ('items', 'Leftovers', '먹다남은음식'),
    ('items', 'Life Orb', '생명의구슬'), ('items', 'Focus Sash', '기합의띠'), ('items', 'Sitrus Berry', '자뭉열매'),
    ('items', 'Garchompite', '한카리아스나이트'),
    ('abilities', 'Rough Skin', '까칠한피부'), ('abilities', 'Intimidate', '위협'), ('abilities', 'Levitate', '부유'),
    ('natures', 'Jolly', '명랑'), ('natures', 'Adamant', '고집'), ('natures', 'Timid', '겁쟁이'),
    ('types', 'Dragon', '드래곤'), ('types', 'Fire', '불꽃'),
]


# ---------------------------------------------------------------------------
# CSV input

def fetch_csvs(target: str, offline: bool) -> str:
    """Make ``target`` hold every file of CSV_FILES (downloading the missing ones unless offline)."""
    os.makedirs(target, exist_ok=True)
    for name in CSV_FILES:
        path = os.path.join(target, name + '.csv')
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        if offline:
            raise SystemExit(f'missing {path} (and --offline given)')
        url = BASE_URL + name + '.csv'
        print(f'downloading {url}', file=sys.stderr)
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        with open(path + '.part', 'wb') as f:
            f.write(data)
        os.replace(path + '.part', path)
    return target


def read_csv(csv_dir: str, name: str) -> list[dict]:
    with open(os.path.join(csv_dir, name + '.csv'), encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


class PokeApi:
    def __init__(self, csv_dir: str):
        langs = {r['identifier']: r['id'] for r in read_csv(csv_dir, 'languages')}
        self.ko, self.en = langs.get('ko', '3'), langs.get('en', '9')
        # species: national dex number -> Korean name
        self.species_ko = {}
        for r in read_csv(csv_dir, 'pokemon_species_names'):
            if r['local_language_id'] == self.ko and r['name']:
                self.species_ko[int(r['pokemon_species_id'])] = r['name'].strip()
        pokemon_species = {r['id']: int(r['species_id']) for r in read_csv(csv_dir, 'pokemon')}
        form_names = {}
        for r in read_csv(csv_dir, 'pokemon_form_names'):
            if r['local_language_id'] == self.ko:
                form_names[r['pokemon_form_id']] = ((r['form_name'] or '').strip(), (r['pokemon_name'] or '').strip())
        # forms grouped by species number
        self.forms: dict[int, list[dict]] = {}
        for r in read_csv(csv_dir, 'pokemon_forms'):
            num = pokemon_species.get(r['pokemon_id'])
            if num is None:
                continue
            form_name, pokemon_name = form_names.get(r['id'], ('', ''))
            self.forms.setdefault(num, []).append({
                'id': r['id'], 'identifier': r['identifier'], 'form': r['form_identifier'] or '',
                'is_mega': r['is_mega'] == '1', 'form_name': form_name, 'pokemon_name': pokemon_name})
        self.moves = self._named(csv_dir, 'moves', 'move_names', 'move_id')
        self.items = self._named(csv_dir, 'items', 'item_names', 'item_id')
        self.abilities = self._named(csv_dir, 'abilities', 'ability_names', 'ability_id')
        self.natures = self._named(csv_dir, 'natures', 'nature_names', 'nature_id')
        self.types = self._named(csv_dir, 'types', 'type_names', 'type_id')

    def _named(self, csv_dir, entity_file, names_file, id_col):
        """{'by_en': {id(English name): ko}, 'by_ident': {id(identifier): ko}, 'by_num': {num: ko}}."""
        ident = {r['id']: r['identifier'] for r in read_csv(csv_dir, entity_file)}
        ko, en = {}, {}
        for r in read_csv(csv_dir, names_file):
            if r['local_language_id'] == self.ko and r['name']:
                ko[r[id_col]] = r['name'].strip()
            elif r['local_language_id'] == self.en and r['name']:
                en[r[id_col]] = r['name'].strip()
        table = {'by_en': {}, 'by_ident': {}, 'by_num': {}}
        for pid in sorted(ko, key=int):  # lowest PokeAPI id wins on duplicate names (main-series first)
            name = ko[pid]
            table['by_num'].setdefault(int(pid), name)
            if pid in en:
                table['by_en'].setdefault(to_id(en[pid]), name)
            if pid in ident:
                table['by_ident'].setdefault(to_id(ident[pid]), name)
        return table


# ---------------------------------------------------------------------------
# lookups

def _lookup(table: dict, name: str, num=None) -> str:
    sid = to_id(name)
    return table['by_en'].get(sid) or table['by_ident'].get(sid) or (
        table['by_num'].get(num, '') if num else '')


def _forme_tokens(text: str) -> tuple:
    out = []
    for tok in text.lower().replace(' ', '-').replace('_', '-').split('-'):
        tok = to_id(tok)
        if not tok or tok in DROP_TOKENS:
            continue
        out.append({'f': 'female', 'm': 'male'}.get(tok, tok))
    return tuple(out)


def _find_form(api: PokeApi, num: int, forme: str):
    want = _forme_tokens(forme)
    want_sorted, want_joined = tuple(sorted(want)), ''.join(want)
    for f in api.forms.get(num, []):
        have = _forme_tokens(f['form'])
        if tuple(sorted(have)) == want_sorted or ''.join(have) == want_joined:
            return f
    return None


def _short_form(label: str) -> str:
    for suffix in ('의 모습',):
        if label.endswith(suffix) and len(label) > len(suffix):
            return label[:-len(suffix)]
    return label


def species_name(api: PokeApi, sp) -> tuple[str, str]:
    """(Korean name, how) for a Showdown species; how in 'pokeapi', 'pokeapi-form', 'constructed', 'missing'."""
    base = api.species_ko.get(sp.num) if sp.num and sp.num > 0 else None
    if not base:
        return '', 'missing'
    forme = sp.forme or ''
    if not forme or sp.name == sp.baseSpecies:
        return base, 'pokeapi'
    tokens = _forme_tokens(forme)
    form = _find_form(api, sp.num, forme)
    form_name = form['form_name'] if form else ''
    pokemon_name = form['pokemon_name'] if form else ''
    if 'mega' in tokens:
        if pokemon_name:
            return pokemon_name, 'pokeapi'
        if form_name and base in form_name:
            return form_name, 'pokeapi'
        suffix = ''.join(t.upper() for t in tokens if t in ('x', 'y', 'z'))
        gender = [GENDER_KO[t] for t in tokens if t in GENDER_KO]
        return f'메가{base}{suffix}' + (f' ({gender[0]})' if gender else ''), 'constructed'
    if 'gmax' in tokens:
        return f'거다이맥스 {base}', 'constructed'
    if pokemon_name:
        return pokemon_name, 'pokeapi'
    if form_name and base in form_name:  # full names such as 워시로토무
        return form_name, 'pokeapi'
    if tokens and tokens[0] in REGION_KO:
        name = f'{REGION_KO[tokens[0]]} {base}'
        rest = tokens[1:]
        if not rest:
            return name, 'pokeapi-form'
        label = FORM_FALLBACK_KO.get(rest)
        if label:
            return f'{name} ({_short_form(label)})', 'constructed'
        return '', 'missing'
    if form_name:
        return f'{base} ({_short_form(form_name)})', 'pokeapi-form'
    label = FORM_FALLBACK_KO.get(tokens)
    if label:
        return f'{base} ({_short_form(label)})', 'constructed'
    return '', 'missing'


def item_name(api: PokeApi, dex, it) -> tuple[str, str]:
    override = OVERRIDES['items'].get(it.id)
    if override:
        return override, 'override'
    name = _lookup(api.items, it.name)
    if name:
        return name, 'pokeapi'
    if it.megaStone:  # new Mega Stone: <base species>나이트 (+ X/Y/Z)
        base_species, forme = next(iter(dict(it.megaStone).items()))
        base = dex.species.get(dex.species.get(base_species).baseSpecies)
        base_ko = api.species_ko.get(base.num, '')
        if base_ko:
            suffix = ''.join(t.upper() for t in _forme_tokens(dex.species.get(forme).forme) if t in ('x', 'y', 'z'))
            return f'{base_ko}나이트{suffix}', 'constructed'
    return '', 'missing'


# ---------------------------------------------------------------------------

def build(api: PokeApi):
    dex = get_dex()
    out = {t: {} for t in TABLES}
    how = {t: {} for t in TABLES}

    def put(table, sid, result):
        name, kind = result
        if name:
            out[table][sid] = name
        how[table][sid] = kind

    for sp in dex.species.all():
        put('species', sp.id, species_name(api, sp))
    for m in dex.moves.all():
        name = _lookup(api.moves, m.name, m.num if (m.num or 0) > 0 else None)
        put('moves', m.id, (name, 'pokeapi' if name else 'missing'))
    for a in dex.abilities.all():
        name = _lookup(api.abilities, a.name, a.num if (a.num or 0) > 0 else None)
        if name:
            put('abilities', a.id, (name, 'pokeapi'))
        elif a.id in ABILITY_FALLBACK_KO:
            put('abilities', a.id, (ABILITY_FALLBACK_KO[a.id], 'constructed'))
        else:
            put('abilities', a.id, ('', 'missing'))
    for it in dex.items.all():
        put('items', it.id, item_name(api, dex, it))
    for n in dex.natures.all():
        name = _lookup(api.natures, n.name)
        put('natures', n.id, (name, 'pokeapi' if name else 'missing'))
    for t in dex.types.names():
        name = _lookup(api.types, t)
        put('types', to_id(t), (name, 'pokeapi' if name else 'missing'))
    return out, how


def suspicious(dex, names: dict) -> list[str]:
    """Mega formes / Mega Stones whose Korean name does not contain the base species' Korean name."""
    warn = []
    for it in dex.items.all():
        if not it.megaStone:
            continue
        for base_species, forme in dict(it.megaStone).items():
            base_ko = names['species'].get(to_id(dex.species.get(base_species).baseSpecies), '')
            stone_ko = names['items'].get(it.id, '')
            forme_ko = names['species'].get(to_id(forme), '')
            if base_ko and stone_ko and base_ko not in stone_ko and it.id not in SUSPICIOUS_OK:
                warn.append(f'items/{it.id}: {stone_ko!r} does not contain {base_ko!r}')
            if base_ko and forme_ko and base_ko not in forme_ko:
                warn.append(f'species/{to_id(forme)}: {forme_ko!r} does not contain {base_ko!r}')
    return sorted(set(warn))


def ui_names() -> dict:
    """The names the battle assistant UI shows (GET /api/data)."""
    from pokechamp.ui.service import AssistantService
    d = AssistantService().game_data()
    species = [s['name'] for s in d['species']]
    species += [m['forme'] for s in d['species'] for m in s['megas']]
    return {'species': species, 'moves': list(d['moves']), 'items': [i['name'] for i in d['items']],
            'abilities': list(d['abilities']), 'natures': [n['name'] for n in d['natures']],
            'types': list(d['types'])}


def report(names: dict, how: dict) -> int:
    dex = get_dex()
    ui = ui_names()
    missing_total = 0
    print('\nCoverage of the names shown by the UI (GET /api/data):')
    for table in TABLES:
        wanted = list(dict.fromkeys(ui[table]))
        missing = [n for n in wanted if not names[table].get(to_id(n))]
        constructed = [n for n in wanted if how[table].get(to_id(n)) in ('constructed', 'override')]
        missing_total += len(missing)
        have = len(wanted) - len(missing)
        print(f'  {table:<10} {have:>4}/{len(wanted):<4} ({100.0 * have / max(1, len(wanted)):5.1f}%)'
              f'  constructed/override: {len(constructed)}')
        if missing:
            print(f'    missing: {", ".join(missing)}')
    print('\nWhole Champions dex:')
    for table in TABLES:
        total = len(how[table])
        print(f'  {table:<10} {len(names[table]):>4}/{total}')
        miss = sorted(k for k, v in how[table].items() if v == 'missing')
        if miss:
            print(f'    missing (not shown by the UI unless listed above): {", ".join(miss)}')
    print('\nConstructed / overridden names (not straight from PokeAPI):')
    for table in TABLES:
        for sid, kind in sorted(how[table].items()):
            if kind in ('constructed', 'override'):
                print(f'  {table}/{sid}: {names[table][sid]}  [{kind}]')
    warn = suspicious(dex, names)
    if warn:
        print('\nSuspicious (Korean name does not contain the base species name):')
        for w in warn:
            print('  ' + w)
    print('\nSpot checks:')
    failed = 0
    for table, name, expected in SPOT_CHECKS:
        got = names[table].get(to_id(name), '')
        ok = got == expected
        failed += not ok
        print(f'  [{"ok" if ok else "FAIL"}] {table}/{name}: {got!r}' + ('' if ok else f' (expected {expected!r})'))
    print(f'\n{len(SPOT_CHECKS) - failed}/{len(SPOT_CHECKS)} spot checks passed; {missing_total} UI names missing.')
    return missing_total + failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--csv-dir', help='directory with the PokeAPI CSVs (missing files are downloaded into it)')
    ap.add_argument('--offline', action='store_true', help='never download; fail if a CSV is missing')
    ap.add_argument('--out', default=DEFAULT_OUT, help=f'output JSON (default: {DEFAULT_OUT})')
    ap.add_argument('--no-report', action='store_true', help='skip the coverage report')
    ap.add_argument('--strict', action='store_true',
                    help='exit with status 1 if a UI name is missing or a spot check fails')
    args = ap.parse_args(argv)

    if args.csv_dir:
        api = PokeApi(fetch_csvs(args.csv_dir, args.offline))
    elif args.offline:
        raise SystemExit('--offline needs --csv-dir')
    else:
        with tempfile.TemporaryDirectory(prefix='pokeapi_csv_') as tmp:
            api = PokeApi(fetch_csvs(tmp, False))
    names, how = build(api)
    constructed = {t: {sid: names[t][sid] for sid, kind in sorted(how[t].items())
                       if kind in ('constructed', 'override')} for t in TABLES}
    doc = {t: dict(sorted(names[t].items())) for t in TABLES}
    doc['_meta'] = {'source': BASE_URL + ' (PokeAPI, local_language_id 3 = Korean)',
                    'generator': 'tools/build_korean_names.py', 'keys': 'Showdown ids (pokechamp.sim.js.to_id)',
                    'constructed': {t: v for t, v in constructed.items() if v}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write('\n')
    os.replace(tmp, args.out)
    print(f'wrote {args.out}: ' + ', '.join(f'{t} {len(names[t])}' for t in TABLES))
    if args.no_report:
        return 0
    problems = report(names, how)
    return 1 if (args.strict and problems) else 0


if __name__ == '__main__':
    sys.exit(main())
