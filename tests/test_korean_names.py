"""Korean names (pokechamp/data/names_ko.json, built by tools/build_korean_names.py) cover the UI's data."""
import json
import os

import pytest

from pokechamp.sim.js import to_id
from pokechamp.ui.service import DATA_DIR, AssistantService

NAMES_KO = os.path.join(DATA_DIR, 'names_ko.json')
pytestmark = pytest.mark.skipif(not os.path.exists(NAMES_KO), reason='names_ko.json not built')


@pytest.fixture(scope='module')
def data():
    return AssistantService().game_data()


def test_status_reports_korean_names():
    assert AssistantService().status()['korean_names'] is True


def test_every_ui_name_has_korean(data):
    assert [s['name'] for s in data['species'] if not s['ko']] == []
    assert [m['forme'] for s in data['species'] for m in s['megas'] if not m['forme_ko']] == []
    assert [m['stone'] for s in data['species'] for m in s['megas'] if not m['stone_ko']] == []
    assert [n for n, m in data['moves'].items() if not m['ko']] == []
    assert [i['name'] for i in data['items'] if not i['ko']] == []
    assert [a for a, ko in data['abilities'].items() if not ko] == []
    assert [n['name'] for n in data['natures'] if not n['ko']] == []
    assert [t for t, ko in data['types'].items() if not ko] == []


def test_spot_checks(data):
    species = {s['name']: s for s in data['species']}
    assert species['Garchomp']['ko'] == '한카리아스'
    assert species['Rotom-Wash']['ko'] == '워시로토무'
    assert species['Ninetales-Alola']['ko'] == '알로라 나인테일'
    megas = {m['forme']: m for m in species['Charizard']['megas']}
    assert megas['Charizard-Mega-X']['forme_ko'] == '메가리자몽X'
    assert megas['Charizard-Mega-X']['stone_ko'] == '리자몽나이트X'
    assert data['moves']['Earthquake']['ko'] == '지진'
    assert {i['name']: i['ko'] for i in data['items']}['Choice Scarf'] == '구애스카프'
    assert data['abilities']['Rough Skin'] == '까칠한피부'
    assert {n['name']: n['ko'] for n in data['natures']}['Jolly'] == '명랑'
    assert data['types']['Dragon'] == '드래곤'


def test_keys_are_showdown_ids():
    with open(NAMES_KO, encoding='utf-8') as f:
        names = json.load(f)
    for table in ('species', 'moves', 'items', 'abilities', 'natures', 'types'):
        assert names[table], table
        assert all(k == to_id(k) for k in names[table]), table
