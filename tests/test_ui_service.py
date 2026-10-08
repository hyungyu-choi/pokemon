"""Tests for the battle assistant UI backend (service functions and the HTTP layer)."""
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from pokechamp.teambuilder import TeamValidator
from pokechamp.ui.server import make_handler
from pokechamp.ui.service import AssistantService, _obj_team

ROOT = os.path.dirname(os.path.dirname(__file__))


@pytest.fixture(scope='module')
def service():
    return AssistantService(seed=1)


@pytest.fixture(scope='module')
def state():
    with open(os.path.join(ROOT, 'examples', 'advisor_state.json'), encoding='utf-8') as f:
        return json.load(f)


def test_game_data(service):
    d = service.game_data()
    assert d['picked'] == 3
    assert len(d['species']) > 250
    names = {s['name'] for s in d['species']}
    assert 'Garchomp' in names
    for s in d['species'][:50]:
        assert s['abilities'] and s['moves']
        assert all(m in d['moves'] for m in s['moves'])
    assert any(i['megaStone'] for i in d['items'])
    assert {x['id'] for x in d['statuses']} >= {'', 'brn', 'par', 'slp', 'frz', 'psn', 'tox'}


def test_recommended_teams_are_legal(service):
    rec = service.recommended_teams()
    if not rec['teams']:
        pytest.skip('no team library')
    val = TeamValidator()
    for t in rec['teams']:
        assert len(t['sets']) == 6
        assert val.validate_team(_obj_team(t['sets'])) == []
        assert 'Ability:' in t['text']


def test_parse_team(service, state):
    out = service.parse_team(state['me']['team'])
    assert out['problems'] == []
    assert [s['species'] for s in out['sets']][0] == 'Garchomp'
    assert out['sets'][0]['stats']['hp'] == 185
    bad = service.parse_team('Garchomp @ Choice Scarf\nAbility: Rough Skin\n- Spore\n')
    assert bad['problems']


def test_advise_respects_choice_lock(service, state):
    res = service.advise(state, samples=2, depth=1)
    labels = [r['label'] for r in res['recommendations']]
    assert labels
    for r in res['recommendations']:
        assert 0.0 <= r['win_rate'] <= 1.0
        kinds = [m[0] for m in r['meaning'] if m]
        if 'move' in kinds:
            # Garchomp is locked into Earthquake by its Choice Scarf
            assert ['move', 'Earthquake'] in r['meaning']


def test_advise_switch_after_faint(service, state):
    st = json.loads(json.dumps(state))
    st['me']['pokemon']['Garchomp'] = {'fainted': True}
    res = service.advise_switch(st, samples=2, depth=1)
    names = {o['switch_to'] for o in res['options']}
    assert names and names <= {'Rotom-Wash', 'Kingambit'}


def test_team_preview(service, state):
    sets = service.parse_team(state['me']['team'])['sets']
    res = service.team_preview(sets, state['foe']['team'], sims=1, top=2)
    assert len(res['options']) == 2
    for o in res['options']:
        assert len(o['order']) == 3 and len(set(o['order'])) == 3
        assert o['lead'] == o['names'][0]


@pytest.fixture(scope='module')
def http(service):
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(service))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f'http://127.0.0.1:{httpd.server_address[1]}'
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(url, data: bytes):
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_routes(http):
    code, body = _get(http + '/api/status')
    assert code == 200 and json.loads(body)['format'] == 'gen9championsbssregmc'
    code, body = _get(http + '/')
    assert code == 200 and b'<' in body
    assert _get(http + '/nope')[0] == 404
    assert _get(http + '/static/../server.py')[0] == 404
    code, out = _post(http + '/api/advise', b'{not json')
    assert code == 400 and 'error' in out
    code, out = _post(http + '/api/advise', json.dumps({'state': {'me': {}}}).encode())
    assert code == 400 and 'error' in out
    code, out = _post(http + '/api/team/parse', json.dumps({'text': 'Garchomp\nAbility: Rough Skin\n- Earthquake\n'}).encode())
    assert code == 200 and out['sets'][0]['species'] == 'Garchomp'
