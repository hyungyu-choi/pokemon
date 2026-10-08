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


def test_http_rejects_malformed_bodies(http):
    for path, body in (('/api/team/parse', b'[1,2]'), ('/api/team/parse', b'{"sets":[1]}'),
                       ('/api/preview', json.dumps({'my_team': [1, 2, 3, 4, 5, 6], 'foe': ['Garchomp'] * 6}).encode()),
                       ('/api/advise', b'"text"'), ('/api/advise', json.dumps({'state': [1]}).encode()),
                       ('/api/advise', json.dumps({'state': {}, 'samples': 1000}).encode()),
                       ('/api/advise', json.dumps({'state': {}, 'depth': 0}).encode()),
                       ('/api/advise_switch', json.dumps({'state': {}, 'depth': 4}).encode()),
                       ('/api/advise_switch', json.dumps({'state': {}, 'depth': 'deep'}).encode()),
                       ('/api/preview', json.dumps({'my_team': [], 'foe': [], 'sims': 'x'}).encode())):
        code, out = _post(http + path, body)
        assert code == 400 and 'error' in out, (path, body, code, out)


def test_http_head_and_bad_content_length(http):
    import socket
    req = urllib.request.Request(http + '/', method='HEAD')
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 200 and r.read() == b''
    host, port = http[len('http://'):].split(':')
    with socket.create_connection((host, int(port)), timeout=5) as s:
        s.sendall(b'POST /api/advise HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: -1\r\n\r\n')
        data = s.recv(4096)
    assert data.startswith(b'HTTP/1.0 400')  # an empty body: "state" is missing, no hang


def test_advise_rejects_unknown_or_impossible_names(service, state):
    for mutate, needle in (
            (lambda st: st['foe']['pokemon']['Dragonite'].update(moves=['Not A Move']), 'unknown move'),
            (lambda st: st['foe']['pokemon']['Dragonite'].update(item='Not An Item'), 'unknown item'),
            (lambda st: st['foe']['pokemon']['Dragonite'].update(ability='Nope'), 'unknown ability'),
            (lambda st: st['foe']['pokemon']['Dragonite'].update(moves=['Shadow Ball']), 'cannot learn'),
            (lambda st: st['me']['pokemon']['Garchomp'].update(locked_move='Body Press'), 'cannot be locked')):
        st = json.loads(json.dumps(state))
        mutate(st)
        with pytest.raises(ValueError, match=needle):
            service.advise(st, samples=1, depth=1)


def test_preview_rejects_duplicate_foe_species(service, state):
    sets = service.parse_team(state['me']['team'])['sets']
    with pytest.raises(ValueError, match='same species'):
        service.team_preview(sets, ['Garchomp'] * 2 + state['foe']['team'][:4], sims=1, top=1)


def test_recommended_teams_are_distinct(service):
    rec = service.recommended_teams()
    keys = [tuple(sorted((s['species'], s['item']) for s in t['sets'])) for t in rec['teams']]
    assert len(keys) == len(set(keys))


def test_cancel_stops_advise_switch(service, state):
    st = json.loads(json.dumps(state))
    st['me']['pokemon']['Garchomp'] = {'fainted': True}
    calls = []
    real = service._advise

    def cancelling(*a, **k):
        res = real(*a, **k)
        calls.append(res)
        service.cancel()  # as if the user pressed 취소 during the first candidate
        return res

    service._advise = cancelling
    try:
        res = service.advise_switch(st, samples=1, depth=1)
    finally:
        service._advise = real
    assert res['cancelled'] is True
    assert len(calls) == 2 and calls[1]['cancelled'] is True  # the 2nd candidate did not run
    assert len(res['options']) == 1  # only the fully evaluated candidate is listed


def test_http_advise_switch_passes_samples_and_depth(http, service, monkeypatch):
    seen = {}

    def fake(state, samples=8, depth=2):
        seen.update(samples=samples, depth=depth)
        return {'options': [], 'elapsed': 0.0, 'cancelled': False}

    monkeypatch.setattr(service, 'advise_switch', fake)
    code, out = _post(http + '/api/advise_switch', json.dumps({'state': {}, 'samples': 6, 'depth': 1}).encode())
    assert code == 200 and seen == {'samples': 6, 'depth': 1}, (code, out)
    code, out = _post(http + '/api/advise_switch', json.dumps({'state': {}}).encode())
    assert code == 200 and seen == {'samples': 8, 'depth': 2}, (code, out)
