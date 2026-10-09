"""Backend logic of the battle assistant UI (independent of HTTP so it can be tested directly).

Everything the browser needs goes through :class:`AssistantService`:

* :meth:`game_data` - legal species / moves / items / abilities / natures (+ Korean names) for the dropdowns
* :meth:`recommended_teams` - the teams the team-building AI recommends
* :meth:`parse_team` - Showdown text -> sets (+ legality problems and computed stats)
* :meth:`team_preview` - which 3 Pokemon to bring and which one to lead against the opponent's six
* :meth:`advise` - ranked actions (moves / switches) with estimated win rates for the current turn
* :meth:`advise_switch` - which Pokemon to send in after one of ours fainted
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import threading
import time
from functools import lru_cache

from ..sim.dex import get_dex
from ..sim.js import Obj, to_id
from ..sim.teams import calc_stats, export_team, import_team

FORMAT = 'gen9championsbssregmc'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MODEL_PATH = os.path.join(ROOT, 'models', 'battle_singles.pt')
LIBRARY_PATH = os.path.join(ROOT, 'models', 'teams_singles.json')
STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')

STATUS_KO = {'': '없음', 'brn': '화상', 'par': '마비', 'slp': '잠듦', 'frz': '얼음', 'psn': '독', 'tox': '맹독'}
WEATHER_KO = {'': '없음', 'sunnyday': '쾌청', 'raindance': '비', 'sandstorm': '모래바람', 'snowscape': '설경'}
TERRAIN_KO = {'': '없음', 'electricterrain': '일렉트릭필드', 'grassyterrain': '그래스필드',
              'mistyterrain': '미스트필드', 'psychicterrain': '사이코필드'}
PSEUDO_KO = {'trickroom': '트릭룸', 'gravity': '중력', 'magicroom': '매직룸', 'wonderroom': '원더룸'}
SIDE_KO = {'reflect': '리플렉터', 'lightscreen': '빛의장막', 'auroraveil': '오로라베일', 'tailwind': '순풍',
           'safeguard': '신비의부적', 'stealthrock': '스텔스록', 'spikes': '압정뿌리기',
           'toxicspikes': '독압정', 'stickyweb': '끈적끈적네트'}
VOLATILE_KO = {'substitute': '대타출동', 'confusion': '혼란', 'leechseed': '씨뿌리기', 'taunt': '도발',
               'encore': '앵콜', 'yawn': '하품', 'focusenergy': '기충전', 'aquaring': '아쿠아링',
               'ingrain': '뿌리박기', 'magnetrise': '전자부유', 'perishsong': '멸망의노래', 'curse': '저주',
               'saltcure': '소금절이', 'healblock': '회복봉인'}
BOOST_KO = {'atk': '공격', 'def': '방어', 'spa': '특공', 'spd': '특방', 'spe': '스피드', 'accuracy': '명중',
            'evasion': '회피'}


def _load_json(path):
    with open(path, encoding='utf-8-sig') as f:  # -sig: also user files saved with a BOM (Windows Notepad)
        return json.load(f)


def default_model_file(name: str) -> str | None:
    """``models/<name>`` of the downloaded project folder (next to the ``pokechamp`` package), or of the current
    directory (when the package was installed with ``pip install .`` and is run from the project folder)."""
    for folder in (os.path.join(ROOT, 'models'), os.path.join(os.getcwd(), 'models')):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None


@lru_cache(maxsize=1)
def korean_names() -> dict:
    """``pokechamp/data/names_ko.json`` (built by ``tools/build_korean_names.py``), or empty tables."""
    path = os.path.join(DATA_DIR, 'names_ko.json')
    empty = {'species': {}, 'moves': {}, 'items': {}, 'abilities': {}, 'natures': {}, 'types': {}}
    if not os.path.exists(path):
        return empty
    data = _load_json(path)
    for k in empty:
        data.setdefault(k, {})
    return data


def _ko(table: str, name: str) -> str:
    return korean_names()[table].get(to_id(name), '')


def _plain_set(s) -> dict:
    evs = s.get('evs') or {}
    return {'species': s.get('species'), 'name': s.get('name') or s.get('species'), 'item': s.get('item') or '',
            'ability': s.get('ability') or '', 'moves': list(s.get('moves') or []), 'nature': s.get('nature') or '',
            'evs': {k: int(evs.get(k) or 0) for k in STATS}, 'level': 50}


def _obj_team(sets):
    return [Obj({**_plain_set(s), 'evs': Obj(_plain_set(s)['evs'])}) for s in sets]


def _need(cond, msg: str):
    """Reject malformed request data with a ValueError (-> HTTP 400 with this message)."""
    if not cond:
        raise ValueError(msg)


def _check_sets(sets, what: str):
    _need(isinstance(sets, list), f'{what} must be a list of sets')
    for i, s in enumerate(sets):
        _need(isinstance(s, dict), f'{what}[{i}] must be an object (a set)')
        _need(isinstance(s.get('species'), str) and s['species'], f'{what}[{i}].species is missing')
        _need(isinstance(s.get('moves') or [], list) and all(isinstance(m, str) for m in s.get('moves') or []),
              f'{what}[{i}].moves must be a list of move names')
        _need(isinstance(s.get('evs') or {}, dict), f'{what}[{i}].evs must be an object')
        for k in ('item', 'ability', 'nature', 'name'):
            _need(s.get(k) is None or isinstance(s.get(k), str), f'{what}[{i}].{k} must be a string')


_ITEM_NONE = ('', 'none', 'consumed', 'knocked off')


def _check_state_names(state):
    """Shape of the <state> and every move / item / ability name in it (unknown names -> ValueError)."""
    dex = get_dex()
    _need(isinstance(state, dict), '"state" must be a JSON object')
    for side in ('me', 'foe'):
        d = state.get(side)
        _need(isinstance(d, dict), f'state.{side} must be an object')
        for k in ('active', 'brought'):
            _need(d.get(k) is None or (isinstance(d[k], list) and all(isinstance(x, str) for x in d[k])),
                  f'state.{side}.{k} must be a list of species names')
        _need(isinstance(d.get('side') or {}, dict), f'state.{side}.side must be an object')
        mons = d.get('pokemon') or {}
        _need(isinstance(mons, dict), f'state.{side}.pokemon must be an object')
        for name, md in mons.items():
            where = f'state.{side}.pokemon["{name}"]'
            _need(isinstance(md, dict), f'{where} must be an object')
            it = md.get('item')
            if it is not None and it not in _ITEM_NONE:
                _need(isinstance(it, str) and dex.items.get(it).exists, f'unknown item {it!r} ({where}.item)')
            ab = md.get('ability')
            if ab:
                _need(isinstance(ab, str) and dex.abilities.get(ab).exists, f'unknown ability {ab!r} ({where}.ability)')
            moves = md.get('moves') or []
            _need(isinstance(moves, list), f'{where}.moves must be a list of move names')
            for m in moves + ([md['locked_move']] if md.get('locked_move') else []):
                _need(isinstance(m, str) and dex.moves.get(m).exists, f'unknown move {m!r} ({where})')
            for k in ('volatiles', 'faster_than', 'slower_than'):
                _need(isinstance(md.get(k) or [], list), f'{where}.{k} must be a list')
            _need(isinstance(md.get('boosts') or {}, dict), f'{where}.boosts must be an object')
    me = state['me']
    _need(isinstance(me.get('team'), (list, str)), 'state.me.team must be a list of sets or a Showdown export')
    if isinstance(me['team'], list):
        _check_sets(me['team'], 'state.me.team')
    foe_team = state['foe'].get('team')
    _need(isinstance(foe_team, list) and all(isinstance(x, str) for x in foe_team),
          'state.foe.team must be a list of species names')
    _need(isinstance(state.get('field') or {}, dict), 'state.field must be an object')


def _check_parsed(inp, formatid: str):
    """Moves that the described Pokemon cannot have (Choice lock outside the set, unlearnable revealed moves)."""
    from ..teambuilder.validator import format_legality
    leg = format_legality(formatid)
    by_species = {s.species: s for s in inp.me.team}
    for name, ms in inp.me.mons.items():
        s = by_species.get(name)
        if ms.locked_move and s is not None:
            _need(to_id(ms.locked_move) in {to_id(m) for m in s.moves},
                  f'{name} cannot be locked into {ms.locked_move}: it is not one of its moves')
    for name, ms in inp.foe.mons.items():
        legal = leg.species.get(name)
        if not legal:
            continue
        learnable = {to_id(m) for m in legal['moves']}
        for m in ms.moves + ([ms.locked_move] if ms.locked_move else []):
            _need(to_id(m) in learnable, f'the opponent\'s {name} cannot learn {m} in this format')


class AssistantService:
    def __init__(self, formatid: str = FORMAT, model_path: str | None = None, library_path: str | None = None,
                 seed: int = 0):
        self.formatid = formatid
        # model_path '' = no neural network (heuristic AI); None = the default model if it is there
        self.model_path = model_path if model_path is not None else default_model_file('battle_singles.pt')
        self.library_path = library_path if library_path is not None else default_model_file('teams_singles.json')
        self.model_error = None  # why the model could not be loaded (the heuristic AI is used then)
        self.seed = seed
        self._advisor = None
        self._advisor_lock = threading.Lock()
        self._lock = threading.Lock()
        self._data = None
        # every AI request gets a generation number; a newer request (or /api/cancel) makes the running
        # one stop at its next sample and return what it has so far
        self._gen = 0
        self._gen_lock = threading.Lock()
        self.torch_threads = 2  # leave CPU for the browser / other processes

    # ------------------------------------------------------------------
    @property
    def advisor(self):
        with self._advisor_lock:
            if self._advisor is None:
                from ..ai.advisor import Advisor
                try:
                    self._advisor = Advisor(model_path=self.model_path, library=self.library_path, seed=self.seed,
                                            threads=self.torch_threads)
                except Exception as exc:  # torch missing / DLL load failure / damaged or incompatible checkpoint
                    if not self.model_path:
                        raise
                    self.model_error = f'{type(exc).__name__}: {exc}'
                    print(f'[주의] AI 모델({os.path.basename(self.model_path)})을 불러오지 못해 휴리스틱 AI로 '
                          f'계산합니다: {self.model_error}', file=sys.stderr, flush=True)
                    self.model_path = None
                    self._advisor = Advisor(model_path=None, library=self.library_path, seed=self.seed)
            return self._advisor

    def warmup(self):
        try:
            self.game_data()
            _ = self.advisor  # loads the network (or falls back to the heuristic AI)
            from ..ai.inference import _hypotheses
            _hypotheses()
        except Exception:  # warm-up is best effort
            pass

    def _new_generation(self) -> int:
        with self._gen_lock:
            self._gen += 1
            return self._gen

    def cancel(self) -> dict:
        """Stop the running AI calculation (it returns its partial result)."""
        return {'cancelled_generation': self._new_generation() - 1}

    def status(self) -> dict:
        return {'format': self.formatid, 'model': os.path.basename(self.model_path) if self.model_path else None,
                'model_error': self.model_error,
                'library': os.path.basename(self.library_path) if self.library_path else None,
                'korean_names': bool(korean_names()['species'])}

    # ------------------------------------------------------------------
    def game_data(self) -> dict:
        """Static data for the selection menus (cached)."""
        if self._data is not None:
            return self._data
        from ..teambuilder.validator import format_legality
        dex = get_dex()
        leg = format_legality(self.formatid)
        species = []
        used_moves, used_abilities = set(), set()
        for name in sorted(leg.species):
            sp = dex.species.get(name)
            megas = []
            for stone in leg.items:
                it = dex.items.get(stone)
                if it.megaStone and name in it.megaStone:
                    forme = it.megaStone[name]
                    msp = dex.species.get(forme)
                    megas.append({'stone': it.name, 'stone_ko': _ko('items', it.name), 'forme': forme,
                                  'forme_ko': _ko('species', forme), 'types': list(msp.types),
                                  'ability': msp.abilities.get('0', ''),
                                  'baseStats': {k: msp.baseStats[k] for k in STATS}})
            entry = {
                'name': name, 'id': sp.id, 'ko': _ko('species', name), 'num': sp.num, 'types': list(sp.types),
                'baseStats': {k: sp.baseStats[k] for k in STATS},
                'abilities': list(leg.species[name]['abilities']),
                'moves': sorted(leg.species[name]['moves']),
                'megas': megas, 'weightkg': sp.weightkg,
            }
            species.append(entry)
            used_moves.update(entry['moves'])
            used_abilities.update(entry['abilities'])
            for m in megas:
                if m['ability']:
                    used_abilities.add(m['ability'])
        moves = {}
        for name in sorted(used_moves):
            m = dex.moves.get(name)
            moves[name] = {'ko': _ko('moves', name), 'type': m.type, 'category': m.category,
                           'basePower': m.basePower or 0, 'accuracy': m.accuracy if m.accuracy is not True else 0,
                           'priority': m.priority or 0, 'pp': m.pp, 'target': m.target}
        items = []
        for name in sorted(leg.items):
            it = dex.items.get(name)
            items.append({'name': it.name, 'ko': _ko('items', it.name), 'megaStone': bool(it.megaStone),
                          'megaFor': list(it.megaStone.keys()) if it.megaStone else []})
        abilities = {a: _ko('abilities', a) for a in sorted(used_abilities)}
        natures = []
        for n in sorted(dex.natures.all(), key=lambda n: n.name):
            natures.append({'name': n.name, 'ko': _ko('natures', n.name), 'plus': n.plus or '', 'minus': n.minus or ''})
        types = sorted({t for s in species for t in s['types']})
        self._data = {
            'format': self.formatid, 'picked': leg.picked_team_size or 3,
            'species': species, 'moves': moves, 'items': items, 'abilities': abilities, 'natures': natures,
            'types': {t: korean_names()['types'].get(to_id(t), '') for t in types},
            'statuses': [{'id': k, 'ko': v} for k, v in STATUS_KO.items()],
            'weathers': [{'id': k, 'ko': v} for k, v in WEATHER_KO.items()],
            'terrains': [{'id': k, 'ko': v} for k, v in TERRAIN_KO.items()],
            'pseudo': [{'id': k, 'ko': v} for k, v in PSEUDO_KO.items()],
            'side_conditions': [{'id': k, 'ko': v, 'layers': 3 if k == 'spikes' else 2 if k == 'toxicspikes' else 1,
                                 'timed': k in ('reflect', 'lightscreen', 'auroraveil', 'tailwind', 'safeguard')}
                                for k, v in SIDE_KO.items()],
            'volatiles': [{'id': k, 'ko': v} for k, v in VOLATILE_KO.items()],
            'boosts': [{'id': k, 'ko': v} for k, v in BOOST_KO.items()],
        }
        return self._data

    # ------------------------------------------------------------------
    def recommended_teams(self, limit: int = 5) -> dict:
        """Teams recommended by the team-building AI (hall of fame first, then the best of the population)."""
        if not self.library_path:
            return {'teams': [], 'note': '팀 라이브러리(models/teams_singles.json)를 찾지 못했습니다. 다운로드한 폴더에서 '
                                         '실행하거나 --library 로 파일을 지정하세요 (직접 만들기: python -m pokechamp evolve).'}
        data = _load_json(self.library_path)
        out, seen = [], set()
        hof = list(reversed(data.get('hall_of_fame') or []))
        pop = list(zip(data.get('teams') or [], data.get('fitness') or []))
        candidates = [(t, None, '명예의 전당') for t in hof] + [(t, f, '개체군') for t, f in pop]
        for team, fit, source in candidates:
            # teams with the same Pokemon holding the same items are near-duplicates (they differ by a move,
            # a nature or an ability at most): keep the first = best of each group
            key = tuple(sorted((s['species'], s.get('item') or '') for s in team))
            if key in seen:
                continue
            seen.add(key)
            sets = [_plain_set(s) for s in team]
            out.append({'rank': len(out) + 1, 'source': source, 'fitness': fit, 'sets': sets,
                        'text': export_team(_obj_team(sets)),
                        'species': [s['species'] for s in sets],
                        'species_ko': [_ko('species', s['species']) for s in sets],
                        'stats': [calc_stats(Obj(s)) for s in _obj_team(sets)]})
            if len(out) >= limit:
                break
        usage = []
        usage_path = os.path.join(os.path.dirname(self.library_path), 'usage_singles.json')
        if os.path.exists(usage_path):
            u = _load_json(usage_path)
            for sp, info in sorted(u.items(), key=lambda kv: -kv[1].get('win_rate', 0) * min(1, kv[1].get('games', 0) / 50)):
                if info.get('games', 0) < 30:
                    continue
                usage.append({'species': sp, 'ko': _ko('species', sp), 'win_rate': info['win_rate'],
                              'games': info['games']})
                if len(usage) >= 15:
                    break
        return {'teams': out, 'top_species': usage, 'generation': data.get('generation')}

    # ------------------------------------------------------------------
    def parse_team(self, text: str | None = None, sets: list | None = None) -> dict:
        from ..teambuilder.validator import TeamValidator
        if text is not None:
            _need(isinstance(text, str), '"text" must be a string (a Showdown export)')
            team = import_team(text)
        else:
            _check_sets(sets or [], '"sets"')
            team = _obj_team(sets or [])
        dex = get_dex()
        out = []
        for s in team:
            sp = dex.species.get(s.species)
            s.species = sp.name if sp.exists else s.species
            ps = _plain_set(s)
            stats = calc_stats(Obj(ps), sp) if sp.exists else {}
            ps['stats'] = stats
            ps['species_ko'] = _ko('species', ps['species'])
            out.append(ps)
        problems = TeamValidator(self.formatid).validate_team(team)
        return {'sets': out, 'problems': problems, 'text': export_team(_obj_team(out)) if out else ''}

    # ------------------------------------------------------------------
    def team_preview(self, my_team: list, foe_species: list, sims: int = 12, top: int = 6) -> dict:
        """Rank the bring-3 options (first = lead) against the opponent's six by simulation."""
        from ..ai.advisor import SetPrior, _species_name
        from ..ai.heuristic import HeuristicAgent
        from ..env.actions import team_preview_options
        from ..env.runner import BattleSession, gen5_seed, play_battle
        t0 = time.time()
        _check_sets(my_team, '"my_team"')
        _need(isinstance(foe_species, list) and all(isinstance(n, str) for n in foe_species),
              '"foe" must be a list of 6 species names')
        my_sets = [_plain_set(s) for s in my_team]
        foe_species = [_species_name(n) for n in foe_species]
        if len(my_sets) != 6 or len(foe_species) != 6:
            raise ValueError('need 6 Pokemon on both teams')
        _need(len(set(foe_species)) == 6, 'the opponent\'s team preview lists the same species twice')
        gen = self._new_generation()
        rng = random.Random(self.seed + 17)
        prior = SetPrior(self.formatid, self.library_path)

        def sample_foe_team(r):
            used = set()
            team = []
            for n in foe_species:
                s = prior.sample(n, None, r, used, mega_allowed=True)
                if s['item']:
                    used.add(s['item'])
                team.append(s)
            return team

        options = team_preview_options(6, 3, 1)
        # scores from the policy network and the heuristic, at the team-preview decision
        session = BattleSession(self.formatid, _obj_team(my_sets), _obj_team(sample_foe_team(rng)),
                                seed=gen5_seed(rng))
        view = session.view('p1')
        req = session.request('p1')
        legal = session.legal('p1')
        heur = HeuristicAgent(0).preview_scores(view, legal)
        policy = [None] * len(legal)
        model = self.advisor.model  # None: no network (or it could not be loaded) -> heuristic only
        if model is not None:
            with self._lock:
                from ..ai.nn_agent import NNAgent
                probs, _v, _ = NNAgent(model).evaluate(view, req, legal)
            policy = [float(p) for p in probs]
        hmin, hmax = min(heur), max(heur)
        ranked = []
        for opt, h, p in zip(legal, heur, policy):
            hn = (h - hmin) / (hmax - hmin) if hmax > hmin else 0.5
            prior_score = hn if p is None else 0.5 * hn + 0.5 * (p / max(policy))
            ranked.append((prior_score, opt, h, p))
        ranked.sort(key=lambda x: -x[0])
        shortlist = ranked[:top]
        # simulate each shortlisted choice against sampled opponent sets (the opponent picks with the heuristic)
        results = []
        agent_me = self._battle_agent()
        cancelled = False
        for prior_score, opt, h, p in shortlist:
            wins, games = 0.0, 0
            for k in range(sims):
                if self._gen != gen:
                    cancelled = True  # cancelled / superseded: return what has been simulated
                    break
                r = random.Random(self.seed * 1000 + k)
                foe_team = sample_foe_team(r)
                res = play_battle(_obj_team(my_sets), _obj_team(foe_team), _FixedPreview(opt, agent_me),
                                  HeuristicAgent(k), self.formatid, seed=gen5_seed(r), max_turns=100)
                wins += 1.0 if res.winner == 'p1' else 0.5 if res.winner is None else 0.0
                games += 1
            if games:
                results.append({'order': list(opt), 'lead': my_sets[opt[0]]['species'],
                                'names': [my_sets[i]['species'] for i in opt],
                                'names_ko': [_ko('species', my_sets[i]['species']) for i in opt],
                                'win_rate': wins / games, 'games': games, 'policy': p, 'matchup': h,
                                'prior_score': prior_score})
            if cancelled:
                break
        results.sort(key=lambda r: (-r['win_rate'], -r['prior_score']))
        return {'options': results, 'considered': len(options), 'elapsed': round(time.time() - t0, 1),
                'cancelled': cancelled}

    def _battle_agent(self):
        model = self.advisor.model
        if model is not None:
            from ..ai.nn_agent import NNAgent
            return NNAgent(model)
        from ..ai.heuristic import HeuristicAgent
        return HeuristicAgent(0)

    # ------------------------------------------------------------------
    def advise(self, state: dict, samples: int = 12, depth: int = 2) -> dict:
        """Ranked actions for the described turn (state = advisor JSON, see examples/advisor_state.json)."""
        _check_state_names(state)
        return self._advise(state, samples, depth, None)

    def _advise(self, state: dict, samples: int, depth: int, gen: int | None) -> dict:
        """One advise run; ``gen`` = the AI-request generation it belongs to (None: start a new one once the
        state has been parsed, so a malformed request does not cancel a running calculation)."""
        from ..ai.advisor import parse_input
        t0 = time.time()
        state = dict(state)
        state.setdefault('format', self.formatid)
        inp = parse_input(state)
        _check_parsed(inp, self.formatid)
        if gen is None:
            gen = self._new_generation()
        log_lines = []
        if self._gen != gen:  # superseded before it started
            return {'recommendations': [], 'value': None, 'foe_replies': [], 'beliefs': {}, 'samples': 0,
                    'elapsed': 0.0, 'cancelled': True, 'log': []}
        with self._lock:
            self.advisor.cancel_check = lambda: self._gen != gen
            try:
                recs = self.advisor.recommend(inp, determinizations=samples, depth=depth, log=log_lines.append)
            finally:
                self.advisor.cancel_check = None
            info = dict(self.advisor.last_info)
        cancelled = self._gen != gen
        total = sum(info.get('foe_replies', {}).values()) or 1.0
        foe_replies = sorted(({'label': k, 'prob': v / total} for k, v in info.get('foe_replies', {}).items()),
                             key=lambda x: -x['prob'])[:6]
        beliefs = {}
        for name, ms in inp.foe.mons.items():
            if ms.belief is not None:
                beliefs[name] = ms.belief.most_likely(name)
        return {
            'recommendations': [{'label': r.label, 'label_ko': self._label_ko(r.meaning, r.label),
                                 'win_rate': r.win_rate, 'stderr': r.stderr, 'policy': r.prior,
                                 'samples': r.samples, 'option': list(r.option),
                                 'meaning': [list(m) if m else None for m in (r.meaning or ())]}
                                for r in recs],
            'value': info.get('value'), 'foe_replies': foe_replies, 'beliefs': beliefs,
            'samples': info.get('samples'), 'elapsed': round(time.time() - t0, 1), 'cancelled': cancelled,
            'log': [x for x in log_lines if 'could not rebuild' in x][:5],
        }

    def advise_switch(self, state: dict, samples: int = 8, depth: int = 2) -> dict:
        """After one of our Pokemon fainted: which teammate to send in (each evaluated as the new active).

        The whole call is one AI request: /api/cancel (or a newer request) stops it after the current sample,
        and only fully evaluated candidates are returned (``cancelled`` then is true)."""
        _check_state_names(state)
        gen = self._new_generation()
        t0 = time.time()
        me = state['me']
        fainted = set(n for n, m in (me.get('pokemon') or {}).items() if m.get('fainted'))
        candidates = [n for n in me.get('brought') or [] if n not in fainted and n not in (me.get('active') or [])]
        if not candidates:
            candidates = [n for n in me.get('brought') or [] if n not in fainted]
        out = []
        cancelled = False
        for c in candidates:
            st = copy.deepcopy(state)
            st['me']['active'] = [c]
            mons = st['me'].setdefault('pokemon', {})
            mons.setdefault(c, {})
            mons[c]['fresh'] = True
            mons[c]['boosts'] = {}
            res = self._advise(st, samples, depth, gen)
            if res['cancelled']:
                cancelled = True  # a partly evaluated candidate is not comparable with the others
                break
            best = res['recommendations'][0] if res['recommendations'] else None
            out.append({'switch_to': c, 'switch_to_ko': _ko('species', c),
                        'win_rate': best['win_rate'] if best else None,
                        'best_next_action': best['label_ko'] if best else None, 'value': res['value']})
        out.sort(key=lambda x: -(x['win_rate'] or 0))
        return {'options': out, 'elapsed': round(time.time() - t0, 1), 'cancelled': cancelled}

    # ------------------------------------------------------------------
    def _label_ko(self, meaning, fallback: str) -> str:
        if not meaning:
            return fallback
        parts = []
        for m in meaning:
            if not m:
                continue
            kind, value = m
            if kind == 'switch':
                parts.append(f"교체 → {_ko('species', value) or value}")
            else:
                parts.append(_ko('moves', value) or value)
        label = ' / '.join(parts) or fallback
        if 'Mega Evolve' in fallback:
            label = '메가진화 + ' + label
        return label


class _FixedPreview:
    """Wraps an agent but plays a fixed team-preview choice."""

    def __init__(self, option, base):
        self.option = tuple(option)
        self.base = base

    def choose(self, session, sid, view, legal):
        req = session.request(sid)
        if req.get('teamPreview'):
            return self.option if self.option in legal else legal[0]
        return self.base.choose(session, sid, view, legal)
