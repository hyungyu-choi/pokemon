"""Local web server for the battle assistant UI (standard library only).

    python -m pokechamp ui            # then open http://127.0.0.1:8765

JSON API (all POST bodies are JSON objects; errors come back as ``{"error": "..."}`` with status 400/413/500):

    GET  /api/status                    model / library / Korean-name availability
    GET  /api/data                      game data for the selection menus
    GET  /api/recommended_teams         teams recommended by the team-building AI
    POST /api/team/parse                {"text": "<Showdown export>"} or {"sets": [...]} -> sets, problems, stats
    POST /api/preview                   {"my_team": [sets], "foe": [6 species], "sims"?: 1..48}
    POST /api/advise                    {"state": <advisor state>, "samples"?: 1..64, "depth"?: 1..3}
    POST /api/advise_switch             {"state": <advisor state>, "samples"?: 1..64, "depth"?: 1..3}
    POST /api/cancel                    stop the running AI calculation (it returns its partial result)
"""
from __future__ import annotations

import json
import mimetypes
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .service import AssistantService

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
MAX_BODY = 2 * 1024 * 1024  # bytes; a <state> is a few KB


class BodyTooLarge(ValueError):
    pass


def _int_param(body: dict, key: str, default: int, lo: int, hi: int) -> int:
    """An optional integer parameter of a request body, checked against its allowed range."""
    v = body.get(key)
    if v is None or v == '':
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError(f'"{key}" must be an integer between {lo} and {hi}')
    try:
        n = int(v)
    except (ValueError, OverflowError):  # "abc", NaN, Infinity
        raise ValueError(f'"{key}" must be an integer between {lo} and {hi}') from None
    if n != float(v) or not lo <= n <= hi:
        raise ValueError(f'"{key}" must be an integer between {lo} and {hi} (got {v!r})')
    return n


def make_handler(service: AssistantService):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'pokechamp-ui/1.0'

        def log_message(self, fmt, *args):  # quieter console
            if os.environ.get('POKECHAMP_UI_LOG'):
                super().log_message(fmt, *args)

        # -- helpers ---------------------------------------------------------
        def _send(self, status: int, body: bytes, ctype: str):
            try:
                self.send_response(status)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                if not getattr(self, '_head_only', False):
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # the browser gave up on the request (e.g. the user pressed "취소" during a long calculation)
                self.close_connection = True

        def _json(self, status: int, obj):
            self._send(status, json.dumps(obj, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')

        def _body(self) -> dict:
            try:
                n = int(self.headers.get('Content-Length') or 0)
            except ValueError:
                raise ValueError('invalid Content-Length') from None
            if n <= 0:
                return {}
            if n > MAX_BODY:
                raise BodyTooLarge(f'request body too large ({n} bytes, limit {MAX_BODY})')
            body = json.loads(self.rfile.read(n).decode('utf-8'))
            if not isinstance(body, dict):
                raise ValueError('the request body must be a JSON object')
            return body

        # -- routes ------------------------------------------------------------
        def do_HEAD(self):
            self._head_only = True
            try:
                self.do_GET()
            finally:
                self._head_only = False

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == '/api/status':
                    return self._json(200, service.status())
                if path == '/api/data':
                    return self._json(200, service.game_data())
                if path == '/api/recommended_teams':
                    return self._json(200, service.recommended_teams())
                if path in ('/', '/index.html'):
                    return self._static('index.html')
                if path.startswith('/static/'):
                    return self._static(path[len('/static/'):])
                return self._json(404, {'error': 'not found'})
            except Exception as exc:  # pragma: no cover - reported to the client
                traceback.print_exc()
                return self._json(500, {'error': f'{type(exc).__name__}: {exc}'})

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                body = self._body()
                if path == '/api/team/parse':
                    return self._json(200, service.parse_team(body.get('text'), body.get('sets')))
                if path == '/api/preview':
                    return self._json(200, service.team_preview(body['my_team'], body['foe'],
                                                                sims=_int_param(body, 'sims', 12, 1, 48)))
                if path == '/api/advise':
                    return self._json(200, service.advise(body['state'],
                                                          samples=_int_param(body, 'samples', 12, 1, 64),
                                                          depth=_int_param(body, 'depth', 2, 1, 3)))
                if path == '/api/cancel':
                    return self._json(200, service.cancel())
                if path == '/api/advise_switch':
                    return self._json(200, service.advise_switch(body['state'],
                                                                 samples=_int_param(body, 'samples', 8, 1, 64),
                                                                 depth=_int_param(body, 'depth', 2, 1, 3)))
                return self._json(404, {'error': 'not found'})
            except BodyTooLarge as exc:
                self.close_connection = True  # the unread body must not be parsed as the next request
                return self._json(413, {'error': str(exc)})
            except (KeyError, ValueError, TypeError) as exc:
                traceback.print_exc()
                return self._json(400, {'error': f'{type(exc).__name__}: {exc}'})
            except Exception as exc:  # pragma: no cover
                traceback.print_exc()
                return self._json(500, {'error': f'{type(exc).__name__}: {exc}'})

        def _static(self, rel: str):
            full = os.path.normpath(os.path.join(STATIC_DIR, rel))
            if os.path.commonpath([full, STATIC_DIR]) != STATIC_DIR or not os.path.isfile(full):
                return self._json(404, {'error': 'not found'})
            ctype = mimetypes.guess_type(full)[0] or 'application/octet-stream'
            if ctype.startswith('text/') or ctype in ('application/javascript',):
                ctype += '; charset=utf-8'
            with open(full, 'rb') as f:
                return self._send(200, f.read(), ctype)

    return Handler


def serve(host: str = '127.0.0.1', port: int = 8765, service: AssistantService | None = None,
          open_browser: bool = False):
    service = service or AssistantService()
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    # load data and the network in the background so the first real request is fast
    import threading
    threading.Thread(target=service.warmup, daemon=True).start()
    url = f'http://{host if host not in ("0.0.0.0", "") else "127.0.0.1"}:{httpd.server_address[1]}/'
    print(f'pokechamp battle assistant: {url}  (model: {service.status()["model"]}, '
          f'team library: {service.status()["library"]})', flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return httpd
