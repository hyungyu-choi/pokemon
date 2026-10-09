"""Local web server for the battle assistant UI (standard library only).

    python -m pokechamp ui            # then open http://127.0.0.1:8765  (or double-click run_ui.bat / run_ui.sh)

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
import socket
import socketserver
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .service import AssistantService

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
MAX_BODY = 2 * 1024 * 1024  # bytes; a <state> is a few KB
# Fixed types for the page's own files: on Windows ``mimetypes`` also reads the registry, where some installed
# programs map .css / .js to text/plain - the browser then ignores the stylesheet and the page is unstyled.
STATIC_TYPES = {
    '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8',
    '.txt': 'text/plain; charset=utf-8', '.md': 'text/markdown; charset=utf-8',
    '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon', '.webp': 'image/webp',
}


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


def static_type(name: str) -> str:
    """Content-Type for a file of static/ (explicit for the usual web types, ``mimetypes`` for the rest)."""
    ext = os.path.splitext(name)[1].lower()
    if ext in STATIC_TYPES:
        return STATIC_TYPES[ext]
    ctype = mimetypes.guess_type(name)[0] or 'application/octet-stream'
    if ctype.startswith('text/') or ctype == 'application/javascript':
        ctype += '; charset=utf-8'
    return ctype


def static_file(rel: str) -> str | None:
    """Path of the file named exactly ``rel`` directly inside static/, else None.

    Only a plain file name listed in the directory is accepted - no '/', '\\', ':', NUL or leading '.' - so no
    request can reach outside static/ on any OS (Windows drive letters, UNC paths, '..\\', alternate data
    streams) and names match case-sensitively everywhere."""
    if not rel or rel.startswith('.') or any(c in rel for c in '/\\:\0'):
        return None
    try:
        names = os.listdir(STATIC_DIR)
    except OSError:
        return None
    if rel not in names:
        return None
    full = os.path.join(STATIC_DIR, rel)
    return full if os.path.isfile(full) else None


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
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                if not getattr(self, '_head_only', False):
                    self.wfile.write(body)
            except ConnectionError:
                # the browser gave up on the request (e.g. the user pressed "취소" during a long calculation):
                # BrokenPipe / ConnectionReset on Linux and macOS, ConnectionAborted (WinError 10053) on Windows
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
            except ConnectionError:  # the browser went away
                self.close_connection = True
            except Exception as exc:  # pragma: no cover - reported to the client
                traceback.print_exc()
                return self._json(500, {'error': _error_text(exc)})

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
            except ConnectionError:  # the browser went away
                self.close_connection = True
            except (KeyError, ValueError, TypeError) as exc:
                # a bad request (unknown name, impossible situation ...): one line, not a scary traceback
                if os.environ.get('POKECHAMP_UI_LOG'):
                    traceback.print_exc()
                else:
                    print(f'[400] {path}: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
                return self._json(400, {'error': f'{type(exc).__name__}: {exc}'})
            except Exception as exc:  # pragma: no cover
                traceback.print_exc()
                return self._json(500, {'error': _error_text(exc)})

        def _static(self, rel: str):
            full = static_file(rel)
            if full is None:
                return self._json(404, {'error': 'not found'})
            with open(full, 'rb') as f:
                return self._send(200, f.read(), static_type(rel))

    return Handler


def _error_text(exc: BaseException) -> str:
    text = f'{type(exc).__name__}: {exc}'
    if isinstance(exc, ImportError):
        text = ('AI 계산에 필요한 파이썬 패키지가 설치되어 있지 않습니다 (서버 창의 안내를 보세요: '
                f'python -m pip install numpy torch). {text}')
    return text


class UIServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that refuses a port another server already listens on (also on Windows), does not
    print tracebacks when a browser drops a connection, and starts without a DNS lookup."""

    # On Windows SO_REUSEADDR lets a second server bind a port that is already in use (both then get requests);
    # SO_EXCLUSIVEADDRUSE makes the second one fail instead. Elsewhere SO_REUSEADDR only allows a quick restart.
    allow_reuse_address = os.name != 'nt'
    daemon_threads = True

    def server_bind(self):
        if os.name == 'nt' and hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        socketserver.TCPServer.server_bind(self)
        # HTTPServer.server_bind would also call socket.getfqdn(host), a reverse DNS lookup that can stall the
        # start for seconds on some networks; the name is not used
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], ConnectionError):
            return  # the browser closed the connection (cancelled request, closed tab): nothing to report
        super().handle_error(request, client_address)


def make_server(host: str, port: int, service: AssistantService) -> UIServer:
    return UIServer((host, port), make_handler(service))


def serve(host: str = '127.0.0.1', port: int = 8765, service: AssistantService | None = None,
          open_browser: bool = False, notes=()):
    """Run the UI until Ctrl+C. A port that cannot be opened ends the program with a message (exit code 1)."""
    from . import console
    service = service or AssistantService()
    try:
        httpd = make_server(host, port, service)
    except (OSError, OverflowError) as exc:  # port in use / reserved, bad host, port out of range
        print(console.port_error_message(host, port, exc), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
    # load data and the network in the background so the first real request is fast
    threading.Thread(target=service.warmup, daemon=True).start()
    port = httpd.server_address[1]
    url = console.local_url(host, port)
    st = service.status()
    print(console.startup_banner(host, port, notes), flush=True)
    print(f'pokechamp battle assistant: {url}  (model: {st["model"]}, team library: {st["library"]})', flush=True)
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:  # no browser available: the address is printed above
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('배틀 도우미를 종료했습니다.', flush=True)
    finally:
        httpd.server_close()
    return httpd
