"""Running the battle assistant on the user's own computer: launchers, start-up checks, console messages,
static file serving (Content-Type, Windows-style paths), port errors and the AI fallbacks."""
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

from pokechamp.ui import console, server
from pokechamp.ui.server import make_server, serve, static_file, static_type
from pokechamp.ui.service import AssistantService

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cp949_safe(text: str) -> bool:
    try:
        text.encode('cp949')
        return True
    except UnicodeEncodeError:
        return False


# ---------------------------------------------------------------------------
# HTTP: static files

@pytest.fixture(scope='module')
def http():
    httpd = make_server('127.0.0.1', 0, AssistantService(model_path='', seed=1))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f'http://127.0.0.1:{httpd.server_address[1]}'
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def test_static_content_types(http):
    for path, ctype in (('/', 'text/html; charset=utf-8'), ('/static/app.js', 'text/javascript; charset=utf-8'),
                        ('/static/combobox.js', 'text/javascript; charset=utf-8'),
                        ('/static/style.css', 'text/css; charset=utf-8'),
                        ('/api/status', 'application/json; charset=utf-8')):
        code, headers, body = _get(http + path)
        assert code == 200, path
        assert headers['Content-Type'] == ctype, path
        assert headers['X-Content-Type-Options'] == 'nosniff'
        assert body


def test_static_types_ignore_the_mimetypes_registry(monkeypatch):
    # on Windows, mimetypes reads HKEY_CLASSES_ROOT, where .css / .js are sometimes text/plain
    monkeypatch.setattr(server.mimetypes, 'guess_type', lambda *a, **k: ('text/plain', None))
    assert static_type('style.css') == 'text/css; charset=utf-8'
    assert static_type('app.js') == 'text/javascript; charset=utf-8'
    assert static_type('index.html') == 'text/html; charset=utf-8'
    assert static_type('icon.svg') == 'image/svg+xml'
    assert static_type('icon.png') == 'image/png'
    assert static_type('readme.unknownext') == 'text/plain; charset=utf-8'  # fallback for other extensions


@pytest.mark.parametrize('rel', ['', '../server.py', '..\\server.py', '..', '.', 'C:/Windows/win.ini', 'C:\\Windows\\win.ini',
                                 'D:server.py', '\\\\evil\\share\\x', '//evil/share/x', 'APP.JS', 'app.js::$DATA',
                                 'app.js.', 'app.js ', '.hidden', 'app.js\0', 'sub/app.js', '%2e%2e/server.py'])
def test_static_file_rejects_anything_but_a_listed_name(rel):
    assert static_file(rel) is None


def test_static_file_accepts_the_page_files():
    for name in ('index.html', 'app.js', 'combobox.js', 'style.css'):
        path = static_file(name)
        assert path and os.path.isfile(path) and os.path.dirname(path) == server.STATIC_DIR


def test_static_windows_style_paths_are_404_not_500(http):
    for path in ('/static/..%5cserver.py', '/static/..\\server.py', '/static/C:/Windows/win.ini', '/static/D:x',
                 '/static///evil/share/x', '/static/APP.JS', '/static/../server.py', '/static/'):
        assert _get(http + path)[0] == 404, path


# ---------------------------------------------------------------------------
# server start: port in use, banner, LAN addresses

def test_port_in_use_gives_a_korean_message_and_exit_1(capsys):
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        blocker.bind(('127.0.0.1', 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]
        with pytest.raises(SystemExit) as e:
            serve('127.0.0.1', port, AssistantService(model_path=''))
        assert e.value.code == 1
    finally:
        blocker.close()
    err = capsys.readouterr().err
    assert f'포트 {port}번을 이미' in err and f'--port {port + 1}' in err and 'Traceback' not in err


@pytest.mark.parametrize('exc,needle', [
    (OSError(10048, 'Only one usage of each socket address'), '이미 다른 프로그램이'),   # WinError 10048
    (OSError(98, 'Address already in use'), '이미 다른 프로그램이'),
    (OSError(10013, 'An attempt was made to access a socket in a way forbidden'), '권한 없음'),  # Hyper-V reserved
    (OSError(99, 'Cannot assign requested address'), '--host 0.0.0.0'),
    (socket.gaierror(-2, 'Name or service not known'), '--host 0.0.0.0'),
    (OverflowError('bind(): port must be 0-65535.'), '--port 8766'),
])
def test_port_error_messages(exc, needle):
    msg = console.port_error_message('127.0.0.1', 8765, exc)
    assert needle in msg and _cp949_safe(msg)


def test_windows_server_does_not_share_ports():
    assert server.UIServer.allow_reuse_address == (os.name != 'nt')


def test_dropped_connections_are_not_reported(capsys):
    httpd = make_server('127.0.0.1', 0, AssistantService(model_path=''))
    try:
        try:
            raise ConnectionAbortedError(10053, 'An established connection was aborted by the software in your host')
        except ConnectionAbortedError:
            httpd.handle_error(None, ('127.0.0.1', 1))
    finally:
        httpd.server_close()
    assert capsys.readouterr().err == ''


def test_send_survives_an_aborted_connection():
    """WinError 10053 while writing the response (the browser cancelled the request) closes quietly."""
    class Aborted(io.RawIOBase):
        def write(self, b):
            raise ConnectionAbortedError(10053, 'aborted')

    handler_cls = server.make_handler(AssistantService(model_path=''))
    h = handler_cls.__new__(handler_cls)
    h.wfile = Aborted()
    h.request_version, h.requestline, h.command, h.client_address = 'HTTP/1.1', 'GET / HTTP/1.1', 'GET', ('x', 0)
    h.close_connection = False
    h._send(200, b'{}', 'application/json')
    assert h.close_connection is True


def test_startup_banner():
    text = console.startup_banner('0.0.0.0', 8765, notes=['[주의] 테스트'], lan=['192.168.0.12'])
    assert 'http://127.0.0.1:8765/' in text and 'http://192.168.0.12:8765/' in text
    assert '방화벽' in text and 'Ctrl+C' in text and '[주의] 테스트' in text and _cp949_safe(text)
    text = console.startup_banner('0.0.0.0', 8765, lan=[])
    assert 'ipconfig' in text and _cp949_safe(text)
    text = console.startup_banner('127.0.0.1', 8790)
    assert 'http://127.0.0.1:8790/' in text and '휴대폰' not in text and 'Ctrl+C' in text
    assert console.local_url('', 8765) == 'http://127.0.0.1:8765/'
    assert console.local_url('192.168.0.5', 80) == 'http://192.168.0.5:80/'


def test_lan_addresses_never_fail(monkeypatch):
    ips = console.lan_addresses()
    assert isinstance(ips, list) and all(isinstance(ip, str) and not ip.startswith('127.') for ip in ips)

    def no_network(*a, **k):
        raise OSError(101, 'Network is unreachable')
    monkeypatch.setattr(console.socket, 'socket', no_network)
    assert console.lan_addresses() == []


# ---------------------------------------------------------------------------
# start-up checks of `python -m pokechamp ui`

def test_dependency_message():
    msg = console.dependency_message({'torch': "ModuleNotFoundError: No module named 'torch'"},
                                     python='C:\\Program Files\\Python313\\python.exe', version=(3, 13),
                                     platform='win32', bits64=True)
    assert '"C:\\Program Files\\Python313\\python.exe" -m pip install numpy torch' in msg
    assert 'run_ui.bat' in msg and '--no-model' in msg and 'PowerShell' in msg and _cp949_safe(msg)
    assert 'Python 3.13 (64비트)을 설치' not in msg  # a supported Python: no version advice
    msg = console.dependency_message({'torch': 'x'}, python='/usr/bin/python3', version=(3, 15), platform='linux',
                                     bits64=True)
    assert '--index-url https://download.pytorch.org/whl/cpu' in msg and 'Python 3.13' in msg and _cp949_safe(msg)
    msg = console.dependency_message({'numpy': 'x'}, python='python', version=(3, 12), platform='darwin', bits64=True)
    assert 'python -m pip install numpy\n' in msg and '--no-model' not in msg and _cp949_safe(msg)


def _run_ui_main(monkeypatch, argv, missing=()):
    """Run ``main(['ui', ...])`` with ``missing`` packages failing to import; serve() is replaced by a recorder."""
    from pokechamp import __main__ as cli
    real = console.import_problem
    monkeypatch.setattr(console, 'import_problem',
                        lambda m: f"ModuleNotFoundError: No module named '{m}'" if m in missing else real(m))
    calls = []
    monkeypatch.setattr(server, 'serve', lambda *a, **k: calls.append((a, k)))
    cli.main(['ui'] + list(argv))
    return calls


def test_ui_without_torch_explains_and_exits_1(monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        _run_ui_main(monkeypatch, ['--port', '0'], missing=('torch',))
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert 'torch (PyTorch' in err and '-m pip install' in err and sys.executable in err and 'Traceback' not in err


def test_ui_without_numpy_exits_1_even_without_the_network(monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        _run_ui_main(monkeypatch, ['--no-model'], missing=('numpy',))
    assert e.value.code == 1 and 'numpy' in capsys.readouterr().err


def test_ui_no_model_needs_no_torch(monkeypatch):
    calls = _run_ui_main(monkeypatch, ['--no-model', '--port', '8799'], missing=('torch',))
    (args, kwargs), = calls
    host, port, service = args
    assert (host, port) == ('127.0.0.1', 8799) and not service.model_path and service.status()['model'] is None
    assert any('--no-model' in n for n in kwargs['notes'])


def test_ui_missing_model_file_is_an_error(monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        _run_ui_main(monkeypatch, ['--model', os.path.join(ROOT, 'models', 'nope.pt')])
    assert e.value.code == 1 and 'nope.pt' in capsys.readouterr().err


def test_ui_default_start(monkeypatch):
    calls = _run_ui_main(monkeypatch, ['--host', '0.0.0.0', '--open'])
    (args, kwargs), = calls
    assert args[0] == '0.0.0.0' and args[1] == 8765 and kwargs['open_browser'] is True
    assert args[2].status()['model'] == 'battle_singles.pt' and kwargs['notes'] == []


# ---------------------------------------------------------------------------
# AI fallbacks and user files

def test_unloadable_model_falls_back_to_the_heuristic(tmp_path, capsys):
    bad = tmp_path / 'broken.pt'
    bad.write_bytes(b'this is not a checkpoint')
    svc = AssistantService(model_path=str(bad))
    assert svc.status()['model'] == 'broken.pt'
    assert svc.advisor.model is None
    st = svc.status()
    assert st['model'] is None and st['model_error']
    assert '휴리스틱' in capsys.readouterr().err


def test_heuristic_ai_works_without_torch():
    """advisor / service import and advise without PyTorch (``import torch`` made to fail)."""
    code = ("import sys, json; sys.modules['torch'] = None\n"
            "from pokechamp.ui.service import AssistantService\n"
            "svc = AssistantService(model_path='')\n"
            "state = json.load(open('examples/advisor_state.json', encoding='utf-8'))\n"
            "res = svc.advise(state, samples=1, depth=1)\n"
            "assert 'torch' not in sys.modules or sys.modules['torch'] is None\n"
            "print(json.dumps([r['label'] for r in res['recommendations']]))\n")
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=300,
                         encoding='utf-8')
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1])


def test_json_files_with_a_bom_load(tmp_path):
    src = os.path.join(ROOT, 'models', 'teams_singles.json')
    lib = tmp_path / 'teams_singles.json'
    lib.write_bytes(b'\xef\xbb\xbf' + open(src, 'rb').read())
    svc = AssistantService(model_path='', library_path=str(lib))
    assert svc.recommended_teams()['teams']
    from pokechamp.ai.advisor import SetPrior
    assert SetPrior(svc.formatid, str(lib)).library


def test_models_found_from_the_project_folder(monkeypatch, tmp_path):
    from pokechamp.ui import service as service_mod
    assert service_mod.default_model_file('battle_singles.pt') == service_mod.MODEL_PATH
    # an installed package (ROOT without models/) run from the project folder still finds them
    monkeypatch.setattr(service_mod, 'ROOT', str(tmp_path))
    monkeypatch.chdir(ROOT)
    assert service_mod.default_model_file('teams_singles.json') == os.path.join(ROOT, 'models', 'teams_singles.json')
    monkeypatch.chdir(tmp_path)
    assert service_mod.default_model_file('teams_singles.json') is None


# ---------------------------------------------------------------------------
# launchers

def test_windows_launcher_is_plain_ascii_with_crlf():
    raw = open(os.path.join(ROOT, 'run_ui.bat'), 'rb').read()
    assert all(b < 128 for b in raw), 'cmd.exe reads .bat files in the console code page: keep it ASCII'
    assert raw.count(b'\n') == raw.count(b'\r\n') > 20, 'cmd.exe needs CRLF line endings'
    assert not raw.startswith(b'\xef\xbb\xbf')
    text = raw.decode('ascii').lower()
    for needle in ('chcp 65001', 'cd /d "%here%"', 'python.org/downloads', '-m venv', 'pip install', 'numpy torch',
                   'pokechamp-deps-ok', '-m pokechamp ui --open %*', 'pause', 'add python.exe to path'):
        assert needle in text, needle


def test_unix_launchers():
    for name in ('run_ui.sh', 'run_ui.command'):
        path = os.path.join(ROOT, name)
        raw = open(path, 'rb').read()
        assert b'\r' not in raw and raw.startswith(b'#!/bin/sh\n'), name
        if os.name != 'nt':
            assert os.access(path, os.X_OK), f'{name} must be executable'
        sh = shutil.which('sh')
        if sh:
            assert subprocess.run([sh, '-n', path]).returncode == 0, name
    text = open(os.path.join(ROOT, 'run_ui.sh'), encoding='utf-8').read()
    for needle in ('download.pytorch.org/whl/cpu', '-m venv', 'pokechamp-deps-ok', '-m pokechamp ui', '"$@"'):
        assert needle in text, needle


def test_gitattributes_keep_launcher_line_endings():
    text = open(os.path.join(ROOT, '.gitattributes'), encoding='utf-8').read()
    assert '*.bat text eol=crlf' in text and '*.sh text eol=lf' in text and '*.pt binary' in text
