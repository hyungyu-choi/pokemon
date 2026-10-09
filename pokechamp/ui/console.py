"""What ``python -m pokechamp ui`` checks before starting and prints in its console window.

The messages are in Korean for the UI's users. They use only characters that the Korean Windows console
code page (cp949) can show (Hangul and ASCII; no arrows, ellipses or emoji), so they also print correctly
when the output is redirected to a cp949 file or pipe.
"""
from __future__ import annotations

import errno
import ipaddress
import socket
import sys

PYTHON_DOWNLOAD = 'https://www.python.org/downloads/'
TORCH_CPU_INDEX = 'https://download.pytorch.org/whl/cpu'

_ADDR_IN_USE = {errno.EADDRINUSE, getattr(errno, 'WSAEADDRINUSE', 10048), 10048}
_ACCESS = {errno.EACCES, getattr(errno, 'WSAEACCES', 10013), 10013}
_ADDR_NOT_AVAIL = {errno.EADDRNOTAVAIL, getattr(errno, 'WSAEADDRNOTAVAIL', 10049), 10049}


def prepare_console():
    """Never fail on a character the console cannot show (stderr already uses backslashreplace)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and stream.errors == 'strict':
                stream.reconfigure(errors='replace')
        except (AttributeError, ValueError, OSError):
            pass


def import_problem(module: str) -> str | None:
    """None when ``module`` can be imported, else why not (one line)."""
    try:
        __import__(module)
    except Exception as exc:  # ImportError, OSError (Windows DLL load failure), RuntimeError ...
        return f'{type(exc).__name__}: {exc}'.strip()
    return None


def missing_dependencies(need_torch: bool = True) -> dict:
    """{package: reason} for the packages the battle AI cannot import (numpy always, torch for the network)."""
    out = {}
    for name in ('numpy',) + (('torch',) if need_torch else ()):
        problem = import_problem(name)
        if problem:
            out[name] = problem
    return out


def _quote(path: str) -> str:
    return f'"{path}"' if (' ' in path or '(' in path or '&' in path) else path


def dependency_message(missing: dict, python: str | None = None, version=None, platform: str | None = None,
                       bits64: bool | None = None) -> str:
    """Korean explanation of missing / broken packages, with the exact command that installs them."""
    python = python or sys.executable or 'python'
    version = tuple(version or sys.version_info[:2])
    platform = platform or sys.platform
    bits64 = (sys.maxsize > 2 ** 32) if bits64 is None else bits64
    labels = {'torch': 'torch (PyTorch, 신경망 AI)', 'numpy': 'numpy'}
    exe = _quote(python)
    lines = ['', '[오류] 배틀 도우미의 AI 계산에 필요한 파이썬 패키지를 불러오지 못했습니다:']
    for name, why in missing.items():
        lines.append(f'  - {labels.get(name, name)}   ({why})')
    lines += ['', '해결 방법:',
              '  1) 다운로드한 폴더의 run_ui.bat (Windows) 또는 run_ui.sh (macOS/Linux) 로 실행하면',
              '     필요한 패키지를 자동으로 설치합니다.',
              '  2) 직접 설치하려면 이 명령을 실행한 뒤 다시 시작하세요:']
    if 'torch' in missing and platform.startswith('linux'):
        lines += [f'       {exe} -m pip install numpy',
                  f'       {exe} -m pip install torch --index-url {TORCH_CPU_INDEX}',
                  '     (CPU 전용 PyTorch: 다운로드가 훨씬 작습니다)']
    else:
        lines.append(f'       {exe} -m pip install {"numpy torch" if "torch" in missing else "numpy"}')
    if exe.startswith('"') and platform == 'win32':
        lines.append('     (PowerShell 에서는 명령 앞에 & 와 공백을 붙이세요)')
    if 'torch' in missing:
        if not bits64 or version < (3, 10) or version >= (3, 15):
            lines += ['  3) PyTorch는 64비트 Python 3.10 ~ 3.14 용으로 나옵니다. 지금 Python은 '
                      f'{version[0]}.{version[1]} ({"64" if bits64 else "32"}비트) 입니다.',
                      f'     {PYTHON_DOWNLOAD} 에서 Python 3.13 (64비트)을 설치한 뒤 다시 실행하세요.']
        lines += ['  신경망 없이 (더 약한 휴리스틱 AI로) 바로 쓰려면 --no-model 옵션을 붙여 실행하세요.']
    return '\n'.join(lines) + '\n'


def is_wildcard(host: str) -> bool:
    return host in ('', '0.0.0.0')


def is_loopback(host: str) -> bool:
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def local_url(host: str, port: int) -> str:
    """The address to open in a browser on this computer."""
    return f'http://{"127.0.0.1" if is_wildcard(host) else host}:{port}/'


def lan_addresses() -> list:
    """IPv4 addresses of this computer that other devices on the local network can use (may be empty).

    A UDP socket is "connected" to an outside address so the OS picks the outgoing interface; no packet is
    sent and nothing blocks. Any error (no network, no route) just gives fewer addresses.
    """
    found = []
    for target in ('10.255.255.255', '192.168.255.255', '8.8.8.8'):
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((target, 9))
            ip = s.getsockname()[0]
            addr = ipaddress.ip_address(ip)
            if not (addr.is_loopback or addr.is_unspecified or addr.is_link_local) and ip not in found:
                found.append(ip)
        except Exception:
            pass
        finally:
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
    return found


def port_error_message(host: str, port: int, exc: BaseException) -> str:
    """Korean explanation when the server cannot listen on host:port."""
    code = getattr(exc, 'winerror', None) or getattr(exc, 'errno', None)
    other = port + 1 if 0 < port < 65535 else 8766
    url = local_url(host, port)
    if code in _ADDR_IN_USE:
        return (f'\n[오류] 포트 {port}번을 이미 다른 프로그램이 쓰고 있습니다.\n'
                f'  배틀 도우미가 이미 다른 창에서 실행 중이라면 그 창을 그대로 두고 브라우저에서 이 주소를 여세요: {url}\n'
                f'  다른 포트로 실행하려면 --port {other} 옵션을 붙이세요 (예: run_ui.bat --port {other}).\n'
                f'  그 다음 브라우저에서 이 주소를 엽니다: {local_url(host, other)}\n')
    if code in _ACCESS:
        return (f'\n[오류] 포트 {port}번을 열 수 없습니다 (권한 없음: 운영체제가 예약했거나 막은 포트일 수 있습니다).\n'
                f'  다른 포트로 실행하세요: --port {other}  (예: run_ui.bat --port {other})\n')
    if code in _ADDR_NOT_AVAIL or isinstance(exc, socket.gaierror):
        return (f'\n[오류] 주소 {host} 에서 서버를 열 수 없습니다 ({exc}).\n'
                '  이 PC에서만 쓰려면 --host 옵션을 빼고, 휴대폰에서도 쓰려면 --host 0.0.0.0 을 쓰세요.\n')
    return (f'\n[오류] {host}:{port} 에서 서버를 열 수 없습니다 ({type(exc).__name__}: {exc}).\n'
            f'  다른 포트로 실행해 보세요: --port {other}\n')


def startup_banner(host: str, port: int, notes=(), lan=None) -> str:
    """The block printed once the server listens: where to open it and how to stop it."""
    line = '=' * 64
    out = [line, ' 포켓몬 챔피언스 배틀 도우미가 실행 중입니다.',
           f' 브라우저에서 이 주소를 여세요:   {local_url(host, port)}']
    if is_wildcard(host):
        lan = lan_addresses() if lan is None else lan
        if lan:
            for ip in lan:
                out.append(f' 휴대폰 (같은 Wi-Fi) 에서는:      http://{ip}:{port}/')
        else:
            out.append(' 휴대폰 (같은 Wi-Fi) 에서는 http://<이 PC의 IP 주소>:' + str(port) + '/ 를 여세요')
            out.append('   (IP 주소 확인: Windows는 명령 프롬프트에서 ipconfig, macOS/Linux는 ifconfig 또는 ip addr)')
        out.append('   Windows 방화벽 창이 뜨면 "허용"을 누르세요. 같은 네트워크의 누구나 접속할 수 있으니')
        out.append('   집처럼 믿을 수 있는 네트워크에서만 쓰세요.')
    elif not is_loopback(host):
        out.append('   같은 네트워크의 휴대폰에서도 이 주소로 접속할 수 있습니다 (방화벽 창이 뜨면 "허용").')
    for note in notes:
        out.append(f' {note}')
    out += [' 이 창은 켜 두세요. 창을 닫거나 Ctrl+C 를 누르면 배틀 도우미가 종료됩니다.', line]
    return '\n'.join(out)
