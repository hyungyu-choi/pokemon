#!/bin/sh
# Pokemon Champions battle assistant - start script for macOS and Linux.
#
#   ./run_ui.sh                   start it and open the browser at http://127.0.0.1:8765
#   ./run_ui.sh --host 0.0.0.0    phones on the same Wi-Fi can open it too (the address is printed)
#   ./run_ui.sh --port 8766       use another port
#   (or: sh run_ui.sh ...  /  bash run_ui.sh ...)
#
# The first run creates a Python environment in .venv next to this file and installs numpy and PyTorch
# into it (a few minutes; on Linux the CPU-only PyTorch). Later runs start right away.
# Optional: POKECHAMP_PYTHON=/path/to/python3 picks the Python, POKECHAMP_VENV=/path/to/env the environment.
set -e

cd "$(dirname "$0")"
HERE="$(pwd)"
VENV="${POKECHAMP_VENV:-$HERE/.venv}"
VPY="$VENV/bin/python"
MARKER="$VENV/pokechamp-deps-ok"
OS="$(uname -s)"
TORCH_CPU_INDEX="${POKECHAMP_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

fail() {
    printf '\n[오류] %s\n' "$1" >&2
    shift
    for line in "$@"; do printf '  %s\n' "$line" >&2; done
    printf '\n' >&2
    exit 1
}

# true for Python 3.10 ... 3.<$2>
py_ok() {
    "$1" -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, $2) else 1)" >/dev/null 2>&1
}

find_python() {
    if [ -n "${POKECHAMP_PYTHON:-}" ]; then
        PY="$POKECHAMP_PYTHON"
        py_ok "$PY" 99 && return 0
        fail "POKECHAMP_PYTHON=$PY 은(는) Python 3.10 이상이 아닙니다 (or it does not run)."
    fi
    max=14
    candidates="python3 python3.13 python3.12 python3.14 python3.11 python3.10 python"
    if [ "$OS" = Darwin ] && [ "$(uname -m)" = x86_64 ]; then
        # Intel Macs: the last PyTorch for them (2.2) supports Python 3.10 - 3.12 only
        max=12
        candidates="python3 python3.12 python3.11 python3.10 python"
    fi
    for c in $candidates; do
        if py_ok "$c" "$max"; then PY="$c"; return 0; fi
    done
    # a newer Python than PyTorch supports is still worth a try
    for c in python3 python3.15 python3.16 python; do
        if py_ok "$c" 99; then PY="$c"; return 0; fi
    done
    return 1
}

pip_install() {
    "$VPY" -m pip install --disable-pip-version-check "$@"
}

if [ ! -f "pokechamp/__main__.py" ]; then
    fail "pokechamp 폴더가 없습니다 (this script must stay in the downloaded project folder)." \
         "다운로드한 ZIP 을 모두 푼 폴더 안에서 run_ui.sh 를 실행하세요."
fi

if [ -f "$MARKER" ] && "$VPY" -c "import sys" >/dev/null 2>&1; then
    : # environment ready
else
    if [ -e "$VENV" ]; then
        [ -f "$VENV/pyvenv.cfg" ] || fail "$VENV 이(가) 이미 있지만 Python 환경이 아닙니다. 옮기거나 지운 뒤 다시 실행하세요."
        echo "Python 환경을 다시 만듭니다 (setting up again): $VENV"
        rm -rf "$VENV"
    fi
    PY=""
    if ! find_python; then
        fail "Python 3.10 이상을 찾지 못했습니다 (Python 3.10 or newer was not found)." \
             "macOS: https://www.python.org/downloads/ 에서 Python 3.13 을 설치한 뒤 다시 실행하세요." \
             "Linux: 패키지 관리자로 Python 3.10 이상과 venv 를 설치하세요 (예: sudo apt install python3 python3-venv)."
    fi
    echo "Python: $("$PY" -c 'import sys; print(sys.version.split()[0], sys.executable)')"
    if ! py_ok "$PY" 14; then
        echo "  참고: 이 Python 버전용 PyTorch 가 아직 없을 수 있습니다. 설치가 실패하면 Python 3.13 을 설치하세요."
    fi
    echo "Python 환경을 만드는 중 (creating the environment): $VENV"
    if ! "$PY" -m venv "$VENV"; then
        rm -rf "$VENV"
        fail "Python 환경(venv)을 만들지 못했습니다 (python -m venv failed)." \
             "Debian/Ubuntu: sudo apt install python3-venv  (또는 python3.X-venv) 를 설치한 뒤 다시 실행하세요."
    fi
    echo ""
    echo "numpy 와 PyTorch 를 설치합니다 (처음 한 번만, 몇 분 걸립니다 / first run only, a few minutes) ..."
    machine="$("$VPY" -c 'import platform; print(platform.machine())')"
    pip_failed() {
        fail "numpy / PyTorch 설치에 실패했습니다 (installing numpy / PyTorch failed; see the messages above)." \
             "인터넷 연결을 확인하고 다시 실행하세요." \
             "PyTorch 는 64비트 Python 3.10 ~ 3.14 용으로 나옵니다 (Intel Mac 은 3.12 까지). 지금 Python: $("$VPY" -c 'import sys; print(sys.version.split()[0])')" \
             "다른 버전이면 https://www.python.org/downloads/ 에서 Python 3.13 (Intel Mac 은 3.12) 을 설치한 뒤 다시 실행하세요."
    }
    if [ "$OS" = Darwin ] && [ "$machine" = x86_64 ]; then
        pip_install "numpy<2" torch || pip_failed       # Intel Mac: PyTorch 2.2 needs numpy 1.x
    elif [ "$OS" = Linux ]; then
        pip_install numpy || pip_failed
        # the CPU-only build (about 200 MB) instead of the default one with CUDA (several GB)
        if ! pip_install torch --index-url "$TORCH_CPU_INDEX"; then
            echo "CPU 전용 PyTorch 서버에 연결하지 못해 기본 서버(PyPI)에서 설치합니다 (다운로드가 큽니다) ..."
            pip_install torch || pip_failed
        fi
    else
        pip_install numpy torch || pip_failed
    fi
    if ! "$VPY" -c "import numpy, torch"; then
        fail "numpy / PyTorch 를 설치했지만 불러오지 못했습니다 (see the messages above)." \
             "다시 실행하면 처음부터 다시 설치합니다."
    fi
    touch "$MARKER"
    echo "설치 완료 (installation finished)."
fi

# open the browser only where there is a desktop (not over ssh / in a text console)
if [ "$OS" = Darwin ] || [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    exec "$VPY" -m pokechamp ui --open "$@"
fi
exec "$VPY" -m pokechamp ui "$@"
