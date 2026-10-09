#!/bin/sh
# Pokemon Champions battle assistant - start script for macOS and Linux.
#
#   ./run_ui.sh                   start it and open the browser at http://127.0.0.1:8765
#   ./run_ui.sh --host 0.0.0.0    phones on the same Wi-Fi can open it too (the address is printed)
#   ./run_ui.sh --port 8766       use another port
#   (or: sh run_ui.sh ...  /  bash run_ui.sh ...)
#
# The first run creates a Python environment in .venv next to this file and installs numpy and PyTorch
# into it (a few minutes; on Linux the CPU-only PyTorch). Later runs start right away. When PyTorch cannot be
# installed (no PyTorch for this Python or computer), it starts with the weaker heuristic AI instead and tries
# PyTorch again on the next run.
# Optional: POKECHAMP_PYTHON=/path/to/python3 picks the Python, POKECHAMP_VENV=/path/to/env the environment
# (a folder this script did not create is never deleted; the packages are installed into it).
set -e

cd "$(dirname "$0")"
HERE="$(pwd)"
VENV="${POKECHAMP_VENV:-$HERE/.venv}"
VPY="$VENV/bin/python"
MARKER="$VENV/pokechamp-deps-ok"      # numpy and PyTorch work
NUMPY_OK="$VENV/pokechamp-numpy-ok"   # numpy works (PyTorch maybe not)
TAG="$VENV/pokechamp-venv"            # this script created the environment (only then may it delete it)
OS="$(uname -s)"
TORCH_CPU_INDEX="${POKECHAMP_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
NO_MODEL=""

# the newest Python minor version PyTorch supports, and the Pythons to look for (best first)
TORCH_MAX=14
CANDIDATES="python3 python3.13 python3.12 python3.14 python3.11 python3.10 python"
if [ "$OS" = Darwin ] && [ "$(uname -m)" = x86_64 ]; then
    # Intel Macs: the last PyTorch for them (2.2) supports Python 3.10 - 3.12 only
    TORCH_MAX=12
    CANDIDATES="python3 python3.12 python3.11 python3.10 python"
fi

fail() {
    printf '\n[오류] %s\n' "$1" >&2
    shift
    for line in "$@"; do printf '  %s\n' "$line" >&2; done
    printf '\n' >&2
    exit 1
}

# true for Python 3.10 ... 3.<$2>; unless $2 is 99 it must also be 64-bit (what PyTorch needs)
py_ok() {
    "$1" -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, $2) and ($2 == 99 or sys.maxsize > 2 ** 32) else 1)" >/dev/null 2>&1
}

# sets PY, and PY_FITS=1 when PyTorch supports that Python
find_python() {
    PY_FITS=""
    if [ -n "${POKECHAMP_PYTHON:-}" ]; then
        PY="$POKECHAMP_PYTHON"
        py_ok "$PY" 99 || fail "POKECHAMP_PYTHON=$PY 은(는) Python 3.10 이상이 아닙니다 (or it does not run)."
        if py_ok "$PY" "$TORCH_MAX"; then PY_FITS=1; fi
        return 0
    fi
    for c in $CANDIDATES; do
        if py_ok "$c" "$TORCH_MAX"; then PY="$c"; PY_FITS=1; return 0; fi
    done
    # a newer Python than PyTorch supports still runs the heuristic AI
    for c in python3 python3.15 python3.16 python; do
        if py_ok "$c" 99; then PY="$c"; return 0; fi
    done
    return 1
}

no_python() {
    fail "Python 3.10 이상을 찾지 못했습니다 (Python 3.10 or newer was not found)." \
         "macOS: https://www.python.org/downloads/ 에서 Python 3.13 을 설치한 뒤 다시 실행하세요." \
         "Linux: 패키지 관리자로 Python 3.10 이상과 venv 를 설치하세요 (예: sudo apt install python3 python3-venv)."
}

venv_runs() {
    "$VPY" -c "import sys" >/dev/null 2>&1
}

venv_has() {
    "$VPY" -c "import $1" >/dev/null 2>&1
}

pip_install() {
    "$VPY" -m pip install --disable-pip-version-check "$@"
}

# PyTorch: after a failed earlier attempt, one retry with a short timeout (offline / blocked network)
pip_torch() {
    if [ -n "$quick" ]; then
        pip_install --retries 1 --timeout 10 "$@"
    else
        pip_install "$@"
    fi
}

if [ ! -f "pokechamp/__main__.py" ]; then
    fail "pokechamp 폴더가 없습니다 (this script must stay in the downloaded project folder)." \
         "다운로드한 ZIP 을 모두 푼 폴더 안에서 run_ui.sh 를 실행하세요."
fi

if [ -f "$MARKER" ] && venv_runs; then
    : # environment ready
else
    if [ -e "$VENV" ] && [ ! -f "$VENV/pyvenv.cfg" ]; then
        fail "$VENV 이(가) 이미 있지만 Python 환경이 아닙니다. 옮기거나 지운 뒤 다시 실행하세요."
    fi
    if [ -e "$VENV" ] && [ ! -f "$TAG" ]; then
        # not created by this script (POKECHAMP_VENV, the README's commands ...): never deleted, installed into
        if ! venv_has "numpy, torch" && ! "$VPY" -m pip --version >/dev/null 2>&1; then
            fail "$VENV 의 Python 또는 pip 가 동작하지 않습니다 (this environment does not work)." \
                 "이 스크립트가 만든 폴더가 아니라서 지우지 않습니다. 직접 옮기거나 지운 뒤 다시 실행하세요." \
                 "(or set POKECHAMP_VENV to another folder)"
        fi
        echo "이미 있는 Python 환경을 사용합니다 (using the existing environment): $VENV"
    else
        PY=""
        find_python || PY=""
        if [ -e "$VENV" ]; then
            # created by this script: kept when only PyTorch is missing, unless PyTorch does not support its
            # Python but supports a Python installed now; else (unfinished or broken) it is made again
            if venv_runs && [ -f "$NUMPY_OK" ] && venv_has numpy && { [ -z "$PY_FITS" ] || py_ok "$VPY" "$TORCH_MAX"; }; then
                : # install PyTorch again below
            else
                [ -n "$PY" ] || no_python
                # not while a battle assistant started from it is still running (it would lose its packages)
                if command -v pgrep >/dev/null 2>&1 && pgrep -f -- "$VENV/bin/python" >/dev/null 2>&1; then
                    fail "$VENV 의 Python 이 아직 실행 중입니다 (a battle assistant from it is still running)." \
                         "다른 창에서 실행 중인 배틀 도우미를 먼저 닫은 뒤 다시 실행하세요."
                fi
                echo "Python 환경을 다시 만듭니다 (setting up again): $VENV"
                rm -rf "$VENV"
            fi
        fi
        if [ ! -e "$VENV" ]; then
            [ -n "$PY" ] || no_python
            echo "Python: $("$PY" -c 'import sys; print(sys.version.split()[0], sys.executable)')"
            if [ -z "$PY_FITS" ]; then
                echo "  참고: 이 Python 용 PyTorch 가 없을 수 있습니다. 그러면 신경망 없이 (휴리스틱 AI 로) 시작합니다."
            fi
            echo "Python 환경을 만드는 중 (creating the environment): $VENV"
            if ! "$PY" -m venv "$VENV"; then
                rm -rf "$VENV"   # did not exist before: only what this failed command left behind
                fail "Python 환경(venv)을 만들지 못했습니다 (python -m venv failed)." \
                     "Debian/Ubuntu: sudo apt install python3-venv  (또는 python3.X-venv) 를 설치한 뒤 다시 실행하세요."
            fi
            echo "created by run_ui.sh" > "$TAG"
        fi
    fi

    machine="$("$VPY" -c 'import platform; print(platform.machine())')" || machine=""
    torch_ok=""
    quick=""
    if venv_has "numpy, torch"; then
        torch_ok=1
    else
        if [ -f "$NUMPY_OK" ] && venv_has numpy; then
            quick=1   # numpy is there from an earlier run, so PyTorch failed then: do not wait long for it again
        else
            echo ""
            echo "numpy 를 설치합니다 (installing numpy) ..."
            numpy_req=numpy
            if [ "$OS" = Darwin ] && [ "$machine" = x86_64 ]; then
                numpy_req="numpy<2"   # Intel Mac: PyTorch 2.2 needs numpy 1.x
            fi
            pip_install "$numpy_req" || fail "numpy 설치에 실패했습니다 (installing numpy failed; see the messages above)." \
                                             "인터넷 연결을 확인하고 다시 실행하세요."
            venv_has numpy || fail "numpy 를 설치했지만 불러오지 못했습니다 (see the messages above)." \
                                   "다시 실행하면 다시 설치합니다."
        fi
        echo "numpy ok" > "$NUMPY_OK"
        echo ""
        echo "PyTorch (신경망 AI) 를 설치합니다 (몇 분 걸릴 수 있습니다 / installing PyTorch, may take a few minutes) ..."
        if [ "$OS" = Linux ]; then
            # the CPU-only build (about 200 MB) instead of the default one with CUDA (several GB)
            if pip_torch torch --index-url "$TORCH_CPU_INDEX"; then
                torch_ok=1
            else
                echo "CPU 전용 PyTorch 를 받지 못해 기본 서버(PyPI)에서 설치해 봅니다 (다운로드가 큽니다) ..."
                if pip_torch torch; then torch_ok=1; fi
            fi
        elif pip_torch torch; then
            torch_ok=1
        fi
        if [ -n "$torch_ok" ] && ! venv_has torch; then
            "$VPY" -c "import torch" || true   # show why it cannot be loaded
            torch_ok=""
        fi
    fi
    if [ -n "$torch_ok" ]; then
        echo "numpy ok" > "$NUMPY_OK"
        echo "numpy and torch ok" > "$MARKER"
        echo "설치 완료 (installation finished)."
    else
        # no full marker: the next run tries PyTorch again (pip stops quickly when there is no PyTorch for it)
        NO_MODEL=1
        printf '\n[참고] PyTorch 를 쓸 수 없어 신경망 없이 시작합니다 (PyTorch could not be installed or loaded: starting WITHOUT the neural network).\n'
        echo "  더 약한 휴리스틱 AI 로 계산합니다 (the heuristic AI is weaker). 이유는 위의 메시지를 보세요."
        echo "  이 Python (this Python): $("$VPY" -c 'import sys, struct; print("%s, %d-bit" % (sys.version.split()[0], struct.calcsize("P") * 8))'), $machine"
        echo "  신경망 AI 를 쓰려면 (for the full AI):"
        echo "  - 인터넷 연결을 확인하고 다시 실행하세요. PyTorch 설치는 실행할 때마다 다시 시도합니다."
        if ! py_ok "$VPY" "$TORCH_MAX"; then
            if [ "$TORCH_MAX" = 12 ]; then
                echo "  - PyTorch 는 64비트 Python 3.10 ~ 3.12 용입니다 (Intel Mac). https://www.python.org/downloads/ 에서"
                echo "    Python 3.12 를 설치한 뒤 다시 실행하세요."
            else
                echo "  - PyTorch 는 64비트 Python 3.10 ~ 3.14 용입니다. https://www.python.org/downloads/ 에서"
                echo "    Python 3.13 (64비트) 을 설치한 뒤 다시 실행하세요."
            fi
            if [ -f "$TAG" ]; then
                echo "    (그러면 $VENV 를 그 Python 으로 다시 만듭니다 / the environment is then made again)"
            else
                echo "    ($VENV 는 이 스크립트가 만든 환경이 아니라서 지우지 않습니다. 이 명령으로 지운 뒤 다시 실행하세요"
                echo "     / this script did not create it; delete it, then run this script again:)"
                echo "      rm -rf \"$VENV\""
            fi
        fi
        echo ""
    fi
fi

if [ -n "$NO_MODEL" ]; then
    set -- --no-model "$@"
fi
# open the browser only where there is a desktop (not over ssh / in a text console)
if [ "$OS" = Darwin ] || [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    exec "$VPY" -m pokechamp ui --open "$@"
fi
exec "$VPY" -m pokechamp ui "$@"
