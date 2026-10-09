#!/bin/sh
# macOS: double-click this file in Finder to start the battle assistant (it runs run_ui.sh in Terminal).
# The first time, macOS may refuse to open a downloaded file: right-click (Control-click) it > Open > Open.
cd "$(dirname "$0")" || exit 1
exec /bin/sh ./run_ui.sh "$@"
