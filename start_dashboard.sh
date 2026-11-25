#!/usr/bin/env bash
# One-click start for macOS/Linux: creates a virtual environment on first run, installs deps, starts the dashboard.
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install -q -r requirements.txt
( sleep 2; (command -v open >/dev/null && open http://127.0.0.1:8000) || (command -v xdg-open >/dev/null && xdg-open http://127.0.0.1:8000) || true ) &
python scripts/serve.py
