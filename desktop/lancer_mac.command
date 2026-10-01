#!/bin/bash
# Double-clic dans le Finder : installe (1re fois) puis lance l'application.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi
./.venv/bin/python app_qt.py
