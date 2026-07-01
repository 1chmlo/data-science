#!/usr/bin/env bash
# Crea un entorno virtual reproducible para el pipeline de forecasting.
#   uso:  bash setup_env.sh
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
echo "OK. Activa el entorno con:  source .venv/bin/activate"
