#!/bin/bash

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt

python experiment.py
python analysis.py
