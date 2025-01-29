#!/bin/bash

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running experiment.py..."
python experiment.py

echo "Running analysis.py..."
python analysis.py

echo "All tasks completed."
