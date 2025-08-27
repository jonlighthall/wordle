#!/bin/bash
# Convenience script to run Wordle with the virtual environment
cd "$(dirname "$0")"
.venv/bin/python run_wordle.py "$@"
