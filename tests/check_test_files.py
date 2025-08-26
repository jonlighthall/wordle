#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))
from core.wordle_utils import load_words

# Check that all the test files exist
files = ['words_past5_date.txt', 'words_missing.txt', 'words_challenging.txt']
for f in files:
    path = os.path.join('..', 'data', f)
    words = load_words(path)
    if words:
        print(f'✓ {f}: {len(words)} words')
    else:
        print(f'✗ {f}: not found or empty')
