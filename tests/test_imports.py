#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

try:
    from core.wordle_utils import load_words
    print("✓ Import successful!")

    # Test loading a file
    test_file = os.path.join('..', 'data', 'words_alpha5.txt')
    words = load_words(test_file)
    if words:
        print(f"✓ Successfully loaded {len(words)} words")
    else:
        print("✗ Failed to load words")

except ImportError as e:
    print(f"✗ Import failed: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
