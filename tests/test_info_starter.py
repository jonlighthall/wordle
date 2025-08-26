#!/usr/bin/env python3
"""
Quick test to find a good information-based starting word from a small set of candidates.
"""

import os
import sys

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'python'))

from core.wordle_utils import load_words, get_word_information_score

# Get the repository root directory (2 levels up from this file)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')

def test_information_starters():
    """Test a few candidate words for information score."""

    # Load word list
    word_list = load_words(os.path.join(DATA_DIR, "words_alpha5.txt"))
    if not word_list:
        print(f"Error: Word file not found at {os.path.join(DATA_DIR, 'words_alpha5.txt')}")
        # Try alternative path
        alt_path = os.path.join(os.getcwd(), "data", "words_alpha5.txt")
        word_list = load_words(alt_path)
        if not word_list:
            print(f"Error: Alternative path also failed: {alt_path}")
            return
        else:
            print(f"Using alternative path: {alt_path}")

    print(f"Loaded {len(word_list)} words")

    # Test some common starting words
    candidates = ["slate", "adieu", "audio", "raise", "arise", "tares", "cares", "roate", "crane", "stare", "tears", "rates"]

    print(f"Testing information scores for common starting words using {len(word_list)} words...")
    print(f"{'Word':<8} {'Info Score':<12}")
    print("-" * 20)

    results = []
    for word in candidates:
        if word in word_list:
            info_score = get_word_information_score(word, word_list)
            results.append((word, info_score))
            print(f"{word:<8} {info_score:<12.4f}")

    # Find the best one
    best_word, best_score = max(results, key=lambda x: x[1])
    print(f"\nBest information starter: '{best_word}' with score {best_score:.4f}")

    return best_word

if __name__ == "__main__":
    test_information_starters()
