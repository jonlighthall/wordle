#!/usr/bin/env python3
"""
Script to find the word with the highest entropy from the word list.
This calculates the entropy for each word when used as a first guess.
"""

import math
import os
import sys
from collections import Counter
from typing import List, Tuple

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.wordle_utils import get_feedback, calculate_entropy, load_words
from core.common_utils import DATA_DIR, ProgressReporter, load_word_list_with_fallback


def find_best_entropy_words(word_list: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
    """Find the words with the highest entropy values."""
    print(f"📊 Calculating entropy for {len(word_list)} words...")
    print("   This may take a few minutes...")

    word_entropies = []
    total_words = len(word_list)
    
    # Use common progress reporting
    progress_reporter = ProgressReporter(total_words, report_interval=10)

    for i, guess in enumerate(word_list):
        progress_reporter.report_progress(i, "words")

        entropy = calculate_entropy(guess, word_list)
        word_entropies.append((guess, entropy))

    progress_reporter.final_report("words")

    # Sort by entropy (highest first)
    word_entropies.sort(key=lambda x: x[1], reverse=True)

    return word_entropies[:top_n]

def main():
    """Main function to find best entropy words."""
    print("🔍 Entropy-Based First Guess Analysis")
    print("=" * 60)
    
    # Load word list using common utilities
    word_list = load_word_list_with_fallback("words_alpha5_100.txt", ["words_alpha5.txt"])
    if not word_list:
        return

    # Find top entropy words
    top_words = find_best_entropy_words(word_list, top_n=20)

    print(f"\n{'='*60}")
    print("TOP 20 WORDS BY ENTROPY (Best First Guesses)")
    print(f"{'='*60}")
    print(f"{'Rank':<4} {'Word':<8} {'Entropy':<10}")
    print(f"{'-'*25}")

    for i, (word, entropy) in enumerate(top_words, 1):
        print(f"{i:<4} {word:<8} {entropy:<10.4f}")

    print(f"\nBest word for first guess: '{top_words[0][0]}' with entropy {top_words[0][1]:.4f}")

if __name__ == "__main__":
    main()
