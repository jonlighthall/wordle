#!/usr/bin/env python3
"""
Script to find the word with the highest entropy from the word list.
This calculates the entropy for each word when used as a first guess.
"""

import math
from collections import Counter
from typing import List, Tuple
from wordle_utils import get_feedback, calculate_entropy, load_words


def find_best_entropy_words(word_list: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
    """Find the words with the highest entropy values."""
    print(f"Calculating entropy for {len(word_list)} words...")
    print("This may take a few minutes...")

    word_entropies = []

    for i, guess in enumerate(word_list):
        if i % 1000 == 0:
            print(f"Progress: {i}/{len(word_list)} words processed...")

        entropy = calculate_entropy(guess, word_list)
        word_entropies.append((guess, entropy))

    # Sort by entropy (highest first)
    word_entropies.sort(key=lambda x: x[1], reverse=True)

    return word_entropies[:top_n]

def main():
    # Load word list from file
    word_list = load_words("/home/jlighthall/examp/wordle/data/words_alpha5_100.txt")
    if not word_list:
        print("Error: Word file not found at /home/jlighthall/examp/wordle/data/words_alpha5.txt")
        return
    print(f"Loaded {len(word_list)} words from file")

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
