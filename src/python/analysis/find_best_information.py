#!/usr/bin/env python3
"""
Script to find the words with the highest information scores from the word list.
This calculates the information content for each word when used as a first guess.
Information score balances letter frequency analysis with diversity bonuses.
"""

import math
import os
import sys
from collections import Counter
from typing import List, Tuple

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.python.core.wordle_utils import get_feedback, calculate_entropy, has_unique_letters, load_words, filter_words_unique_letters, filter_wordle_appropriate, get_word_information_score

# Get the repository root directory (4 levels up from this file)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')


def find_best_information_words(word_list: List[str], top_n: int = 10, find_lowest: bool = False, calculate_entropy_upfront: bool = False, unique_letters_only: bool = True, wordle_appropriate_only: bool = True) -> List[Tuple[str, float, float, float]]:
    """Find the words with the highest (or lowest) information scores."""

    # Filter word list based on parameters
    if wordle_appropriate_only:
        word_list = filter_wordle_appropriate(word_list)
        print(f"Filtered to {len(word_list)} Wordle-appropriate words")

    if unique_letters_only:
        word_list = filter_words_unique_letters(word_list)
        print(f"Filtered to {len(word_list)} words with unique letters")

    print(f"Information analysis starting. Scoring {len(word_list)} words...")

    word_scores = []

    # For efficiency, we'll only calculate entropy for the top candidates if requested
    quick_scores = []

    for i, word in enumerate(word_list):
        if i % 500 == 0 and len(word_list) > 500:
            print(f"Progress: {i}/{len(word_list)} words processed...")

        # Calculate information score using the existing utility function
        info_score = get_word_information_score(word, word_list)

        # Calculate frequency-based likelihood for comparison (quick calculation)
        position_freqs = [Counter() for _ in range(5)]
        for w in word_list:
            for j, char in enumerate(w):
                position_freqs[j][char] += 1

        freq_score = sum(position_freqs[j][word[j]] for j in range(len(word)))
        likelihood_score = freq_score / len(word_list)

        quick_scores.append((word, info_score, likelihood_score, 0.0))

    # Sort by information score and get top candidates
    quick_scores.sort(key=lambda x: x[1], reverse=not find_lowest)
    top_candidates = quick_scores[:min(50, len(quick_scores))]  # Get top 50 for entropy calculation

    # Only calculate entropy for top candidates if explicitly requested
    if calculate_entropy_upfront:
        print(f"Calculating entropy for top {len(top_candidates)} candidates...")
        for i, (word, info_score, likelihood_score, _) in enumerate(top_candidates):
            if i % 10 == 0:
                print(f"Entropy progress: {i}/{len(top_candidates)}")
            entropy = calculate_entropy(word, word_list)
            word_scores.append((word, info_score, likelihood_score, entropy))
    else:
        word_scores = top_candidates

    return word_scores[:top_n]


def main():
    # Load word list from file
    word_list = load_words(os.path.join(DATA_DIR, "words_alpha5.txt"))
    if not word_list:
        print("Error: Word file not found at data/words_alpha5.txt")
        return
    print(f"Loaded {len(word_list)} words from file")

    # Find highest information score words
    print(f"\n{'='*80}")
    print("HIGHEST INFORMATION SCORE ANALYSIS (5-LETTER WORDS)")
    print(f"{'='*80}")

    top_words = find_best_information_words(word_list, top_n=20, find_lowest=False,
                                          unique_letters_only=False, wordle_appropriate_only=False)

    print(f"{'Rank':<4} {'Word':<8} {'Info Score':<12} {'Likelihood':<12} {'Entropy':<10}")
    print(f"{'-'*60}")

    for i, (word, info_score, likelihood_score, entropy) in enumerate(top_words, 1):
        print(f"{i:<4} {word:<8} {info_score:<12.4f} {likelihood_score:<12.4f} {entropy:<10.4f}")

    print(f"\nBest word for highest information score: '{top_words[0][0]}' with score {top_words[0][1]:.4f} (likelihood {top_words[0][2]:.4f})")

    # Find highest information score words (unique letters only)
    print(f"\n{'='*80}")
    print("HIGHEST INFORMATION SCORE ANALYSIS (UNIQUE LETTERS)")
    print(f"{'='*80}")

    top_words = find_best_information_words(word_list, top_n=20, find_lowest=False,
                                          unique_letters_only=True, wordle_appropriate_only=False)

    print(f"{'Rank':<4} {'Word':<8} {'Info Score':<12} {'Likelihood':<12} {'Entropy':<10}")
    print(f"{'-'*60}")

    for i, (word, info_score, likelihood_score, entropy) in enumerate(top_words, 1):
        print(f"{i:<4} {word:<8} {info_score:<12.4f} {likelihood_score:<12.4f} {entropy:<10.4f}")

    print(f"\nBest word for highest information score (unique letters): '{top_words[0][0]}' with score {top_words[0][1]:.4f} (likelihood {top_words[0][2]:.4f})")

    # Find lowest information score words (Wordle-appropriate, unique letters only)
    print(f"\n{'='*80}")
    print("LOWEST INFORMATION SCORE ANALYSIS (WORDLE-APPROPRIATE, UNIQUE LETTERS)")
    print(f"{'='*80}")

    bottom_words = find_best_information_words(word_list, top_n=20, find_lowest=True,
                                             unique_letters_only=True, wordle_appropriate_only=True)

    print(f"{'Rank':<4} {'Word':<8} {'Info Score':<12} {'Likelihood':<12} {'Entropy':<10}")
    print(f"{'-'*60}")

    for i, (word, info_score, likelihood_score, entropy) in enumerate(bottom_words, 1):
        print(f"{i:<4} {word:<8} {info_score:<12.4f} {likelihood_score:<12.4f} {entropy:<10.4f}")

    print(f"\nBest word for lowest information score: '{bottom_words[0][0]}' with score {bottom_words[0][1]:.4f} (likelihood {bottom_words[0][2]:.4f})")

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"Highest information score word: {top_words[0][0]} (score: {top_words[0][1]:.4f}, entropy: {top_words[0][3]:.4f})")
    print(f"Lowest information score word:  {bottom_words[0][0]} (score: {bottom_words[0][1]:.4f}, entropy: {bottom_words[0][3]:.4f})")

    # Find the word that appears in both entropy and information top lists
    # (Would need to run entropy analysis here for comparison, but this gives the basic structure)


if __name__ == "__main__":
    main()
