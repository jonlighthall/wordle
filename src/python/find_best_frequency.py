#!/usr/bin/env python3
"""
Script to find the words with the highest and lowest frequency scores from the word list.
This calculates the letter frequency for each position and scores words accordingly.
"""

import math
from collections import Counter
from typing import List, Tuple
from wordle_utils import get_feedback, calculate_entropy, has_unique_letters, load_words, filter_words_unique_letters, filter_wordle_appropriate


def calculate_frequency_score(word: str, word_list: List[str]) -> Tuple[int, float]:
    """Calculate frequency score for a word based on letter frequencies in each position."""
    word_length = len(word)
    total_words = len(word_list)

    # Calculate letter frequencies for each position
    freq = [Counter() for _ in range(word_length)]

    for w in word_list:
        for i, char in enumerate(w):
            freq[i][char] += 1

    # Calculate raw frequency score and normalized likelihood score
    freq_score = sum(freq[i][word[i]] for i in range(word_length))
    likelihood_score = freq_score / total_words

    return freq_score, likelihood_score

def find_best_frequency_words(word_list: List[str], top_n: int = 10, find_lowest: bool = False, calculate_entropy_upfront: bool = False, unique_letters_only: bool = True, wordle_appropriate_only: bool = True) -> List[Tuple[str, int, float, float]]:
    """Find the words with the highest (or lowest) frequency scores."""
    analysis_type = "lowest" if find_lowest else "highest"
    original_count = len(word_list)

    # Filter to Wordle-appropriate words (no plurals, past tense, etc.) if requested
    if wordle_appropriate_only:
        word_list = filter_wordle_appropriate(word_list)
        filtered_count = len(word_list)
        print(f"Filtered to {filtered_count} Wordle-appropriate words from {original_count} total words")
        original_count = filtered_count

    # Filter to isograms (words with unique letters only) if requested
    if unique_letters_only:
        word_list = filter_words_unique_letters(word_list)
        filtered_count = len(word_list)
        if wordle_appropriate_only:
            print(f"Further filtered to {filtered_count} words with unique letters (isograms)")
        else:
            print(f"Filtered to {filtered_count} words with unique letters (isograms) from {original_count} total words")

    print(f"Calculating {analysis_type} frequency scores for {len(word_list)} words...")

    # OPTIMIZED: Calculate frequencies once for the entire word list
    word_length = 5
    freq = [Counter() for _ in range(word_length)]

    for word in word_list:
        for i, char in enumerate(word):
            freq[i][char] += 1

    print(f"Frequency analysis complete. Now scoring {len(word_list)} words...")

    word_scores = []

    for i, word in enumerate(word_list):
      #  if i % 1000 == 0 and len(word_list) > 1000:
       #     print(f"Progress: {i}/{len(word_list)} words processed...")

        # OPTIMIZED: Simple arithmetic using pre-calculated frequencies
        freq_score = sum(freq[i][word[i]] for i in range(word_length))
        likelihood_score = freq_score / len(word_list)

        # Only calculate entropy if explicitly requested (much slower)
        if calculate_entropy_upfront:
            entropy = calculate_entropy(word, word_list)
        else:
            entropy = 0.0  # Placeholder - will calculate only for top words if needed

        word_scores.append((word, freq_score, likelihood_score, entropy))

    # Sort by frequency score (lowest first if find_lowest, highest first otherwise)
    word_scores.sort(key=lambda x: x[1], reverse=not find_lowest)

    # If entropy wasn't calculated upfront, calculate it only for the top words
    if not calculate_entropy_upfront:
        print(f"Calculating entropy for top {top_n} words...")
        top_words = word_scores[:top_n]
        for i, (word, freq_score, likelihood_score, _) in enumerate(top_words):
            entropy = calculate_entropy(word, word_list)
            word_scores[i] = (word, freq_score, likelihood_score, entropy)

    return word_scores[:top_n]

def main():
    # Load word list from file
    word_list = load_words("/home/jlighthall/examp/common/words_alpha5.txt")
    if not word_list:
        print("Error: Word file not found at /home/jlighthall/examp/common/words_alpha5.txt")
        return
    print(f"Loaded {len(word_list)} words from file")

    # Find highest frequency words
    print(f"\n{'='*80}")
    print("HIGHEST FREQUENCY ANALYSIS (5-LETTER WORDS)")
    print(f"{'='*80}")

    top_words = find_best_frequency_words(word_list, top_n=20, find_lowest=False,
                                         unique_letters_only=False, wordle_appropriate_only=False)

    print(f"{'Rank':<4} {'Word':<8} {'Freq Score':<10} {'Likelihood':<12} {'Entropy':<10}")
    print(f"{'-'*55}")

    for i, (word, freq_score, likelihood_score, entropy) in enumerate(top_words, 1):
        print(f"{i:<4} {word:<8} {freq_score:<10} {likelihood_score:<12.4f} {entropy:<10.4f}")

    print(f"\nBest word for highest frequency: '{top_words[0][0]}' with score {top_words[0][1]} (likelihood {top_words[0][2]:.4f})")

    # Find highest frequency words (unique letters only)
    print(f"\n{'='*80}")
    print("HIGHEST FREQUENCY ANALYSIS (UNIQUE LETTERS)")
    print(f"{'='*80}")

    top_words = find_best_frequency_words(word_list, top_n=20, find_lowest=False,
                                         unique_letters_only=True, wordle_appropriate_only=False)

    print(f"{'Rank':<4} {'Word':<8} {'Freq Score':<10} {'Likelihood':<12} {'Entropy':<10}")
    print(f"{'-'*55}")

    for i, (word, freq_score, likelihood_score, entropy) in enumerate(top_words, 1):
        print(f"{i:<4} {word:<8} {freq_score:<10} {likelihood_score:<12.4f} {entropy:<10.4f}")

    print(f"\nBest word for highest frequency: '{top_words[0][0]}' with score {top_words[0][1]} (likelihood {top_words[0][2]:.4f})")

    # Find highest frequency words (Wordle-appropriate, unique letters only)
    print(f"\n{'='*80}")
    print("HIGHEST FREQUENCY ANALYSIS (WORDLE-APPROPRIATE, UNIQUE LETTERS)")
    print(f"{'='*80}")

    top_words = find_best_frequency_words(word_list, top_n=20, find_lowest=False,
                                         unique_letters_only=True, wordle_appropriate_only=True)

    print(f"{'Rank':<4} {'Word':<8} {'Freq Score':<10} {'Likelihood':<12} {'Entropy':<10}")
    print(f"{'-'*55}")

    for i, (word, freq_score, likelihood_score, entropy) in enumerate(top_words, 1):
        print(f"{i:<4} {word:<8} {freq_score:<10} {likelihood_score:<12.4f} {entropy:<10.4f}")

    print(f"\nBest word for highest frequency: '{top_words[0][0]}' with score {top_words[0][1]} (likelihood {top_words[0][2]:.4f})")

    # Find lowest frequency words (Wordle-appropriate, unique letters only)
    print(f"\n{'='*80}")
    print("LOWEST FREQUENCY ANALYSIS (WORDLE-APPROPRIATE, UNIQUE LETTERS)")
    print(f"{'='*80}")

    bottom_words = find_best_frequency_words(word_list, top_n=20, find_lowest=True,
                                           unique_letters_only=True, wordle_appropriate_only=True)

    print(f"{'Rank':<4} {'Word':<8} {'Freq Score':<10} {'Likelihood':<12} {'Entropy':<10}")
    print(f"{'-'*55}")

    for i, (word, freq_score, likelihood_score, entropy) in enumerate(bottom_words, 1):
        print(f"{i:<4} {word:<8} {freq_score:<10} {likelihood_score:<12.4f} {entropy:<10.4f}")

    print(f"\nBest word for lowest frequency: '{bottom_words[0][0]}' with score {bottom_words[0][1]} (likelihood {bottom_words[0][2]:.4f})")

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"Highest frequency word: {top_words[0][0]} (score: {top_words[0][1]}, entropy: {top_words[0][3]:.4f})")
    print(f"Lowest frequency word:  {bottom_words[0][0]} (score: {bottom_words[0][1]}, entropy: {bottom_words[0][3]:.4f})")

    # Find the word that appears in both entropy and frequency top lists
    # (Would need to run entropy analysis here for comparison, but this gives the basic structure)

if __name__ == "__main__":
    main()
