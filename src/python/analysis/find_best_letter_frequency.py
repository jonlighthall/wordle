#!/usr/bin/env python3
"""
Script to find the words with the highest and lowest letter frequency scores from the word list.
This calculates the letter frequency for each position and scores words accordingly.
"""

import math
import os
import sys
from collections import Counter
from typing import List, Tuple

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.wordle_utils import get_feedback, calculate_entropy, has_unique_letters, load_words, filter_words_unique_letters, filter_wordle_appropriate
from core.common_utils import DATA_DIR, ProgressReporter, load_word_list_with_fallback


def calculate_frequency_score(word: str, word_list: List[str]) -> Tuple[int, float]:
    """Calculate frequency score for a word based on letter frequencies in each position."""
    word_length = len(word)
    total_words = len(word_list)

    # Calculate letter frequencies for each position
    freq = [Counter() for _ in range(word_length)]

    for w in word_list:
        for i, char in enumerate(w):
            freq[i][char] += 1

    # After calculating frequencies, show the most likely letter in each position
    most_likely_letters = []
    print(f"\nMost likely letters by position:")
    for i in range(word_length):
        if freq[i]:  # Check if position has any letters
            most_common_letter = freq[i].most_common(1)[0][0]
            most_likely_letters.append(most_common_letter)
            print(f"Position {i+1}: '{most_common_letter}' (appears {freq[i][most_common_letter]} times)")
        else:
            most_likely_letters.append('?')
            print(f"Position {i+1}: No data")

    # Create word from most likely letters and check if it exists
    most_likely_word = ''.join(most_likely_letters)
    print(f"\nPattern formed from most likely letters: '{most_likely_word}'")
    if most_likely_word in word_list:
        print(f"✓ '{most_likely_word}' is a valid word in the list!")
    else:
        print(f"✗ '{most_likely_word}' is not in the word list")

    # Calculate raw frequency score and normalized likelihood score
    freq_score = sum(freq[i][word[i]] for i in range(word_length))
    likelihood_score = freq_score / total_words

    return freq_score, likelihood_score

def find_best_letter_frequency_words(word_list: List[str], top_n: int = 10, find_lowest: bool = False, calculate_entropy_upfront: bool = False, unique_letters_only: bool = True, wordle_appropriate_only: bool = True) -> List[Tuple[str, int, float, float]]:
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

    print(f"📊 Calculating {analysis_type} frequency scores for {len(word_list)} words...")

    # OPTIMIZED: Calculate frequencies once for the entire word list
    word_length = 5
    freq = [Counter() for _ in range(word_length)]

    for word in word_list:
        for i, char in enumerate(word):
            freq[i][char] += 1

    print(f"   Letter frequency analysis complete. Now scoring {len(word_list)} words...")

    word_scores = []
    total_words = len(word_list)

    # Use common progress reporting
    progress_reporter = ProgressReporter(total_words, report_interval=10)

    for i, word in enumerate(word_list):
        progress_reporter.report_progress(i, "words")

        # OPTIMIZED: Simple arithmetic using pre-calculated frequencies
        freq_score = sum(freq[i][word[i]] for i in range(word_length))
        likelihood_score = freq_score / len(word_list)

        # Only calculate entropy if explicitly requested (much slower)
        if calculate_entropy_upfront:
            entropy = calculate_entropy(word, word_list)
        else:
            entropy = 0.0  # Placeholder - will calculate only for top words if needed

        word_scores.append((word, freq_score, likelihood_score, entropy))

    progress_reporter.final_report("words")

    # Sort by frequency score (lowest first if find_lowest, highest first otherwise)
    word_scores.sort(key=lambda x: x[1], reverse=not find_lowest)

    # If entropy wasn't calculated upfront, calculate it only for the top words
    if not calculate_entropy_upfront:
        print(f"\nCalculating entropy for top {top_n} words...")
        top_words = word_scores[:top_n]
        for i, (word, freq_score, likelihood_score, _) in enumerate(top_words):
            entropy = calculate_entropy(word, word_list)
            word_scores[i] = (word, freq_score, likelihood_score, entropy)

    return word_scores[:top_n]

def main():
    """Main function to analyze letter frequencies."""
    print("🔍 Letter Frequency Analysis")
    print("=" * 70)

    # Load word list using common utilities
    word_list = load_word_list_with_fallback("words_alpha5.txt", ["words_alpha5_100.txt"])
    if not word_list:
        return

    # Print most likely letters by position
    calculate_frequency_score(word_list[0], word_list)

    # Find highest letter frequency words
    print(f"\n{'='*80}")
    print("HIGHEST LETTER FREQUENCY ANALYSIS (5-LETTER WORDS)")
    print(f"{'='*80}")

    top_words = find_best_letter_frequency_words(word_list, top_n=20, find_lowest=False,
                                                 unique_letters_only=False, wordle_appropriate_only=False)

    print(f"{'Rank':<4} {'Word':<8} {'Freq Score':<10} {'Likelihood':<12} {'Entropy':<10}")
    print(f"{'-'*55}")

    for i, (word, freq_score, likelihood_score, entropy) in enumerate(top_words, 1):
        print(f"{i:<4} {word:<8} {freq_score:<10} {likelihood_score:<12.4f} {entropy:<10.4f}")

    print(f"\nBest word for highest frequency: '{top_words[0][0]}' with score {top_words[0][1]} (likelihood {top_words[0][2]:.4f})")

    # Find highest letter frequency words (unique letters only)
    print(f"\n{'='*80}")
    print("HIGHEST LETTER FREQUENCY ANALYSIS (UNIQUE LETTERS)")
    print(f"{'='*80}")

    top_words = find_best_letter_frequency_words(word_list, top_n=20, find_lowest=False,
                                                 unique_letters_only=True, wordle_appropriate_only=False)

    print(f"{'Rank':<4} {'Word':<8} {'Freq Score':<10} {'Likelihood':<12} {'Entropy':<10}")
    print(f"{'-'*55}")

    for i, (word, freq_score, likelihood_score, entropy) in enumerate(top_words, 1):
        print(f"{i:<4} {word:<8} {freq_score:<10} {likelihood_score:<12.4f} {entropy:<10.4f}")

    print(f"\nBest word for highest frequency: '{top_words[0][0]}' with score {top_words[0][1]} (likelihood {top_words[0][2]:.4f})")

    # Find highest letter frequency words (Wordle-appropriate, unique letters only)
    print(f"\n{'='*80}")
    print("HIGHEST LETTER FREQUENCY ANALYSIS (WORDLE-APPROPRIATE, UNIQUE LETTERS)")
    print(f"{'='*80}")

    top_words = find_best_letter_frequency_words(word_list, top_n=20, find_lowest=False,
                                                 unique_letters_only=True, wordle_appropriate_only=True)

    print(f"{'Rank':<4} {'Word':<8} {'Freq Score':<10} {'Likelihood':<12} {'Entropy':<10}")
    print(f"{'-'*55}")

    for i, (word, freq_score, likelihood_score, entropy) in enumerate(top_words, 1):
        print(f"{i:<4} {word:<8} {freq_score:<10} {likelihood_score:<12.4f} {entropy:<10.4f}")

    print(f"\nBest word for highest frequency: '{top_words[0][0]}' with score {top_words[0][1]} (likelihood {top_words[0][2]:.4f})")

    # Find lowest letter frequency words (Wordle-appropriate, unique letters only)
    print(f"\n{'='*80}")
    print("LOWEST LETTER FREQUENCY ANALYSIS (WORDLE-APPROPRIATE, UNIQUE LETTERS)")
    print(f"{'='*80}")

    bottom_words = find_best_letter_frequency_words(word_list, top_n=20, find_lowest=True,
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
