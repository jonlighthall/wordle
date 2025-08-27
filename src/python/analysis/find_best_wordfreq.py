#!/usr/bin/env python3
"""
Script to find the words with the highest and lowest real-world frequency scores from the word list.
This uses the wordfreq library to get actual word usage frequencies from real text corpora.
"""

import math
import os
import sys
from typing import List, Tuple
from collections import defaultdict

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.common_utils import (
    get_raw_word_frequency, scale_word_frequency, WORDFREQ_AVAILABLE,
    DATA_DIR, ProgressReporter, load_word_list_with_fallback,
    format_frequency_for_display, format_log10_for_display
)
from core.wordle_utils import load_words
def analyze_word_frequencies(word_list: List[str]) -> Tuple[List[Tuple[str, float, float]],
                                                           List[Tuple[str, float, float]]]:
    """Analyze word frequencies and return sorted lists of most and least common words.

    Returns:
        Tuple of (most_common, least_common) where each is a list of
        (word, raw_frequency, scaled_score) tuples
    """
    print(f"📊 Analyzing real-world frequencies for {len(word_list)} words...")
    print("   Using wordfreq 'large' dataset (~321k words for better coverage)")

    # Quick coverage comparison
    print("   📈 Coverage improvement: Large wordlist finds ~3.7x more words than small wordlist")

    word_scores = []
    frequency_distribution = defaultdict(int)
    words_not_found = []
    total_words = len(word_list)

    # Use common progress reporting
    progress_reporter = ProgressReporter(total_words, report_interval=10)

    for i, word in enumerate(word_list):
        progress_reporter.report_progress(i, "words")

        raw_freq = get_raw_word_frequency(word)
        scaled_score = scale_word_frequency(raw_freq)
        word_scores.append((word, raw_freq, scaled_score))

        # Track words not found
        if raw_freq == 0.0:
            words_not_found.append(word)

        # Track frequency distribution
        if raw_freq == 0.0:  # Use raw_freq to match words_not_found logic
            frequency_distribution['zero'] += 1
        elif scaled_score < 2.0:  # 1.0-2.0: log10(-8) to log10(-7)
            frequency_distribution['very_rare'] += 1
        elif scaled_score < 4.0:  # 2.0-4.0: log10(-7) to log10(-5)
            frequency_distribution['rare'] += 1
        elif scaled_score < 5.0:  # 4.0-5.0: log10(-5) to log10(-4)
            frequency_distribution['uncommon'] += 1
        elif scaled_score < 6.0:  # 5.0-6.0: log10(-4) to log10(-3)
            frequency_distribution['common'] += 1
        else:  # 6.0+: log10(-3) and higher
            frequency_distribution['very_common'] += 1

    progress_reporter.final_report("words")    # Sort by scaled score
    word_scores.sort(key=lambda x: x[2], reverse=True)

    # Split into most and least common
    most_common = word_scores[:50]  # Top 50

    # For least common, only include words with non-zero frequencies
    words_with_freq = [w for w in word_scores if w[1] > 0.0]  # Only non-zero raw frequencies
    least_common = words_with_freq[-50:]  # Bottom 50 with actual frequencies

    print(f"\n📈 Frequency Distribution:")
    print(f"   Very Common (6.0+):    {frequency_distribution['very_common']:>5} words ({frequency_distribution['very_common']/total_words*100:>5.1f}%)")
    print(f"   Common (5.0-6.0):      {frequency_distribution['common']:>5} words ({frequency_distribution['common']/total_words*100:>5.1f}%)")
    print(f"   Uncommon (4.0-5.0):    {frequency_distribution['uncommon']:>5} words ({frequency_distribution['uncommon']/total_words*100:>5.1f}%)")
    print(f"   Rare (2.0-4.0):        {frequency_distribution['rare']:>5} words ({frequency_distribution['rare']/total_words*100:>5.1f}%)")
    print(f"   Very Rare (1.0-2.0):   {frequency_distribution['very_rare']:>5} words ({frequency_distribution['very_rare']/total_words*100:>5.1f}%)")
    print(f"   Not Found (0.0):       {frequency_distribution['zero']:>5} words ({frequency_distribution['zero']/total_words*100:>5.1f}%)")

    # Report words not found
    if words_not_found:
        print(f"\n❌ Words Not Found in Wordfreq Database:")
        print(f"   {len(words_not_found)} words were not found in the wordfreq 'large' dataset")
        # Show a short example of words not found (up to 10)
        example_not_found = words_not_found[:10]
        print(f"   Examples: {', '.join([w.upper() for w in example_not_found[:5]])}", end="")
        if len(example_not_found) > 5:
            print(f", {', '.join([w.upper() for w in example_not_found[5:]])}", end="")
        if len(words_not_found) > 10:
            print(f" ... and {len(words_not_found) - 10} more")
        else:
            print()

    return most_common, least_common


def print_word_analysis(words: List[Tuple[str, float, float]], title: str, max_display: int = 25):
    """Print formatted analysis of words with their frequencies."""
    print(f"\n🎯 {title}")
    print("=" * 85)
    print(f"{'Rank':<4} {'Word':<8} {'Raw Frequency':<15} {'Log10':<8} {'Scaled Score':<12} {'Classification'}")
    print("-" * 85)

    for i, (word, raw_freq, scaled_score) in enumerate(words[:max_display], 1):
        # Classify the word
        if scaled_score >= 6.0:
            classification = "Very Common"
        elif scaled_score >= 5.0:
            classification = "Common"
        elif scaled_score >= 4.0:
            classification = "Uncommon"
        elif scaled_score >= 2.0:
            classification = "Rare"
        elif scaled_score > 0.0:
            classification = "Very Rare"
        else:
            classification = "Not Found"

        # Format raw frequency and log10 using common utilities
        freq_str = format_frequency_for_display(raw_freq)
        log10_str = format_log10_for_display(raw_freq)

        print(f"{i:<4} {word.upper():<8} {freq_str:<15} {log10_str:<8} {scaled_score:<12.3f} {classification}")


def main():
    """Main function to analyze word frequencies."""
    print("🔍 Real-World Word Frequency Analysis")
    print("=" * 70)

    if not WORDFREQ_AVAILABLE:
        print("❌ wordfreq library is required but not available.")
        print("   Install with: pip install wordfreq")
        return

    # Load the word list using common utilities
    words = load_word_list_with_fallback("words_alpha5.txt", ["words_alpha5_100.txt"])
    if not words:
        return

    # Analyze frequencies
    most_common, least_common = analyze_word_frequencies(words)

    # Display results
    print_word_analysis(most_common, "MOST COMMON WORDS (Highest Real-World Usage)", 25)
    print_word_analysis(least_common, "LEAST COMMON WORDS (Lowest Real-World Usage)", 25)

    # Show scaling explanation
    print(f"\n📐 SCALING METHODOLOGY")
    print("=" * 70)
    print("The wordfreq library provides raw frequencies from real text corpora.")
    print("Using 'large' dataset (~321k words) for better coverage of uncommon words.")
    print("Raw frequencies range from ~1e-8 (very rare) to ~1e-2 (very common).")
    print()
    print("Scaling formula: score = max(0, log10(freq) + 8)")
    print("  • Words not found:    score = 0.0")
    print("  • log10(1e-8) = -8  → score = 1.0  (baseline rare)")
    print("  • log10(1e-7) = -7  → score = 2.0  (10x more common)")
    print("  • log10(1e-6) = -6  → score = 3.0  (100x more common)")
    print("  • log10(1e-5) = -5  → score = 4.0  (1,000x more common)")
    print("  • log10(1e-2) = -2  → score = 7.0  (1,000,000x more common)")
    print()
    print("Each unit increase in score = 10x increase in frequency.")
    print("A difference of 3 units = 1,000x difference in frequency.")
    print("This preserves the meaningful logarithmic relationships between words.")

    # Show some example scalings
    print(f"\n📊 EXAMPLE SCALING COMPARISONS")
    print("-" * 65)
    print(f"{'Word':<8} {'Raw Frequency':<12} {'Log10':<8} {'Scaled Score'}")
    print("-" * 65)
    example_words = ['about', 'house', 'water', 'plant', 'crane', 'tares', 'xylem']
    for word in example_words:
        if word in [w[0] for w in most_common + least_common]:
            raw_freq = get_raw_word_frequency(word)
            scaled_score = scale_word_frequency(raw_freq)
            freq_str = format_frequency_for_display(raw_freq)
            log10_str = format_log10_for_display(raw_freq)

            print(f"{word.upper():<8} {freq_str:<12} {log10_str:<8} {scaled_score:.3f}")


if __name__ == "__main__":
    main()
