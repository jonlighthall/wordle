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

try:
    from wordfreq import word_frequency
    WORDFREQ_AVAILABLE = True
except ImportError:
    WORDFREQ_AVAILABLE = False
    print("❌ wordfreq library not available. Install with: pip install wordfreq")
    sys.exit(1)

from core.wordle_utils import load_words

# Get the repository root directory (4 levels up from this file)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')


def get_raw_frequency(word: str, lang: str = "en") -> float:
    """Get raw frequency score from wordfreq library."""
    try:
        return word_frequency(word, lang)
    except Exception:
        return 0.0


def scale_frequency_score(raw_freq: float) -> float:
    """Scale raw frequency to 0-1 range using logarithmic scaling.

    The wordfreq library returns frequencies in a wide range:
    - Very common words: ~1e-2 (0.01)
    - Common words: ~1e-3 to 1e-4 (0.001 to 0.0001)
    - Uncommon words: ~1e-5 to 1e-6 (0.00001 to 0.000001)
    - Very rare words: ~1e-7 or less

    We use log10 scaling to map this range to [0, 1]:
    - log10(1e-2) = -2  → maps to 1.0 (most common)
    - log10(1e-7) = -7  → maps to 0.0 (least common)
    """
    if raw_freq <= 0:
        return 0.0

    # Use log10 to handle the wide range
    log_freq = math.log10(raw_freq)

    # Scale from expected range [-7, -2] to [0, 1]
    # Formula: (log_freq - min_log) / (max_log - min_log)
    # Where min_log = -7, max_log = -2
    score = max(0.0, min(1.0, (log_freq + 7) / 5))
    return score


def analyze_word_frequencies(word_list: List[str]) -> Tuple[List[Tuple[str, float, float]],
                                                           List[Tuple[str, float, float]]]:
    """Analyze word frequencies and return sorted lists of most and least common words.

    Returns:
        Tuple of (most_common, least_common) where each is a list of
        (word, raw_frequency, scaled_score) tuples
    """
    print(f"📊 Analyzing real-world frequencies for {len(word_list)} words...")

    word_scores = []
    frequency_distribution = defaultdict(int)
    total_words = len(word_list)
    step_size = max(1, total_words // 10)  # Calculate step size for 10 progress reports

    for i, word in enumerate(word_list):
        if i % step_size == 0 or i == total_words - 1:
            progress_percent = (i / total_words) * 100
            print(f"   Progress: {i}/{total_words} words processed ({progress_percent:.1f}%)...")

        raw_freq = get_raw_frequency(word)
        scaled_score = scale_frequency_score(raw_freq)
        word_scores.append((word, raw_freq, scaled_score))

        # Track frequency distribution
        if scaled_score == 0.0:
            frequency_distribution['zero'] += 1
        elif scaled_score < 0.1:
            frequency_distribution['very_rare'] += 1
        elif scaled_score < 0.3:
            frequency_distribution['rare'] += 1
        elif scaled_score < 0.5:
            frequency_distribution['uncommon'] += 1
        elif scaled_score < 0.7:
            frequency_distribution['common'] += 1
        else:
            frequency_distribution['very_common'] += 1

    print(f"   Completed processing {len(word_list)} words (100.0%).")    # Sort by scaled score
    word_scores.sort(key=lambda x: x[2], reverse=True)

    # Split into most and least common
    most_common = word_scores[:50]  # Top 50
    least_common = [w for w in word_scores if w[2] == 0.0]  # All zero-score words
    if not least_common:
        least_common = word_scores[-50:]  # Bottom 50 if no zero scores

    print(f"\n📈 Frequency Distribution:")
    print(f"   Very Common (0.7-1.0): {frequency_distribution['very_common']} words")
    print(f"   Common (0.5-0.7):      {frequency_distribution['common']} words")
    print(f"   Uncommon (0.3-0.5):    {frequency_distribution['uncommon']} words")
    print(f"   Rare (0.1-0.3):        {frequency_distribution['rare']} words")
    print(f"   Very Rare (0.0-0.1):   {frequency_distribution['very_rare']} words")
    print(f"   Not Found (0.0):       {frequency_distribution['zero']} words")

    return most_common, least_common


def print_word_analysis(words: List[Tuple[str, float, float]], title: str, max_display: int = 25):
    """Print formatted analysis of words with their frequencies."""
    print(f"\n🎯 {title}")
    print("=" * 85)
    print(f"{'Rank':<4} {'Word':<8} {'Raw Frequency':<15} {'Log10':<8} {'Scaled Score':<12} {'Classification'}")
    print("-" * 85)

    for i, (word, raw_freq, scaled_score) in enumerate(words[:max_display], 1):
        # Classify the word
        if scaled_score >= 0.7:
            classification = "Very Common"
        elif scaled_score >= 0.5:
            classification = "Common"
        elif scaled_score >= 0.3:
            classification = "Uncommon"
        elif scaled_score >= 0.1:
            classification = "Rare"
        else:
            classification = "Very Rare"

        # Format raw frequency in scientific notation if very small
        if raw_freq < 1e-4:
            freq_str = f"{raw_freq:.2e}"
        else:
            freq_str = f"{raw_freq:.6f}"

        # Calculate log10 value
        if raw_freq > 0:
            log10_val = math.log10(raw_freq)
            log10_str = f"{log10_val:.2f}"
        else:
            log10_str = "N/A"

        print(f"{i:<4} {word.upper():<8} {freq_str:<15} {log10_str:<8} {scaled_score:<12.3f} {classification}")


def main():
    """Main function to analyze word frequencies."""
    print("🔍 Real-World Word Frequency Analysis")
    print("=" * 70)

    if not WORDFREQ_AVAILABLE:
        print("❌ wordfreq library is required but not available.")
        print("   Install with: pip install wordfreq")
        return

    # Load the word list
    words_file = os.path.join(DATA_DIR, 'words_alpha5.txt')
    if not os.path.exists(words_file):
        print(f"❌ Word file not found: {words_file}")
        return

    print(f"📚 Loading words from {words_file}")
    words = load_words(words_file)
    print(f"   Loaded {len(words)} five-letter words")

    # Analyze frequencies
    most_common, least_common = analyze_word_frequencies(words)

    # Display results
    print_word_analysis(most_common, "MOST COMMON WORDS (Highest Real-World Usage)", 25)
    print_word_analysis(least_common, "LEAST COMMON WORDS (Lowest Real-World Usage)", 25)

    # Show scaling explanation
    print(f"\n📐 SCALING METHODOLOGY")
    print("=" * 70)
    print("The wordfreq library provides raw frequencies from real text corpora.")
    print("Raw frequencies range from ~1e-7 (very rare) to ~1e-2 (very common).")
    print()
    print("Scaling formula: score = max(0, min(1, (log10(freq) + 7) / 5))")
    print("  • log10(1e-2) = -2  →  (-2 + 7) / 5 = 1.0  (most common)")
    print("  • log10(1e-7) = -7  →  (-7 + 7) / 5 = 0.0  (least common)")
    print()
    print("This logarithmic scaling captures the intuitive difference between")
    print("common words (like 'about', 'house') and rare words (like 'xylem', 'nixie').")

    # Show some example scalings
    print(f"\n📊 EXAMPLE SCALING COMPARISONS")
    print("-" * 65)
    print(f"{'Word':<8} {'Raw Frequency':<12} {'Log10':<8} {'Scaled Score'}")
    print("-" * 65)
    example_words = ['about', 'house', 'water', 'plant', 'crane', 'tares', 'xylem']
    for word in example_words:
        if word in [w[0] for w in most_common + least_common]:
            raw_freq = get_raw_frequency(word)
            scaled_score = scale_frequency_score(raw_freq)
            freq_str = f"{raw_freq:.2e}" if raw_freq < 1e-4 else f"{raw_freq:.6f}"

            # Calculate log10 value
            if raw_freq > 0:
                log10_val = math.log10(raw_freq)
                log10_str = f"{log10_val:.2f}"
            else:
                log10_str = "N/A"

            print(f"{word.upper():<8} {freq_str:<12} {log10_str:<8} {scaled_score:.3f}")


if __name__ == "__main__":
    main()
