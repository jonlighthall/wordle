#!/usr/bin/env python3
"""
Common utilities for Wordle analysis - consolidates duplicate code across the project.
This module provides shared functionality for wordfreq integration, progress reporting,
path management, and other common patterns.
"""

import os
import sys
import math
from typing import List, Optional, Callable, Any
from pathlib import Path

# Import wordfreq for real-world word frequency data
try:
    from wordfreq import word_frequency
    WORDFREQ_AVAILABLE = True
except ImportError:
    WORDFREQ_AVAILABLE = False

# Project structure constants
def get_repo_root() -> str:
    """Get the repository root directory from any file in the project."""
    current_file = Path(__file__).resolve()
    # Find the root by looking for specific marker files
    for parent in current_file.parents:
        if (parent / 'run_wordle.py').exists() or (parent / 'data').exists():
            return str(parent)
    # Fallback: assume we're in src/python/core and go up 3 levels
    return str(current_file.parents[2])

REPO_ROOT = get_repo_root()
DATA_DIR = os.path.join(REPO_ROOT, 'data')
LOGS_DIR = os.path.join(REPO_ROOT, 'logs')

def check_wordfreq_availability() -> bool:
    """Check if wordfreq library is available and warn if not."""
    if not WORDFREQ_AVAILABLE:
        print("⚠️  wordfreq library not available. Install with: pip install wordfreq")
        print("   Real-world frequency scoring will use fallback values.")
        return False
    return True

def get_raw_word_frequency(word: str, lang: str = "en", wordlist: str = "large") -> float:
    """Get raw frequency score from wordfreq library.

    Args:
        word: Word to get frequency for
        lang: Language code (default: "en")
        wordlist: Size of wordlist to use ("small", "large", "best").
                 "large" contains ~321k words vs "small" with ~29k words.

    Returns:
        Raw frequency value, or 0.0 if word not found or wordfreq unavailable.
    """
    if not WORDFREQ_AVAILABLE:
        return 0.0

    try:
        return word_frequency(word, lang, wordlist=wordlist)
    except Exception:
        return 0.0

def scale_word_frequency(raw_freq: float) -> float:
    """Scale raw frequency using log10 transform that preserves relative differences.

    The wordfreq library returns frequencies in a wide range:
    - Very common words: ~1e-2 (0.01)
    - Common words: ~1e-3 to 1e-4 (0.001 to 0.0001)
    - Uncommon words: ~1e-5 to 1e-6 (0.00001 to 0.000001)
    - Very rare words: ~1e-7 or less

    We use a linear transform of log10 to preserve meaningful differences:
    - Words not found: score = 0.0
    - log10(1e-8) = -8  → score = 1.0 (baseline rare)
    - log10(1e-7) = -7  → score = 2.0
    - log10(1e-6) = -6  → score = 3.0
    - log10(1e-5) = -5  → score = 4.0
    - log10(1e-2) = -2  → score = 7.0 (most common)

    Each unit difference in score = 10x difference in frequency.
    A difference of 3 = 1000x difference in frequency.
    """
    if raw_freq <= 0:
        return 0.0

    # Use log10 and shift so -8 maps to 1.0
    log_freq = math.log10(raw_freq)
    score = max(0.0, log_freq + 8.0)
    return score

def get_word_frequency_score(word: str, lang: str = "en", wordlist: str = "large") -> float:
    """Get scaled real-world frequency score for a word.

    This is the main function that combines raw frequency retrieval and scaling.
    Returns scaled score where each unit represents 10x frequency difference.
    If wordfreq is unavailable, returns 2.5 as neutral middle-range score.
    """
    if not WORDFREQ_AVAILABLE:
        return 2.5  # Neutral score in the new scale (between rare and common)

    raw_freq = get_raw_word_frequency(word, lang, wordlist)
    if raw_freq > 0:
        return scale_word_frequency(raw_freq)
    else:
        return 0.0  # Word not found

class ProgressReporter:
    """Standardized progress reporting for long-running analysis tasks."""

    def __init__(self, total_items: int, report_interval: int = 10):
        """Initialize progress reporter.

        Args:
            total_items: Total number of items to process
            report_interval: Number of progress reports (default: 10 for 10% intervals)
        """
        self.total_items = total_items
        self.step_size = max(1, total_items // report_interval)
        self.report_interval = report_interval

    def report_progress(self, current_index: int, context: str = "items") -> None:
        """Report progress if at a reporting interval or at the end.

        Args:
            current_index: Current item index (0-based)
            context: Description of what's being processed (e.g., "words", "guesses")
        """
        if current_index % self.step_size == 0 or current_index == self.total_items - 1:
            progress_percent = (current_index / self.total_items) * 100
            print(f"   Progress: {current_index}/{self.total_items} {context} processed ({progress_percent:.1f}%)...")

    def final_report(self, context: str = "items") -> None:
        """Report completion."""
        print(f"   Completed processing {self.total_items} {context} (100.0%).")

def load_word_list_with_fallback(primary_file: str, fallback_files: Optional[List[str]] = None) -> List[str]:
    """Load word list with fallback options and standardized error handling.

    Args:
        primary_file: Primary word file to try (relative to DATA_DIR)
        fallback_files: List of fallback files to try if primary fails

    Returns:
        List of words, or empty list if all files fail
    """
    from .wordle_utils import load_words  # Import here to avoid circular imports

    fallback_files = fallback_files or []
    files_to_try = [primary_file] + fallback_files

    for filename in files_to_try:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            words = load_words(filepath)
            if words:
                print(f"📚 Loaded {len(words)} words from {filename}")
                return words
            else:
                print(f"⚠️  Warning: {filename} exists but is empty")
        else:
            print(f"⚠️  Warning: {filename} not found")

    print(f"❌ Error: No valid word files found. Tried: {', '.join(files_to_try)}")
    return []

def get_standard_word_file() -> str:
    """Get the standard word file name used across the project."""
    return "words_alpha5.txt"

def get_test_word_file() -> str:
    """Get the test word file name used for smaller datasets."""
    return "words_alpha5_100.txt"

def format_frequency_for_display(raw_freq: float) -> str:
    """Format raw frequency for consistent display across the project."""
    if raw_freq < 1e-4:
        return f"{raw_freq:.2e}"
    else:
        return f"{raw_freq:.6f}"

def format_log10_for_display(raw_freq: float) -> str:
    """Format log10 value for consistent display across the project."""
    if raw_freq > 0:
        log10_val = math.log10(raw_freq)
        return f"{log10_val:.2f}"
    else:
        return "N/A"

def setup_analysis_environment() -> tuple:
    """Set up common environment for analysis scripts.

    Returns:
        Tuple of (repo_root, data_dir, logs_dir, wordfreq_available)
    """
    return REPO_ROOT, DATA_DIR, LOGS_DIR, WORDFREQ_AVAILABLE
