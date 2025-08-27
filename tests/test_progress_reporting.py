#!/usr/bin/env python3
"""Test script to verify 10% progress reporting works with different file sizes."""

import os
import sys

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'python')))

from core.wordle_utils import load_words

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')

def test_progress_reporting():
    """Test progress reporting with different word list sizes."""

    # Test with different file sizes
    test_files = [
        ('words_alpha5_100.txt', 'Small list (100 words)'),
        ('words_alpha5.txt', 'Full list (15913 words)')
    ]

    for filename, description in test_files:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            words = load_words(filepath)
            total_words = len(words)
            step_size = max(1, total_words // 10)

            print(f"\n📊 {description}:")
            print(f"   Total words: {total_words}")
            print(f"   Step size: {step_size}")
            print("   Progress points would be:")

            progress_points = []
            for i in range(0, total_words):
                if i % step_size == 0 or i == total_words - 1:
                    progress_percent = (i / total_words) * 100
                    progress_points.append(f"   {i}/{total_words} ({progress_percent:.1f}%)")

            for point in progress_points[:5]:  # Show first 5 progress points
                print(point)
            if len(progress_points) > 5:
                print("   ...")
                print(progress_points[-1])  # Show final point
        else:
            print(f"\n❌ {description}: File not found - {filepath}")

if __name__ == "__main__":
    test_progress_reporting()
