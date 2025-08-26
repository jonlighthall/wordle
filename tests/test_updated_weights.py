#!/usr/bin/env python3

import sys
import os
import random

# Add the src/python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from core.wordle_utils import load_words
from cli.wordle import WordleSolver

def test_updated_weights():
    """Test the updated adaptive hybrid weights based on 300-word analysis."""

    # Load word lists
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    word_list = load_words(os.path.join(data_dir, 'words_alpha5.txt'))
    test_words = load_words(os.path.join(data_dir, 'words_past5_date.txt'))

    if not word_list or not test_words:
        print("Error: Could not load word files")
        return

    # Test on a sample of words
    test_sample = test_words[:30]  # Test on first 30 words

    print(f"Testing updated adaptive hybrid (frequency-favored) on {len(test_sample)} words...")
    print("="*65)

    solver = WordleSolver(word_list)
    total_attempts = 0
    solved_count = 0
    results = []

    for i, target in enumerate(test_sample, 1):
        result = solver.solve_automated(target, 'adaptive_hybrid')
        attempts = result['attempts']
        solved = result['solved']

        total_attempts += attempts
        if solved:
            solved_count += 1
        results.append(attempts)

        status = '✓' if solved else '✗'
        print(f'{i:2d}. {target.upper()}: {attempts} attempts {status}')

    avg_attempts = total_attempts / len(test_sample)
    success_rate = solved_count / len(test_sample) * 100

    print("="*65)
    print(f"Results for updated adaptive hybrid:")
    print(f"Average attempts: {avg_attempts:.2f}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Total attempts: {total_attempts}")

    # Show distribution
    attempt_dist = {}
    for attempts in results:
        attempt_dist[attempts] = attempt_dist.get(attempts, 0) + 1

    print(f"\nAttempt distribution:")
    for attempts in sorted(attempt_dist.keys()):
        count = attempt_dist[attempts]
        percentage = count / len(results) * 100
        print(f"  {attempts} attempts: {count} words ({percentage:.1f}%)")

    return avg_attempts, success_rate

if __name__ == "__main__":
    test_updated_weights()
