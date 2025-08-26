#!/usr/bin/env python3
"""
Quick test of the streamlined automated testing with 3 algorithms only.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from src.python.cli.wordle import WordleSolver, load_words, format_result

def test_streamlined_algorithms():
    """Test all 3 streamlined algorithms with a few words."""

    # Load word list
    word_list = load_words('../data/words_alpha5.txt')
    print(f'Loaded {len(word_list)} words\n')

    # Test words
    test_words = ['crane', 'house', 'smile', 'grape', 'stone']
    algorithms = ['entropy', 'frequency', 'ultra_efficient', 'adaptive_hybrid']

    results = {}

    print("="*60)
    print("STREAMLINED AUTOMATED TESTING")
    print("="*60)

    for target in test_words:
        print(f'\nTesting {target}:')
        print("-" * 40)

        for algorithm in algorithms:
            solver = WordleSolver(word_list)
            result = solver.solve_automated(target, guess_algorithm=algorithm, start_strategy='fixed')

            key = f"{algorithm} (fixed)"
            if key not in results:
                results[key] = []
            results[key].append(result)

            print(f'  {algorithm:15}: {format_result(result["solved"], result["attempts"])}')

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for algorithm_key, algorithm_results in results.items():
        total_attempts = sum(r['attempts'] for r in algorithm_results)
        solved_count = sum(1 for r in algorithm_results if r['solved'])
        avg_attempts = total_attempts / len(algorithm_results) if algorithm_results else 0
        success_rate = (solved_count / len(algorithm_results) * 100) if algorithm_results else 0

        print(f"{algorithm_key:20}: {solved_count}/{len(algorithm_results)} solved ({success_rate:.1f}%), avg {avg_attempts:.2f} attempts")

if __name__ == "__main__":
    test_streamlined_algorithms()