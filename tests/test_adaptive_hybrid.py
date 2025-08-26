#!/usr/bin/env python3
"""
Test the new adaptive hybrid algorithm that dynamically weights entropy and frequency.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from src.python.cli.wordle import WordleSolver, load_words, format_result, analyze_algorithm_performance_by_step

def test_adaptive_hybrid():
    """Test the adaptive hybrid algorithm with detailed analysis."""

    # Load word list
    word_list = load_words('../data/words_alpha5.txt')
    print(f'Loaded {len(word_list)} words\n')

    # Test words including some challenging ones
    test_words = ['crane', 'house', 'smile', 'grape', 'stone', 'slate', 'adieu']
    algorithms = ['entropy', 'frequency', 'adaptive_hybrid']

    results = {}

    print("="*80)
    print("ADAPTIVE HYBRID ALGORITHM TESTING")
    print("="*80)

    for target in test_words:
        print(f'\nTesting {target.upper()}:')
        print("-" * 50)

        for algorithm in algorithms:
            solver = WordleSolver(word_list)
            result = solver.solve_automated(target, guess_algorithm=algorithm, start_strategy='fixed')

            key = f"{algorithm} (fixed)"
            if key not in results:
                results[key] = []
            results[key].append(result)

            print(f'  {algorithm:15}: {format_result(result["solved"], result["attempts"])}')

    # Summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    for algorithm_key, algorithm_results in results.items():
        total_attempts = sum(r['attempts'] for r in algorithm_results)
        solved_count = sum(1 for r in algorithm_results if r['solved'])
        avg_attempts = total_attempts / len(algorithm_results) if algorithm_results else 0
        success_rate = (solved_count / len(algorithm_results) * 100) if algorithm_results else 0

        print(f"{algorithm_key:20}: {solved_count}/{len(algorithm_results)} solved ({success_rate:.1f}%), avg {avg_attempts:.2f} attempts")

    # Step-by-step analysis
    step_analysis = analyze_algorithm_performance_by_step(results)
    if step_analysis:
        print("\n" + "="*80)
        print("STEP-BY-STEP ANALYSIS (Entropy vs Frequency)")
        print("="*80)

        print(f"{'Step':<4} | {'Entropy Avg':<11} | {'Frequency Avg':<13} | {'Better Algorithm':<15} | {'Recommended Weights'}")
        print("-" * 75)

        for step in sorted(step_analysis.keys()):
            data = step_analysis[step]
            entropy_avg = data['entropy_avg_reduction']
            freq_avg = data['frequency_avg_reduction']
            better = data['better_algorithm']
            entropy_weight = data['entropy_weight']
            freq_weight = data['frequency_weight']

            print(f"{step:<4} | {entropy_avg:8.1f}%   | {freq_avg:10.1f}%   | {better:<15} | E:{entropy_weight:.1f} F:{freq_weight:.1f}")

        print("\nInsights:")
        print("- Adaptive hybrid uses these weights to balance entropy and frequency")
        print("- Early steps typically favor entropy for maximum elimination")
        print("- Later steps may favor frequency for precision targeting")

if __name__ == "__main__":
    test_adaptive_hybrid()
