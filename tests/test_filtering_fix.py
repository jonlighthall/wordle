#!/usr/bin/env python3
"""Quick test to verify the redundant filtering message fix."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from core.wordle_utils import load_words
from cli.wordle import WordleSolver

# Create test solvers
word_list = ['crane', 'slate', 'trace', 'stare', 'audio', 'ratio', 'arise', 'raise']
solvers = {}
for alg in ['entropy', 'frequency', 'information']:
    solvers[alg] = WordleSolver(word_list)

print("Testing filtering with multiple solvers...")
print("Should only see ONE 'Filtered from X to Y' message:")

# Simulate the fixed filtering loop
chosen_word = "crane"
feedback = "XGGXX"

for i, solver in enumerate(solvers.values()):
    solver.guesses.append(chosen_word)
    solver.feedbacks.append(feedback)
    # Only print for first solver (i == 0)
    solver.filter_words(chosen_word, feedback, verbose=(i == 0))

print(f"✅ Test completed. Remaining words: {[len(s.possible_words) for s in solvers.values()]}")
