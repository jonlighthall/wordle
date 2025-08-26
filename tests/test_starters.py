#!/usr/bin/env python3

# Simple test script to check algorithm starter words
import os
import sys

# Add the source directory to Python path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one more level
sys.path.insert(0, os.path.join(repo_root, 'src', 'python'))

# Import what we need
from core.wordle_utils import load_words
from cli.wordle import WordleSolver

# Load word list
word_file_path = os.path.join(repo_root, 'data', 'words_alpha5.txt')
word_list = load_words(word_file_path)

if not word_list:
    print("Could not load word list")
    sys.exit(1)

print(f"Loaded {len(word_list)} words from file")

# Initialize solver
solver = WordleSolver(word_list)

# Test each algorithm's first guess
algorithms = [
    ('entropy', 'Entropy'),
    ('frequency', 'Frequency'),
    ('information', 'Information'),
    ('ultra_efficient', 'Ultra-Efficient'),
    ('adaptive_hybrid', 'Adaptive-Hybrid')
]

starters = []
print("\nAlgorithm starter words:")
print("-" * 40)

for alg_key, alg_name in algorithms:
    if alg_key == 'entropy':
        starter = solver.choose_guess_entropy(False)
    elif alg_key == 'frequency':
        starter = solver.choose_guess_frequency(start_strategy="fixed")
    elif alg_key == 'information':
        starter = solver.choose_guess_information(False)
    elif alg_key == 'ultra_efficient':
        starter = solver.choose_guess_ultra_efficient()
    elif alg_key == 'adaptive_hybrid':
        starter = solver.choose_guess_adaptive_hybrid()

    print(f"{alg_name:15}: {starter.upper()}")
    starters.append(starter.upper())

print(f"\nUnique starters: {len(set(starters))} out of {len(starters)}")

if len(set(starters)) == len(starters):
    print("✅ All algorithms have unique starter words!")
else:
    print("❌ Some algorithms share starter words:")
    from collections import Counter
    counts = Counter(starters)
    for word, count in counts.items():
        if count > 1:
            print(f"  {word}: used {count} times")
