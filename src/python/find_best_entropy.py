#!/usr/bin/env python3
"""
Script to find the word with the highest entropy from the word list.
This calculates the entropy for each word when used as a first guess.
"""

import math
from collections import Counter
from typing import List, Tuple

def get_feedback(guess: str, target: str) -> str:
    """Generate feedback for a guess against the target word.
    Returns a string of 'G' (green), 'Y' (yellow), 'X' (gray)."""
    word_length = len(guess)
    feedback = ['X'] * word_length
    target_chars = list(target)

    # First pass: Mark green (correct letter, correct position)
    for i in range(word_length):
        if guess[i] == target[i]:
            feedback[i] = 'G'
            target_chars[i] = None  # Remove matched letter

    # Second pass: Mark yellow (correct letter, wrong position)
    for i in range(word_length):
        if feedback[i] == 'G':
            continue
        if guess[i] in target_chars:
            feedback[i] = 'Y'
            target_chars[target_chars.index(guess[i])] = None

    return ''.join(feedback)

def calculate_entropy(guess: str, possible_words: List[str]) -> float:
    """Calculate the entropy for a given guess against all possible target words."""
    pattern_counts = Counter()

    for possible_target in possible_words:
        feedback = get_feedback(guess, possible_target)
        pattern_counts[feedback] += 1

    total_words = len(possible_words)
    entropy = 0
    for count in pattern_counts.values():
        probability = count / total_words
        entropy -= probability * math.log2(probability) if probability > 0 else 0

    return entropy

def find_best_entropy_words(word_list: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
    """Find the words with the highest entropy values."""
    print(f"Calculating entropy for {len(word_list)} words...")
    print("This may take a few minutes...")

    word_entropies = []

    for i, guess in enumerate(word_list):
        if i % 1000 == 0:
            print(f"Progress: {i}/{len(word_list)} words processed...")

        entropy = calculate_entropy(guess, word_list)
        word_entropies.append((guess, entropy))

    # Sort by entropy (highest first)
    word_entropies.sort(key=lambda x: x[1], reverse=True)

    return word_entropies[:top_n]

def main():
    # Load word list from file
    try:
        with open("/home/jlighthall/examp/common/words_alpha5_100.txt", "r") as f:
            word_list = [word.strip() for word in f.readlines()]
        print(f"Loaded {len(word_list)} words from file")
    except FileNotFoundError:
        print("Error: Word file not found at /home/jlighthall/examp/common/words_alpha5.txt")
        return

    # Find top entropy words
    top_words = find_best_entropy_words(word_list, top_n=20)

    print(f"\n{'='*60}")
    print("TOP 20 WORDS BY ENTROPY (Best First Guesses)")
    print(f"{'='*60}")
    print(f"{'Rank':<4} {'Word':<8} {'Entropy':<10}")
    print(f"{'-'*25}")

    for i, (word, entropy) in enumerate(top_words, 1):
        print(f"{i:<4} {word:<8} {entropy:<10.4f}")

    print(f"\nBest word for first guess: '{top_words[0][0]}' with entropy {top_words[0][1]:.4f}")

if __name__ == "__main__":
    main()
