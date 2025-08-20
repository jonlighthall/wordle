#!/usr/bin/env python3
"""
Script to find the words with the highest and lowest frequency scores from the word list.
This calculates the letter frequency for each position and scores words accordingly.
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

def has_unique_letters(word: str) -> bool:
    """Check if a word has all unique letters (no repeating letters) - i.e., is an isogram."""
    return len(set(word)) == len(word)

def filter_words_unique_letters(word_list: List[str]) -> List[str]:
    """Filter word list to only include words with unique letters (isograms)."""
    return [word for word in word_list if has_unique_letters(word)]

def calculate_frequency_score(word: str, word_list: List[str]) -> Tuple[int, float]:
    """Calculate frequency score for a word based on letter frequencies in each position."""
    word_length = len(word)
    total_words = len(word_list)

    # Calculate letter frequencies for each position
    freq = [Counter() for _ in range(word_length)]

    for w in word_list:
        for i, char in enumerate(w):
            freq[i][char] += 1

    # Calculate raw frequency score and normalized likelihood score
    freq_score = sum(freq[i][word[i]] for i in range(word_length))
    likelihood_score = freq_score / total_words

    return freq_score, likelihood_score

def find_best_frequency_words(word_list: List[str], top_n: int = 10, find_lowest: bool = False, calculate_entropy_upfront: bool = False, unique_letters_only: bool = True) -> List[Tuple[str, int, float, float]]:
    """Find the words with the highest (or lowest) frequency scores."""
    analysis_type = "lowest" if find_lowest else "highest"
    
    # Filter to isograms (words with unique letters only) if requested
    if unique_letters_only:
        original_count = len(word_list)
        word_list = filter_words_unique_letters(word_list)
        filtered_count = len(word_list)
        print(f"Filtered to {filtered_count} words with unique letters (isograms) from {original_count} total words")
    
    print(f"Calculating {analysis_type} frequency scores for {len(word_list)} words...")
    
    # OPTIMIZED: Calculate frequencies once for the entire word list
    word_length = 5
    freq = [Counter() for _ in range(word_length)]
    
    for word in word_list:
        for i, char in enumerate(word):
            freq[i][char] += 1
    
    print(f"Frequency analysis complete. Now scoring {len(word_list)} words...")
    
    word_scores = []

    for i, word in enumerate(word_list):
        if i % 1000 == 0 and len(word_list) > 1000:
            print(f"Progress: {i}/{len(word_list)} words processed...")

        # OPTIMIZED: Simple arithmetic using pre-calculated frequencies
        freq_score = sum(freq[i][word[i]] for i in range(word_length))
        likelihood_score = freq_score / len(word_list)

        # Only calculate entropy if explicitly requested (much slower)
        if calculate_entropy_upfront:
            entropy = calculate_entropy(word, word_list)
        else:
            entropy = 0.0  # Placeholder - will calculate only for top words if needed

        word_scores.append((word, freq_score, likelihood_score, entropy))

    # Sort by frequency score (lowest first if find_lowest, highest first otherwise)
    word_scores.sort(key=lambda x: x[1], reverse=not find_lowest)

    # If entropy wasn't calculated upfront, calculate it only for the top words
    if not calculate_entropy_upfront:
        print(f"Calculating entropy for top {top_n} words...")
        top_words = word_scores[:top_n]
        for i, (word, freq_score, likelihood_score, _) in enumerate(top_words):
            entropy = calculate_entropy(word, word_list)
            word_scores[i] = (word, freq_score, likelihood_score, entropy)

    return word_scores[:top_n]

def main():
    # Load word list from file
    try:
        with open("/home/jlighthall/examp/common/words_alpha5.txt", "r") as f:
            word_list = [word.strip() for word in f.readlines()]
        print(f"Loaded {len(word_list)} words from file")
    except FileNotFoundError:
        print("Error: Word file not found at /home/jlighthall/examp/common/words_alpha5.txt")
        return

    # Find highest frequency words (unique letters only)
    print(f"\n{'='*80}")
    print("HIGHEST FREQUENCY ANALYSIS (UNIQUE LETTERS ONLY)")
    print(f"{'='*80}")
    
    top_words = find_best_frequency_words(word_list, top_n=20, find_lowest=False, unique_letters_only=True)
    
    print(f"{'Rank':<4} {'Word':<8} {'Freq Score':<10} {'Likelihood':<12} {'Entropy':<10}")
    print(f"{'-'*55}")
    
    for i, (word, freq_score, likelihood_score, entropy) in enumerate(top_words, 1):
        print(f"{i:<4} {word:<8} {freq_score:<10} {likelihood_score:<12.4f} {entropy:<10.4f}")
    
    print(f"\nBest word for highest frequency: '{top_words[0][0]}' with score {top_words[0][1]} (likelihood {top_words[0][2]:.4f})")
    
    # Find lowest frequency words (unique letters only)
    print(f"\n{'='*80}")
    print("LOWEST FREQUENCY ANALYSIS (UNIQUE LETTERS ONLY)")
    print(f"{'='*80}")
    
    bottom_words = find_best_frequency_words(word_list, top_n=20, find_lowest=True, unique_letters_only=True)
    
    print(f"{'Rank':<4} {'Word':<8} {'Freq Score':<10} {'Likelihood':<12} {'Entropy':<10}")
    print(f"{'-'*55}")

    for i, (word, freq_score, likelihood_score, entropy) in enumerate(bottom_words, 1):
        print(f"{i:<4} {word:<8} {freq_score:<10} {likelihood_score:<12.4f} {entropy:<10.4f}")

    print(f"\nBest word for lowest frequency: '{bottom_words[0][0]}' with score {bottom_words[0][1]} (likelihood {bottom_words[0][2]:.4f})")

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"Highest frequency word: {top_words[0][0]} (score: {top_words[0][1]}, entropy: {top_words[0][3]:.4f})")
    print(f"Lowest frequency word:  {bottom_words[0][0]} (score: {bottom_words[0][1]}, entropy: {bottom_words[0][3]:.4f})")

    # Find the word that appears in both entropy and frequency top lists
    # (Would need to run entropy analysis here for comparison, but this gives the basic structure)

if __name__ == "__main__":
    main()
