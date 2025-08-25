#!/usr/bin/env python3
"""
Shared utilities for Wordle-related functions.
Contains common functions used across multiple Wordle scripts.
"""

import math
from collections import Counter
from typing import List


def get_feedback(guess: str, target: str) -> str:
    """Generate feedback for a guess against the target word.
    Returns a string of 'G' (green), 'Y' (yellow), 'X' (gray)."""
    # Validate input lengths
    if len(guess) != len(target):
        raise ValueError(f"Length mismatch: guess='{guess}' ({len(guess)} chars), target='{target}' ({len(target)} chars)")

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
    """Check if a word has all unique letters (no repeated letters)."""
    return len(word) == len(set(word))


def is_valid_word(word: str, guess: str, feedback: str) -> bool:
    """Check if a word is consistent with the given guess and feedback.
    Uses Counter-based logic to handle duplicate letters correctly."""
    if len(word) != len(guess) or len(feedback) != len(guess):
        return False

    guess_counter = Counter(guess)
    word_counter = Counter(word)

    # Count how many of each letter should be in the target word
    required_letters = Counter()
    forbidden_letters = set()
    position_requirements = {}  # position -> required letter
    position_forbidden = {}     # position -> set of forbidden letters

    for i, (g_char, f_char) in enumerate(zip(guess, feedback)):
        if f_char == 'G':  # Green: correct letter, correct position
            required_letters[g_char] += 1
            position_requirements[i] = g_char
        elif f_char == 'Y':  # Yellow: correct letter, wrong position
            required_letters[g_char] += 1
            if i not in position_forbidden:
                position_forbidden[i] = set()
            position_forbidden[i].add(g_char)
        else:  # Gray: letter not in word at all, OR no more instances needed
            # This is tricky - gray means either:
            # 1. Letter is not in the word at all
            # 2. Letter is in the word, but we already found all instances via G/Y
            pass

    # Check position requirements (Green letters)
    for pos, required_char in position_requirements.items():
        if word[pos] != required_char:
            return False

    # Check position forbidden (Yellow letters can't be in their guessed position)
    for pos, forbidden_chars in position_forbidden.items():
        if word[pos] in forbidden_chars:
            return False

    # Check that word contains at least the required letters
    for letter, min_count in required_letters.items():
        if word_counter[letter] < min_count:
            return False

    # Handle gray letters - they indicate no additional instances beyond what we found
    for i, (g_char, f_char) in enumerate(zip(guess, feedback)):
        if f_char == 'X':  # Gray
            # Count how many we should have found via green/yellow
            expected_count = required_letters.get(g_char, 0)
            actual_count = word_counter.get(g_char, 0)
            if actual_count != expected_count:
                return False

    return True


def load_words(filename: str = "words_alpha5.txt") -> List[str]:
    """Load words from a file."""
    try:
        with open(filename, 'r') as f:
            words = []
            for line in f.readlines():
                word = line.strip().lower()
                # Only include 5-letter words to prevent length mismatches
                if len(word) == 5 and word.isalpha():
                    words.append(word)
            return words
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        return []


def is_wordle_appropriate(word: str) -> bool:
    """Check if a word is appropriate for Wordle (base form, common words)."""
    word = word.lower()

    # Simple filtering - just exclude plurals and past tense verbs

    # Filter out plurals (words ending in 's')
    if word.endswith('s'):
        return False

    # Filter out past tense verbs (words ending in 'ed')
    if word.endswith('ed'):
        return False

    return True


def filter_words_unique_letters(words: List[str]) -> List[str]:
    """Filter words to only include those with unique letters (isograms)."""
    return [word for word in words if has_unique_letters(word)]


def filter_wordle_appropriate(words: List[str]) -> List[str]:
    """Filter words to only include Wordle-appropriate words."""
    return [word for word in words if is_wordle_appropriate(word)]


def should_prefer_isograms(possible_words: List[str], total_guesses: int = 0) -> bool:
    """Determine if we should prefer isograms based on remaining possibilities and guess count."""
    if total_guesses >= 3:  # After 3 guesses, be more flexible
        return False

    if len(possible_words) == 0:
        return True  # Default to isograms when no constraints

    # Count how many remaining words are isograms
    isogram_count = sum(1 for word in possible_words if has_unique_letters(word))
    total_count = len(possible_words)
    isogram_ratio = isogram_count / total_count if total_count > 0 else 0

    # Prefer isograms if they make up more than 30% of remaining possibilities
    # or if we're early in the game (first 2 guesses)
    return isogram_ratio > 0.3 or total_guesses < 2


def remove_word_from_list(word_list: List[str], word_to_remove: str) -> List[str]:
    """Remove a word from the word list (case-insensitive)."""
    word_to_remove = word_to_remove.lower()
    return [word for word in word_list if word.lower() != word_to_remove]


def save_words_to_file(filename: str, words: List[str]) -> bool:
    """Save words to a file, one per line."""
    try:
        with open(filename, 'w') as f:
            for word in words:
                f.write(word + '\n')
        return True
    except Exception as e:
        print(f"Error saving words to {filename}: {e}")
        return False


def calculate_position_frequencies(words: List[str]) -> List[Counter]:
    """Calculate letter frequencies for each position."""
    word_length = 5  # Assuming 5-letter words
    position_freqs = [Counter() for _ in range(word_length)]

    for word in words:
        for i, letter in enumerate(word):
            position_freqs[i][letter] += 1

    return position_freqs


def get_word_information_score(word: str, possible_words: List[str]) -> float:
    """Calculate an information score for a word based on position frequencies."""
    if not possible_words:
        return 0.0

    position_freqs = calculate_position_frequencies(possible_words)
    total_words = len(possible_words)

    score = 0.0
    used_letters = set()

    for i, letter in enumerate(word):
        # Frequency of this letter in this position
        pos_freq = position_freqs[i].get(letter, 0)
        # Bonus for letters we haven't used yet (encourages diversity)
        diversity_bonus = 1.0 if letter not in used_letters else 0.5
        # Information content (inverse frequency gives more points for rare letters)
        info_score = (total_words - pos_freq) / total_words * diversity_bonus
        score += info_score
        used_letters.add(letter)

    return score


def get_optimal_endgame_guess(possible_words: List[str], word_list: List[str]) -> str:
    """Get optimal guess when few possibilities remain."""
    if len(possible_words) <= 1:
        return possible_words[0] if possible_words else None

    if len(possible_words) == 2:
        # With 2 possibilities, just guess one
        return possible_words[0]

    if len(possible_words) <= 4:
        # With 3-4 possibilities, find best discriminator
        best_word = None
        best_score = -1

        # Consider both possible words and some exploration words
        candidates = list(set(possible_words + [w for w in word_list if has_unique_letters(w)][:50]))

        for word in candidates:
            entropy = calculate_entropy(word, possible_words)
            # Slight preference for actual possible answers
            bonus = 0.1 if word in possible_words else 0.0
            score = entropy + bonus

            if score > best_score:
                best_score = score
                best_word = word

        return best_word

    return None  # Use normal strategy
