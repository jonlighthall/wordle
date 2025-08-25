#!/usr/bin/env python3
"""
Test the interactive word rejection feature
"""

from src.python.cli.wordle import WordleSolver
from src.python.core.wordle_utils import load_words

def test_interactive_rejection():
    """Test the word rejection feature with the WordleSolver."""
    print("Testing word rejection feature...")

    # Load test words
    test_file = "test_words.txt"
    words = load_words(test_file)
    print(f"Loaded {len(words)} words: {words}")

    # Create solver with file path
    solver = WordleSolver(words, word_file_path=test_file)
    print(f"Solver created with {len(solver.word_list)} words")

    # Test rejecting a word
    print("\nTesting rejection of 'flamb'...")
    solver.remove_rejected_word("flamb")

    print(f"After rejection: {len(solver.word_list)} words remain")
    print(f"Remaining words: {solver.word_list}")

    # Check if file was updated
    updated_words = load_words(test_file)
    print(f"Words in file after update: {updated_words}")

    return len(updated_words) == 4 and "flamb" not in updated_words

if __name__ == "__main__":
    success = test_interactive_rejection()
    print(f"\nTest {'PASSED' if success else 'FAILED'}!")
