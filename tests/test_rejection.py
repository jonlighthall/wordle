#!/usr/bin/env python3
"""
Test the word rejection functionality
"""

from src.python.core.wordle_utils import remove_word_from_list, save_words_to_file, load_words

def test_word_removal():
    """Test word removal functionality."""
    print("Testing word removal functionality...")

    # Create test word list
    test_words = ["crane", "house", "smile", "grape", "flamb"]
    print(f"Original words: {test_words}")

    # Remove a word
    updated_words = remove_word_from_list(test_words, "flamb")
    print(f"After removing 'flamb': {updated_words}")

    # Test case insensitive removal
    updated_words = remove_word_from_list(test_words, "CRANE")
    print(f"After removing 'CRANE': {updated_words}")

    print("Word removal test completed!")

if __name__ == "__main__":
    test_word_removal()
