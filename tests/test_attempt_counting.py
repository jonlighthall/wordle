#!/usr/bin/env python3
"""
Test that rejected words don't count as guesses
"""

def test_attempt_counting():
    """Simulate the attempt counting logic."""
    print("Testing attempt counting with rejections...")

    attempt = 0
    max_attempts = 6

    # Simulate the game flow
    scenarios = [
        ("CRANE", "R"),      # Rejected - shouldn't count
        ("HOUSE", "XYGXX"),  # Accepted - should count as guess 1
        ("FLAMB", "R"),      # Rejected - shouldn't count
        ("SMILE", "GGGGG")   # Accepted - should count as guess 2 and win
    ]

    for word, feedback in scenarios:
        print(f"\n🤖 Guess {attempt + 1}: I suggest '{word}'")
        print(f"   User enters: {word}")
        print(f"   Feedback: {feedback}")

        if feedback == 'R':
            print(f"   '{word}' was rejected by Wordle - removing from word list")
            print("   Generating new suggestion... (this doesn't count as a guess)")
            continue  # Don't increment attempt

        # Only increment for accepted words
        attempt += 1

        if feedback == "GGGGG":
            print(f"\n🎉 Congratulations! You solved it in {attempt} guesses!")
            break

        print(f"   Processed as guess #{attempt}")

    print(f"\nFinal result: Solved in {attempt} guesses (rejections didn't count)")
    return attempt == 2  # Should be 2 guesses despite 4 attempts

if __name__ == "__main__":
    success = test_attempt_counting()
    print(f"Test {'PASSED' if success else 'FAILED'}!")
