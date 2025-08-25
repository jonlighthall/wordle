#!/usr/bin/env python3
"""
Quick test of feedback input handling
"""

def test_feedback_input():
    """Test that feedback input accepts lowercase letters."""
    print("Testing feedback input handling...")

    test_cases = ["gyyxx", "GYYXX", "GYyXx", "gyxgx"]

    for test_input in test_cases:
        feedback_input = test_input.strip().upper()
        valid = len(feedback_input) == 5 and all(c in 'GYX' for c in feedback_input)
        print(f"Input: '{test_input}' -> Converted: '{feedback_input}' -> Valid: {valid}")

if __name__ == "__main__":
    test_feedback_input()
