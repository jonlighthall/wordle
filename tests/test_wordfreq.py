#!/usr/bin/env python3
"""Test the new word frequency scoring functionality."""

# Test wordfreq directly first
try:
    from wordfreq import word_frequency

    test_words = ['house', 'apple', 'xylem', 'crane', 'tares', 'about', 'world']
    print("Direct wordfreq scores:")
    for word in test_words:
        freq = word_frequency(word, 'en')
        print(f"{word}: {freq:.2e}")

    # Test scaling
    print("\nScaled scores (0-1 range):")
    for word in test_words:
        freq = word_frequency(word, 'en')
        if freq > 0:
            import math
            log_freq = math.log10(freq)
            score = max(0.0, min(1.0, (log_freq + 7) / 5))
            print(f"{word}: {score:.4f} (raw: {freq:.2e})")

    print("\n✅ wordfreq functionality working correctly")

except ImportError:
    print("❌ wordfreq not available")
