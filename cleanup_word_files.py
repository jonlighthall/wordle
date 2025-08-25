#!/usr/bin/env python3
"""
Word file cleanup utility for Wordle project.
Checks for and removes duplicates in word files while preserving original backups.
"""

import os
import sys
from datetime import datetime

# Add the src/python directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

from core.wordle_utils import load_words, save_words_to_file

def clean_word_file(filepath):
    """
    Clean a word file by removing duplicates and normalizing case.
    Returns (original_count, cleaned_count, duplicates_found).
    """
    if not os.path.exists(filepath):
        print(f"⚠️  File not found: {filepath}")
        return 0, 0, 0

    # Load words
    words = load_words(filepath)
    if not words:
        print(f"⚠️  No words found in: {filepath}")
        return 0, 0, 0

    original_count = len(words)

    # Normalize to lowercase and remove duplicates while preserving order
    seen = set()
    cleaned_words = []

    for word in words:
        word_normalized = word.lower().strip()
        if word_normalized and word_normalized not in seen:
            seen.add(word_normalized)
            cleaned_words.append(word_normalized)

    cleaned_count = len(cleaned_words)
    duplicates_found = original_count - cleaned_count

    # Create backup if duplicates were found
    if duplicates_found > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{filepath}.backup_{timestamp}"

        try:
            # Copy original file to backup
            with open(filepath, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            print(f"📋 Created backup: {backup_path}")

            # Save cleaned words back to original file
            save_words_to_file(filepath, cleaned_words)
            print(f"✅ Cleaned {filepath}: {original_count} → {cleaned_count} words ({duplicates_found} duplicates removed)")

        except Exception as e:
            print(f"❌ Error cleaning {filepath}: {e}")
            return original_count, original_count, 0
    else:
        print(f"✅ {filepath}: {original_count} words (no duplicates found)")

    return original_count, cleaned_count, duplicates_found

def main():
    """Main cleanup function."""
    print("🧹 Wordle Word File Cleanup Utility")
    print("=" * 50)

    # Define the files to check
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    files_to_check = [
        ("words_past5_date.txt", "Past Wordle words"),
        ("words_missing.txt", "Failed/missing words"),
        ("words_challenging.txt", "Challenging words")
    ]

    total_original = 0
    total_cleaned = 0
    total_duplicates = 0

    for filename, description in files_to_check:
        filepath = os.path.join(data_dir, filename)
        print(f"\n📁 Checking {description} ({filename})...")

        original, cleaned, duplicates = clean_word_file(filepath)
        total_original += original
        total_cleaned += cleaned
        total_duplicates += duplicates

    # Summary
    print("\n" + "=" * 50)
    print("📊 CLEANUP SUMMARY")
    print("=" * 50)
    print(f"Total words processed: {total_original}")
    print(f"Total words after cleanup: {total_cleaned}")
    print(f"Total duplicates removed: {total_duplicates}")

    if total_duplicates > 0:
        print(f"💾 Backup files created for any modified files")
        print(f"✨ Cleanup completed! Removed {total_duplicates} duplicate words.")
    else:
        print("✨ All files were already clean - no duplicates found!")

if __name__ == "__main__":
    main()
