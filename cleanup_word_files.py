#!/usr/bin/env python3
"""
Word file cleanup utility for Wordle project.
Checks for and removes duplicates in word files while preserving original backups.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Set

# Add the src/python directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

from core.wordle_utils import load_words, save_words_to_file

def clean_word_file(filepath: str, dry_run: bool = False) -> Tuple[int, int, int]:
    """
    Clean a word file by removing duplicates and normalizing case.

    Args:
        filepath: Path to the word file
        dry_run: If True, only report what would be changed without making changes

    Returns:
        Tuple of (original_count, cleaned_count, duplicates_found)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"⚠️  File not found: {filepath}")
        return 0, 0, 0

    # Load words more efficiently
    try:
        words = load_words(str(filepath))
        if not words:
            print(f"⚠️  No words found in: {filepath}")
            return 0, 0, 0
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return 0, 0, 0

    original_count = len(words)

    # Use dict.fromkeys() to remove duplicates while preserving order (Python 3.7+)
    # This is more efficient than using a set and list
    normalized_words = [word.lower().strip() for word in words if word.strip()]
    cleaned_words = list(dict.fromkeys(word for word in normalized_words if word))

    cleaned_count = len(cleaned_words)
    duplicates_found = original_count - cleaned_count

    if dry_run:
        if duplicates_found > 0:
            print(f"🔍 Would clean {filepath}: {original_count} → {cleaned_count} words ({duplicates_found} duplicates)")
        else:
            print(f"✅ {filepath}: {original_count} words (no duplicates found)")
        return original_count, cleaned_count, duplicates_found

    # Create backup if duplicates were found
    if duplicates_found > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.with_suffix(f"{filepath.suffix}.backup_{timestamp}")

        try:
            # Use shutil.copy2 for better file copying (preserves metadata)
            shutil.copy2(filepath, backup_path)
            print(f"📋 Created backup: {backup_path}")

            # Save cleaned words back to original file
            if not save_words_to_file(str(filepath), cleaned_words):
                print(f"❌ Failed to save cleaned words to {filepath}")
                return original_count, original_count, 0

            print(f"✅ Cleaned {filepath}: {original_count} → {cleaned_count} words ({duplicates_found} duplicates removed)")

        except Exception as e:
            print(f"❌ Error cleaning {filepath}: {e}")
            return original_count, original_count, 0
    else:
        print(f"✅ {filepath}: {original_count} words (no duplicates found)")

    return original_count, cleaned_count, duplicates_found

def main():
    """Main cleanup function with enhanced options."""
    print("🧹 Wordle Word File Cleanup Utility")
    print("=" * 50)

    # Add command line argument support
    import argparse
    parser = argparse.ArgumentParser(description='Clean word files by removing duplicates')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without making changes')
    parser.add_argument('--files', nargs='+',
                       help='Specific files to clean (default: all standard files)')

    # Parse args if called from command line, otherwise use defaults
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        # Interactive mode - ask user for options
        print("\nOptions:")
        print("1. Clean all files")
        print("2. Dry run (preview changes)")
        print("3. Select specific files")

        choice = input("\nSelect option (1-3, or Enter for option 1): ").strip()

        class Args:
            def __init__(self):
                self.dry_run = choice == "2"
                self.files = None

        args = Args()

        if choice == "3":
            print("\nAvailable files:")
            available_files = ["words_past5_date.txt", "words_missing.txt", "words_challenging.txt"]
            for i, filename in enumerate(available_files, 1):
                print(f"{i}. {filename}")

            selections = input("\nEnter file numbers (e.g., 1,3): ").strip()
            if selections:
                try:
                    indices = [int(x.strip()) - 1 for x in selections.split(',')]
                    args.files = [available_files[i] for i in indices if 0 <= i < len(available_files)]
                except (ValueError, IndexError):
                    print("Invalid selection, using all files.")

    # Define the files to check
    data_dir = Path(__file__).parent / 'data'

    if args.files:
        files_to_check = [(f, f.replace('_', ' ').replace('.txt', '').title()) for f in args.files]
    else:
        files_to_check = [
            ("words_past5_date.txt", "Past Wordle words"),
            ("words_missing.txt", "Failed/missing words"),
            ("words_challenging.txt", "Challenging words")
        ]

    if args.dry_run:
        print(f"\n🔍 DRY RUN MODE - No changes will be made")
        print("=" * 50)

    total_original = 0
    total_cleaned = 0
    total_duplicates = 0

    # Process files with better error handling
    for filename, description in files_to_check:
        filepath = data_dir / filename
        print(f"\n📁 Checking {description} ({filename})...")

        original, cleaned, duplicates = clean_word_file(filepath, dry_run=args.dry_run)
        total_original += original
        total_cleaned += cleaned
        total_duplicates += duplicates

    # Enhanced summary with performance metrics
    print("\n" + "=" * 50)
    print("📊 CLEANUP SUMMARY")
    print("=" * 50)
    print(f"Files processed: {len(files_to_check)}")
    print(f"Total words processed: {total_original:,}")
    print(f"Total words after cleanup: {total_cleaned:,}")
    print(f"Total duplicates {'found' if args.dry_run else 'removed'}: {total_duplicates:,}")

    if total_duplicates > 0:
        percentage_reduction = (total_duplicates / total_original) * 100
        print(f"Space savings: {percentage_reduction:.1f}%")

        if not args.dry_run:
            print(f"💾 Backup files created for modified files")
            print(f"✨ Cleanup completed! Removed {total_duplicates:,} duplicate words.")
        else:
            print(f"🔍 Run without --dry-run to apply these changes")
    else:
        print("✨ All files were already clean - no duplicates found!")

if __name__ == "__main__":
    main()
