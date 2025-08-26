#!/usr/bin/env python3
"""
Main entry point for the Wordle analysis project.
Run this from the repository root directory.
"""

import sys
import os

# Add the src/python directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

def main():
    """Main entry point with options for different analysis modes."""
    print("🎯 Wordle Analysis Suite")
    print("="*50)
    print("1. Interactive Wordle Solver")
    print("2. Automated Testing")
    print("3. Find Best Entropy Words")
    print("4. Find Best Frequency Words")
    print("5. Run Tests")
    print("6. Clean Word Files (remove duplicates)")
    print("="*50)

    try:
        choice = input("Select an option (1-6) [default: 1]: ").strip()
    except KeyboardInterrupt:
        print("\nThanks for playing! 🎯")
        return

    # Make option 1 the default if no input is provided
    if not choice:
        choice = "1"

    if choice == "1":
        from src.python.cli.wordle import interactive_mode
        interactive_mode()
    elif choice == "2":
        from src.python.cli.wordle import automated_testing
        automated_testing()
    elif choice == "3":
        from src.python.analysis.find_best_entropy import main as entropy_analysis
        entropy_analysis()
    elif choice == "4":
        from src.python.analysis.find_best_frequency import main as frequency_analysis
        frequency_analysis()
    elif choice == "5":
        print("Running tests...")
        from tests.test_feedback import test_feedback_input
        from tests.test_rejection import test_word_removal
        from tests.test_attempt_counting import test_attempt_counting

        test_feedback_input()
        test_word_removal()
        test_attempt_counting()
    elif choice == "6":
        # Import and run the cleanup script
        import subprocess
        subprocess.run([sys.executable, "cleanup_word_files.py"])
    else:
        print("Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()
