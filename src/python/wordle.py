import random
import math
from collections import Counter
from typing import List, Tuple
from wordle_utils import get_feedback, calculate_entropy, has_unique_letters, is_valid_word, load_words, filter_words_unique_letters, filter_wordle_appropriate, should_prefer_isograms, remove_word_from_list, save_words_to_file

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'  # Reset to default color

def format_result(solved: bool, attempts: int, wordle_limit: int = 6) -> str:
    """Format the solve result with appropriate colors."""
    if solved:
        if attempts <= wordle_limit:
            return f"{Colors.GREEN}✓ Solved in {attempts} guesses!{Colors.RESET}"
        else:
            return f"{Colors.RED}✓ Solved in {attempts} guesses (exceeds Wordle limit of {wordle_limit})!{Colors.RESET}"
    else:
        return f"{Colors.RED}✗ Failed to solve after {attempts} guesses.{Colors.RESET}"

def write_failed_word(word: str, method: str, strategy: str = "fixed"):
    """Write a failed word to a log file with timestamp and method info."""
    import datetime

    # Create filename with current date
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    filename = f"failed_words_{date_str}.txt"

    # Format the entry
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} | {word.upper()} | {method}"
    if strategy != "fixed":
        entry += f" ({strategy})"
    entry += "\n"

    try:
        with open(filename, "a") as f:
            f.write(entry)
        print(f"    Failed word logged to {filename}")
    except Exception as e:
        print(f"    Warning: Could not log failed word to {filename}: {e}")

def write_solver_state_after_6(target: str, method: str, strategy: str, possible_words: List[str]):
    """Write solver state when it exceeds 6 guesses (Wordle limit)."""
    import datetime

    # Create filename with current date
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    filename = f"solver_state_after_6_{date_str}.txt"

    # Format the entry
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    num_remaining = len(possible_words)

    # Create header line
    method_info = f"{method}"
    if strategy != "fixed":
        method_info += f" ({strategy})"

    entry = f"{timestamp} | {target.upper()} | {method_info} | {num_remaining} words remaining\n"

    # Add the remaining words (limit to first 50 for readability)
    if num_remaining > 0:
        if num_remaining <= 50:
            entry += f"Remaining words: {', '.join([w.upper() for w in sorted(possible_words)])}\n"
        else:
            entry += f"First 50 remaining words: {', '.join([w.upper() for w in sorted(possible_words)[:50]])}\n"
    else:
        entry += "No remaining words (solver error)\n"

    entry += "-" * 80 + "\n"  # Separator line

    try:
        with open(filename, "a") as f:
            f.write(entry)
        print(f"    Solver state after 6 guesses logged to {filename}")
    except Exception as e:
        print(f"    Warning: Could not log solver state to {filename}: {e}")

class WordleSolver:
    def __init__(self, word_list: List[str], word_length: int = 5, max_guesses: int = 20, word_file_path: str = None):
        """Initialize the solver (like a constructor in C++)."""
        self.word_list = word_list  # List of possible words (like a Fortran array)
        self.word_length = word_length  # Length of words (default 5 for Wordle)
        self.max_guesses = max_guesses  # Max attempts allowed (increased for automated testing)
        self.possible_words = word_list.copy()  # Working copy of word list
        self.guesses = []  # Store guesses made
        self.feedbacks = []  # Store feedback for each guess
        self.word_file_path = word_file_path  # Path to word file for saving when words are removed

    def filter_words_unique_letters(self, word_list: List[str]) -> List[str]:
        """Filter word list to only include words with unique letters."""
        return filter_words_unique_letters(word_list)

    def remove_rejected_word(self, rejected_word: str) -> None:
        """Remove a rejected word from both the main word list and possible words."""
        print(f"    Removing '{rejected_word}' from word lists (rejected by Wordle)")
        self.word_list = remove_word_from_list(self.word_list, rejected_word)
        self.possible_words = remove_word_from_list(self.possible_words, rejected_word)
        print(f"    Word lists updated: {len(self.word_list)} total words, {len(self.possible_words)} possible words")

        # Save updated word list to file if we have a file path
        if self.word_file_path:
            if save_words_to_file(self.word_file_path, self.word_list):
                print(f"    Updated word list saved to {self.word_file_path}")
            else:
                print(f"    Warning: Could not save updated word list to {self.word_file_path}")

    def filter_words(self, guess: str, feedback: str) -> None:
        """Filter possible words based on guess and feedback."""
        old_count = len(self.possible_words)
        self.possible_words = [
            word for word in self.possible_words
            if is_valid_word(word, guess, feedback)
        ]
        new_count = len(self.possible_words)
        print(f"    Filtered from {old_count} to {new_count} possible words")

    def choose_guess_random(self) -> str:
        """Choose a random guess from possible words."""
        if not self.possible_words:
            print("    No possible words left, using random from full list")
            return random.choice(self.word_list)

        # Prefer isograms when it makes sense
        if should_prefer_isograms(self.possible_words, len(self.guesses)):
            isogram_candidates = [word for word in self.possible_words if has_unique_letters(word)]
            if isogram_candidates:
                print(f"    Preferring isograms: choosing from {len(isogram_candidates)} words with unique letters")
                return random.choice(isogram_candidates)

        if len(self.guesses) == 0:
            return "crane"  # Common first guess in Wordle (hardcoded for simplicity)
        return random.choice(self.possible_words)

    def choose_guess_entropy(self, use_optimal_start: bool = False) -> str:
        """Choose a guess based on maximum entropy."""
        if not self.possible_words:
            print("    No possible words left, using random from full list")
            return random.choice(self.word_list)

        # Handle first guess
        if len(self.guesses) == 0:
            if not use_optimal_start:
                return "tares"  # Hardcoded first guess based on results from previous analysis
            else:
                print("    Computing optimal entropy-based first guess from full word list...")

        # Calculate entropy for all words and find the best ones
        search_space = self.word_list if (len(self.guesses) == 0 and use_optimal_start) else self.possible_words

        # Prefer isograms when it makes sense (early in game or when many possibilities are isograms)
        if should_prefer_isograms(self.possible_words, len(self.guesses)):
            unique_search_space = self.filter_words_unique_letters(search_space)
            if unique_search_space:  # Only use if we have unique-letter words available
                search_space = unique_search_space
                print(f"    Preferring isograms: filtered to {len(search_space)} words with unique letters")

        word_entropies = []

        # Add progress indicator for long computations
        total_words_to_check = len(search_space)
        progress_interval = max(1, total_words_to_check // 10)  # Print dot every 10%

        if total_words_to_check > 100:  # Only show progress for longer computations
            print(f"    Computing entropy for {total_words_to_check} words: ", end="", flush=True)

        for i, guess in enumerate(search_space):
            pattern_counts = Counter()
            for possible_target in self.possible_words:
                feedback = get_feedback(guess, possible_target)
                pattern_counts[feedback] += 1

            total_words = len(self.possible_words)
            entropy = 0
            for count in pattern_counts.values():
                probability = count / total_words
                entropy -= probability * math.log2(probability) if probability > 0 else 0

            word_entropies.append((guess, entropy))

            # Print progress dots
            if total_words_to_check > 100 and (i + 1) % progress_interval == 0:
                print(".", end="", flush=True)

        # Finish progress line if we started one
        if total_words_to_check > 100:
            print(" done")

        # Find the maximum entropy
        max_entropy = max(entropy for _, entropy in word_entropies)

        # Get all words with the maximum entropy
        top_words = [word for word, entropy in word_entropies if entropy == max_entropy]

        # Handle ties and print results
        if len(top_words) > 1:
            search_desc = "full word list" if (len(self.guesses) == 0 and use_optimal_start) else "possible words"
            if len(self.guesses) == 0 and unique_search_space:
                search_desc += " (unique letters only)"
            print(f"    Tie between {len(top_words)} words with entropy {max_entropy:.3f} from {search_desc}: {top_words}")
            # For entropy ties, just pick the first one (they're all equally good)
            return top_words[0]
        else:
            search_desc = "full word list" if (len(self.guesses) == 0 and use_optimal_start) else "possible words"
            if len(self.guesses) == 0 and unique_search_space:
                search_desc += " (unique letters only)"
            print(f"    Best word: '{top_words[0]}' with entropy {max_entropy:.3f} from {search_desc}")
            return top_words[0]

    def choose_guess_frequency(self, use_optimal_start: bool = False, start_strategy: str = "fixed") -> str:
        """Choose a guess based on letter frequency/likelihood in each position with entropy tie-breaking."""
        if not self.possible_words:
            print("    No possible words left, using random from full list")
            return random.choice(self.word_list)

        # Handle first guess with different strategies
        if len(self.guesses) == 0:
            if start_strategy == "fixed":
                return "cares"  # Hardcoded first guess based on isogram results
            elif start_strategy == "random":
                chosen = random.choice(self.word_list)
                print(f"    Random first guess: '{chosen}'")
                return chosen
            elif start_strategy == "highest":
                print("    Computing highest frequency-based first guess from full word list (unique letters only)...")
                use_optimal_start = True  # Force optimal computation
            elif start_strategy == "lowest":
                print("    Computing lowest frequency-based first guess from full word list (unique letters only)...")
                use_optimal_start = True  # Force optimal computation
            else:
                return "crane"  # Default fallback

        # Calculate letter frequencies for each position (equivalent to likelihood matrix)
        search_space = self.word_list if (len(self.guesses) == 0 and use_optimal_start) else self.possible_words

        # Prefer isograms when it makes sense (early in game or when many possibilities are isograms)
        if should_prefer_isograms(self.possible_words, len(self.guesses)):
            unique_search_space = self.filter_words_unique_letters(search_space)
            if unique_search_space:  # Only use if we have unique-letter words available
                search_space = unique_search_space
                print(f"    Preferring isograms: filtered to {len(search_space)} words with unique letters")

        total_words = len(self.possible_words)
        freq = [Counter() for _ in range(self.word_length)]

        for word in search_space:
            for i, char in enumerate(word):
                freq[i][char] += 1

        # Score each word based on letter frequencies (raw counts work same as probabilities for ranking)
        word_scores = []
        for word in search_space:
            # Calculate both raw frequency score and normalized likelihood score for display
            freq_score = sum(freq[i][word[i]] for i in range(self.word_length))
            likelihood_score = freq_score / len(search_space)  # Normalized version
            word_scores.append((word, freq_score, likelihood_score))

        # For first guess with lowest strategy, find minimum instead of maximum
        if len(self.guesses) == 0 and start_strategy == "lowest":
            target_score = min(freq_score for _, freq_score, _ in word_scores)
            target_words = [(word, freq_score, likelihood_score) for word, freq_score, likelihood_score in word_scores
                           if freq_score == target_score]
            search_desc = "full word list (lowest frequency, unique letters only)"
        else:
            # Find the maximum frequency score (normal case)
            target_score = max(freq_score for _, freq_score, _ in word_scores)
            target_words = [(word, freq_score, likelihood_score) for word, freq_score, likelihood_score in word_scores
                           if freq_score == target_score]
            search_desc = "full word list" if (len(self.guesses) == 0 and use_optimal_start) else "possible words"
            if len(self.guesses) == 0 and use_optimal_start and unique_search_space:
                search_desc += " (unique letters only)"

        # Handle ties and print results
        if len(target_words) > 1:
            word_list = [word for word, _, _ in target_words]
            likelihood_score = target_words[0][2]  # All tied words have same likelihood score
            score_type = "lowest" if (len(self.guesses) == 0 and start_strategy == "lowest") else "highest"
            print(f"    Tie between {len(target_words)} words with {score_type} frequency score {target_score} (likelihood {likelihood_score:.3f}) from {search_desc}: {word_list}")

            # Use entropy to break ties
            best_guess = None
            best_entropy = -1

            for word, _, _ in target_words:
                pattern_counts = Counter()
                for possible_target in self.possible_words:
                    feedback = get_feedback(word, possible_target)
                    pattern_counts[feedback] += 1

                entropy = 0
                for count in pattern_counts.values():
                    probability = count / total_words
                    entropy -= probability * math.log2(probability) if probability > 0 else 0

                if entropy > best_entropy:
                    best_entropy = entropy
                    best_guess = word

            print(f"    Selected '{best_guess}' based on entropy ({best_entropy:.3f})")
            return best_guess
        else:
            word, freq_score, likelihood_score = target_words[0]
            score_type = "lowest" if (len(self.guesses) == 0 and start_strategy == "lowest") else "highest"
            print(f"    Best word: '{word}' with {score_type} frequency score {freq_score} (likelihood {likelihood_score:.3f}) from {search_desc}")
            return word

    def solve(self, target: str, guess_method: str = "random", start_strategy: str = "fixed") -> Tuple[bool, int]:
        """Attempt to solve Wordle for the given target word using specified guess method.
        guess_method: 'random', 'entropy', or 'frequency'.
        start_strategy: For frequency method: 'fixed', 'random', 'highest', 'lowest'.
        Returns (solved, number_of_guesses)."""
        self.possible_words = self.word_list.copy()
        self.guesses = []
        self.feedbacks = []

        # Select the guessing method
        if guess_method == "random":
            choose_func = lambda: self.choose_guess_random()
        elif guess_method == "entropy":
            choose_func = lambda: self.choose_guess_entropy(False)  # Never use optimal start for entropy
        elif guess_method == "frequency":
            choose_func = lambda: self.choose_guess_frequency(start_strategy=start_strategy)
        else:
            raise ValueError("Invalid guess_method. Use 'random', 'entropy', or 'frequency'.")

        for attempt in range(self.max_guesses):
            guess = choose_func()
            if not guess:
                return False, attempt

            feedback = get_feedback(guess, target)
            self.guesses.append(guess)
            self.feedbacks.append(feedback)

            method_desc = f"{guess_method}"
            if attempt == 0 and start_strategy != "fixed":
                method_desc += f" ({start_strategy} start)"
            print(f"Guess {attempt + 1}: {guess} -> {feedback} (Method: {method_desc})")

            if feedback == 'G' * self.word_length:
                return True, attempt + 1

            self.filter_words(guess, feedback)

            # Log solver state if we just completed the 6th guess without solving
            if attempt + 1 == 6 and feedback != 'G' * self.word_length:
                write_solver_state_after_6(target, guess_method, start_strategy, self.possible_words)

        return False, self.max_guesses

def interactive_mode():
    """Interactive mode for playing Wordle with AI assistance."""
    print("\n" + "="*60)
    print("🎯 INTERACTIVE WORDLE SOLVER")
    print("="*60)

    # Load word list
    word_file_path = "/home/jlighthall/examp/common/words_alpha5.txt"
    word_list = load_words(word_file_path)
    if not word_list:
        print("Word file not found, using fallback list")
        word_list = ["crane", "house", "smile", "grape", "stone", "flame", "lakes"]
        word_file_path = None  # Don't save fallback list
    else:
        print(f"Loaded {len(word_list)} words from file")

    # Choose solving method
    print("\nChoose your AI assistant method:")
    print("1. Random guesses")
    print("2. Entropy-based (information theory)")
    print("3. Frequency-based (letter frequency)")

    while True:
        try:
            method_choice = input("\nEnter choice (1-3): ").strip()
            if method_choice == "1":
                guess_method = "random"
                break
            elif method_choice == "2":
                guess_method = "entropy"
                break
            elif method_choice == "3":
                guess_method = "frequency"
                # Choose start strategy for frequency method
                print("\nChoose starting strategy:")
                print("1. fixed (classic)")
                print("2. random")
                print("3. highest frequency")
                print("4. lowest frequency")

                while True:
                    try:
                        start_choice = input("Enter choice (1-4): ").strip()
                        if start_choice == "1":
                            start_strategy = "fixed"
                            break
                        elif start_choice == "2":
                            start_strategy = "random"
                            break
                        elif start_choice == "3":
                            start_strategy = "highest"
                            break
                        elif start_choice == "4":
                            start_strategy = "lowest"
                            break
                        else:
                            print("Invalid choice. Please enter 1-4.")
                    except KeyboardInterrupt:
                        print("\nGoodbye!")
                        return
                break
            else:
                print("Invalid choice. Please enter 1-3.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            return

    # Initialize solver
    if guess_method == "frequency":
        solver = WordleSolver(word_list, word_file_path=word_file_path)
    else:
        solver = WordleSolver(word_list, word_file_path=word_file_path)
        start_strategy = "fixed"  # Default for non-frequency methods

    print(f"\n🤖 Using {guess_method} method" + (f" with {start_strategy} start" if guess_method == "frequency" and start_strategy != "fixed" else ""))

    # Choose target word mode
    print("\nHow do you want to set the target word?")
    print("1. I'll tell you the target word")
    print("2. Pick a random word for me")
    print("3. I want to play against a real Wordle (I'll input feedback manually)")

    while True:
        try:
            target_choice = input("\nEnter choice (1-3): ").strip()
            if target_choice in ["1", "2", "3"]:
                break
            else:
                print("Invalid choice. Please enter 1-3.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            return

    if target_choice == "1":
        # User provides target word
        while True:
            target = input("\nEnter the target word (5 letters): ").strip().lower()
            if len(target) == 5 and target.isalpha():
                if target in word_list:
                    break
                else:
                    print(f"'{target}' is not in the word list. Continue anyway? (y/n)")
                    if input().strip().lower().startswith('y'):
                        break
            else:
                print("Please enter exactly 5 letters.")

        # Automated solving mode
        print(f"\n🎯 Target word: {target.upper()}")
        print("🤖 AI will solve this automatically...\n")

        solved, attempts = solver.solve(target, guess_method=guess_method, start_strategy=start_strategy)

        if solved:
            print(f"\n🎉 Solved in {attempts} guesses!")
        else:
            print(f"\n😞 Failed to solve after {attempts} guesses.")

    elif target_choice == "2":
        # Random target word
        target = random.choice(word_list)
        print(f"\n🎯 Random target word selected!")
        print("🤖 AI will solve this automatically...\n")

        solved, attempts = solver.solve(target, guess_method=guess_method, start_strategy=start_strategy)

        print(f"\n🎯 The target word was: {target.upper()}")
        if solved:
            print(f"🎉 Solved in {attempts} guesses!")
        else:
            print(f"😞 Failed to solve after {attempts} guesses.")

    else:
        # Manual feedback mode (real Wordle)
        print(f"\n🎮 MANUAL WORDLE MODE")
        print("You'll play on the real Wordle website and input the feedback here.")
        print("Feedback format: G=Green (correct), Y=Yellow (wrong position), X=Gray (not in word)")
        print("Special: R=Rejected (if Wordle doesn't accept the word)")
        print("Example: CRANE -> XYGXX means C=gray, R=yellow, A=green, N=gray, E=gray")
        print("If Wordle rejects a word, just type 'R' and we'll remove it and suggest another word.")
        print("Note: Rejected words don't count toward your 6-guess limit!\n")

        attempt = 0
        max_attempts = 6  # Standard Wordle limit

        while attempt < max_attempts:
            # Get AI suggestion
            if guess_method == "random":
                suggestion = solver.choose_guess_random()
            elif guess_method == "entropy":
                suggestion = solver.choose_guess_entropy(False)
            else:  # frequency
                suggestion = solver.choose_guess_frequency(start_strategy=start_strategy)

            print(f"🤖 Guess {attempt + 1}: I suggest '{suggestion.upper()}'")

            # Get user's actual guess
            while True:
                user_guess = input(f"What word did you actually guess? (or press Enter for '{suggestion}'): ").strip().lower()
                if not user_guess:
                    user_guess = suggestion
                    break
                elif len(user_guess) == 5 and user_guess.isalpha():
                    break
                else:
                    print("Please enter exactly 5 letters.")

            # Get feedback from user
            while True:
                feedback_input = input(f"Enter Wordle feedback for '{user_guess.upper()}' (5 chars: G/Y/X or g/y/x, or 'R' if rejected): ").strip().upper()

                # Check if word was rejected by Wordle
                if feedback_input == 'R':
                    print(f"   '{user_guess.upper()}' was rejected by Wordle - removing from word list")
                    solver.remove_rejected_word(user_guess)
                    print("   Generating new suggestion... (this doesn't count as a guess)")
                    break  # Exit feedback loop to get new suggestion

                # Check for valid feedback
                elif len(feedback_input) == 5 and all(c in 'GYX' for c in feedback_input):
                    feedback = feedback_input
                    print(f"   Result: {user_guess.upper()} -> {feedback}")
                    break
                else:
                    print("Please enter exactly 5 characters using G/Y/X (uppercase or lowercase), or 'R' if rejected.")

            # If word was rejected, continue to get new suggestion (don't increment attempt)
            if feedback_input == 'R':
                continue  # Go back to get new suggestion

            # Only increment attempt counter for accepted words
            attempt += 1

            # Check if solved
            if feedback == "GGGGG":
                print(f"\n🎉 Congratulations! You solved it in {attempt} guesses!")
                print(f"🎯 The word was: {user_guess.upper()}")
                break

            # Update solver with the guess and feedback
            solver.guesses.append(user_guess)
            solver.feedbacks.append(feedback)
            solver.filter_words(user_guess, feedback)

            if not solver.possible_words:
                print("⚠️  No possible words left! There might be an error in the feedback.")
                break

        else:
            print(f"\n😞 Game over! You used all {max_attempts} guesses.")

    # Ask if user wants to play again
    print(f"\nWould you like to play again? (y/n): ", end="")
    try:
        if input().strip().lower().startswith('y'):
            interactive_mode()
    except KeyboardInterrupt:
        pass

    print("Thanks for playing! 🎯")

def main():
    """Main function - choose between interactive mode and automated testing."""
    print("🎯 WORDLE SOLVER")
    print("================")
    print("1. Interactive mode (play with AI assistance)")
    print("2. Automated testing (run test suite)")

    while True:
        try:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            if choice == "1":
                interactive_mode()
                return
            elif choice == "2":
                break  # Continue to automated testing below
            else:
                print("Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            return

    # Original automated testing code
    # Load word list from file
    word_list = load_words("/home/jlighthall/examp/common/words_alpha5.txt")
    if not word_list:
        print("Word file not found, using fallback list")
        word_list = ["crane", "house", "smile", "grape", "stone", "flame", "lakes"]
    else:
        print(f"Loaded {len(word_list)} words from file")

    # Test each guessing method with multiple target words and start strategies
    # Check for past Wordle words file first
    past_words_file = "/home/jlighthall/examp/common/words_past5.txt"
    past_words = load_words(past_words_file)

    if past_words:
        print(f"Found past Wordle words file: {len(past_words)} words")

        # Prompt user for how many words to test
        while True:
            try:
                user_input = input(f"How many words to test? (number, 'all', or Enter for default 10): ").strip().lower()

                if not user_input:  # Empty input (just pressed Enter)
                    num_words = 10
                    break
                elif user_input == "all":
                    num_words = len(past_words)
                    break
                else:
                    num_words = int(user_input)
                    if num_words <= 0:
                        print("Please enter a positive number.")
                        continue
                    elif num_words > len(past_words):
                        print(f"Only {len(past_words)} words available. Using all {len(past_words)} words.")
                        num_words = len(past_words)
                    break
            except ValueError:
                print("Please enter a valid number, 'all', or press Enter for default.")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                return

        test_words = past_words[:num_words]
        print(f"Using {len(test_words)} past Wordle words for testing: {[w.upper() for w in test_words]}")
    else:
        print("Past Wordle words file not found, using default test words")
        test_words = ["smile", "house", "grape"]  # Using fewer words for manageable output
        print(f"Using default test words: {[w.upper() for w in test_words]}")

    methods = ["random", "entropy", "frequency"]

    # Track results for summary
    results = {}

    for target in test_words:
        print(f"\n{'='*80}")
        print(f"Testing target word: {target.upper()}")
        print(f"{'='*80}")

        for method in methods:
            print(f"\n{'-'*60}")
            print(f"Method: {method.upper()}")
            print(f"{'-'*60}")

            if method == "random":
                # Random method only uses random start
                print(f"\nTesting {method} method:")
                solver = WordleSolver(word_list)
                solved, attempts = solver.solve(target, guess_method=method)
                print(format_result(solved, attempts))

                # Log failed words
                if not solved:
                    write_failed_word(target, method)

                # Track results
                key = f"{method}"
                if key not in results:
                    results[key] = []
                results[key].append((solved, attempts))

            elif method == "entropy":
                # Entropy method: only test fixed start (highest is too slow)
                print(f"\nTesting {method} method (fixed start):")
                solver = WordleSolver(word_list)
                solved, attempts = solver.solve(target, guess_method=method, start_strategy="fixed")
                print(format_result(solved, attempts))

                # Log failed words
                if not solved:
                    write_failed_word(target, method, "fixed")

                # Track results
                key = f"{method} (fixed)"
                if key not in results:
                    results[key] = []
                results[key].append((solved, attempts))

            else:  # frequency method
                # Frequency method: only test fixed start (like other methods)
                print(f"\nTesting {method} method (fixed start):")
                solver = WordleSolver(word_list)
                solved, attempts = solver.solve(target, guess_method=method, start_strategy="fixed")
                print(format_result(solved, attempts))

                # Log failed words
                if not solved:
                    write_failed_word(target, method, "fixed")

                # Track results
                key = f"{method} (fixed)"
                if key not in results:
                    results[key] = []
                results[key].append((solved, attempts))

    # Print summary of results
    print(f"\n{'='*80}")
    print("📊 PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Tested {len(test_words)} words: {[w.upper() for w in test_words]}")
    print()

    # Calculate and display statistics for each method
    for method_key in sorted(results.keys()):
        method_results = results[method_key]
        total_tests = len(method_results)
        successful_solves = sum(1 for solved, _ in method_results if solved)
        failed_solves = total_tests - successful_solves

        # Count wins (solved within 6 guesses) and losses (exceeded 6 guesses or failed)
        wins = sum(1 for solved, attempts in method_results if solved and attempts <= 6)
        exceeded_limit = sum(1 for solved, attempts in method_results if attempts > 6)

        if successful_solves > 0:
            successful_attempts = [attempts for solved, attempts in method_results if solved]
            avg_attempts = sum(successful_attempts) / len(successful_attempts)
            min_attempts = min(successful_attempts)
            max_attempts = max(successful_attempts)
        else:
            avg_attempts = 0
            min_attempts = 0
            max_attempts = 0

        # Color code the method name based on overall performance
        if wins == total_tests and avg_attempts <= 4:
            color = Colors.GREEN
        elif wins == total_tests and avg_attempts <= 6:
            color = Colors.YELLOW
        else:
            color = Colors.RED

        print(f"{color}{method_key:20}{Colors.RESET} | ", end="")
        print(f"Solve: {successful_solves:2}/{total_tests} ({successful_solves/total_tests*100:5.1f}%) | ", end="")
        print(f"Win: {wins:2}/{total_tests} ({wins/total_tests*100:5.1f}%) | ", end="")

        if successful_solves > 0:
            avg_color = Colors.GREEN if avg_attempts <= 4 else Colors.YELLOW if avg_attempts <= 6 else Colors.RED
            print(f"Avg: {avg_color}{avg_attempts:4.1f}{Colors.RESET} | ", end="")
            print(f"Range: {min_attempts}-{max_attempts}")
        else:
            print(f"Avg: {Colors.RED} N/A{Colors.RESET} | Range: N/A")

    print(f"\n{Colors.GREEN}Legend:{Colors.RESET}")
    print(f"  {Colors.GREEN}Green{Colors.RESET}: Excellent performance (100% success, avg ≤ 4 guesses)")
    print(f"  {Colors.YELLOW}Yellow{Colors.RESET}: Good performance (100% success, avg ≤ 6 guesses)")
    print(f"  {Colors.RED}Red{Colors.RESET}: Poor performance (failures or avg > 6 guesses)")

if __name__ == "__main__":
    main()