import random
import math
from collections import Counter
from typing import List, Tuple

class WordleSolver:
    def __init__(self, word_list: List[str], word_length: int = 5, max_guesses: int = 20):
        """Initialize the solver (like a constructor in C++)."""
        self.word_list = word_list  # List of possible words (like a Fortran array)
        self.word_length = word_length  # Length of words (default 5 for Wordle)
        self.max_guesses = max_guesses  # Max attempts allowed (increased for automated testing)
        self.possible_words = word_list.copy()  # Working copy of word list
        self.guesses = []  # Store guesses made
        self.feedbacks = []  # Store feedback for each guess

    def get_feedback(self, guess: str, target: str) -> str:
        """Generate feedback for a guess against the target word.
        Returns a string of 'G' (green), 'Y' (yellow), 'X' (gray)."""
        feedback = ['X'] * self.word_length
        target_chars = list(target)

        # First pass: Mark green (correct letter, correct position)
        for i in range(self.word_length):
            if guess[i] == target[i]:
                feedback[i] = 'G'
                target_chars[i] = None  # Remove matched letter

        # Second pass: Mark yellow (correct letter, wrong position)
        for i in range(self.word_length):
            if feedback[i] == 'G':
                continue
            if guess[i] in target_chars:
                feedback[i] = 'Y'
                target_chars[target_chars.index(guess[i])] = None

        return ''.join(feedback)

    def is_valid_word(self, word: str, guess: str, feedback: str) -> bool:
        """Check if a word is consistent with the guess and its feedback."""
        # Track required letters (from green/yellow) and their counts
        required_letters = Counter()
        for i in range(self.word_length):
            if feedback[i] in ('G', 'Y'):
                required_letters[guess[i]] += 1

        # Count letters in the candidate word
        word_letters = Counter(word)

        # Ensure all required letters appear with at least the required frequency
        for letter, count in required_letters.items():
            if word_letters[letter] < count:
                return False

        # Check position-specific constraints
        for i in range(self.word_length):
            if feedback[i] == 'G' and word[i] != guess[i]:
                return False  # Must match exactly for green
            if feedback[i] == 'Y' and (guess[i] not in word or word[i] == guess[i]):
                return False  # Must be in word but not in this position for yellow
            if feedback[i] == 'X' and guess[i] in word:
                # For gray, letter can appear only if required by green/yellow elsewhere
                if word_letters[guess[i]] > required_letters[guess[i]]:
                    return False

        return True

    def has_unique_letters(self, word: str) -> bool:
        """Check if a word has all unique letters (no repeating letters)."""
        return len(set(word)) == len(word)

    def filter_words_unique_letters(self, word_list: List[str]) -> List[str]:
        """Filter word list to only include words with unique letters."""
        return [word for word in word_list if self.has_unique_letters(word)]

    def filter_words(self, guess: str, feedback: str) -> None:
        """Filter possible words based on guess and feedback."""
        old_count = len(self.possible_words)
        self.possible_words = [
            word for word in self.possible_words
            if self.is_valid_word(word, guess, feedback)
        ]
        new_count = len(self.possible_words)
        print(f"    Filtered from {old_count} to {new_count} possible words")

    def choose_guess_random(self) -> str:
        """Choose a random guess from possible words."""
        if not self.possible_words:
            print("    No possible words left, using random from full list")
            return random.choice(self.word_list)
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
                return "crane"  # Hardcoded first guess for simplicity
            else:
                print("    Computing optimal entropy-based first guess from full word list...")

        # Calculate entropy for all words and find the best ones
        search_space = self.word_list if (len(self.guesses) == 0 and use_optimal_start) else self.possible_words

        # For first guess, filter to unique letters only to speed up calculation and get better starting words
        if len(self.guesses) == 0:
            unique_search_space = self.filter_words_unique_letters(search_space)
            if unique_search_space:  # Only use if we have unique-letter words available
                search_space = unique_search_space
                print(f"    Filtered to {len(search_space)} words with unique letters for first guess")

        word_entropies = []

        for guess in search_space:
            pattern_counts = Counter()
            for possible_target in self.possible_words:
                feedback = self.get_feedback(guess, possible_target)
                pattern_counts[feedback] += 1

            total_words = len(self.possible_words)
            entropy = 0
            for count in pattern_counts.values():
                probability = count / total_words
                entropy -= probability * math.log2(probability) if probability > 0 else 0

            word_entropies.append((guess, entropy))

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

    def choose_guess_frequency(self, use_optimal_start: bool = False, start_strategy: str = "crane") -> str:
        """Choose a guess based on letter frequency/likelihood in each position with entropy tie-breaking."""
        if not self.possible_words:
            print("    No possible words left, using random from full list")
            return random.choice(self.word_list)

        # Handle first guess with different strategies
        if len(self.guesses) == 0:
            if start_strategy == "crane":
                return "crane"  # Hardcoded first guess
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

        # For first guess, filter to unique letters only to prevent repeated letters like "sanes"
        if len(self.guesses) == 0 and use_optimal_start:
            unique_search_space = self.filter_words_unique_letters(search_space)
            if unique_search_space:  # Only use if we have unique-letter words available
                search_space = unique_search_space
                print(f"    Filtered to {len(search_space)} words with unique letters for first guess")

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
                    feedback = self.get_feedback(word, possible_target)
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

    def solve(self, target: str, guess_method: str = "random", start_strategy: str = "crane") -> Tuple[bool, int]:
        """Attempt to solve Wordle for the given target word using specified guess method.
        guess_method: 'random', 'entropy', or 'frequency'.
        start_strategy: For frequency method: 'crane', 'random', 'highest', 'lowest'.
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

            feedback = self.get_feedback(guess, target)
            self.guesses.append(guess)
            self.feedbacks.append(feedback)

            method_desc = f"{guess_method}"
            if attempt == 0 and start_strategy != "crane":
                method_desc += f" ({start_strategy} start)"
            print(f"Guess {attempt + 1}: {guess} -> {feedback} (Method: {method_desc})")

            if feedback == 'G' * self.word_length:
                return True, attempt + 1

            self.filter_words(guess, feedback)

        return False, self.max_guesses

def main():
    # Load word list from file
    try:
        with open("/home/jlighthall/examp/common/words_alpha5.txt", "r") as f:
            word_list = [word.strip() for word in f.readlines()]
        print(f"Loaded {len(word_list)} words from file")
    except FileNotFoundError:
        print("Word file not found, using fallback list")
        word_list = ["crane", "house", "smile", "grape", "stone", "flame", "lakes"]

    # Test each guessing method with multiple target words and start strategies
    test_words = ["smile", "house", "grape"]  # Using fewer words for manageable output
    methods = ["random", "entropy", "frequency"]
    start_strategies = ["crane", "random", "highest", "lowest"]

    for target in test_words:
        print(f"\n{'='*80}")
        print(f"Testing target word: {target.upper()}")
        print(f"{'='*80}")

        for method in methods:
            print(f"\n{'-'*60}")
            print(f"Method: {method.upper()}")
            print(f"{'-'*60}")

            if method == "random":
                # Random method only uses crane start
                print(f"\nTesting {method} method:")
                solver = WordleSolver(word_list)
                solved, attempts = solver.solve(target, guess_method=method)
                if solved:
                    print(f"✓ Solved in {attempts} guesses!")
                else:
                    print(f"✗ Failed to solve after {attempts} guesses.")

            elif method == "entropy":
                # Entropy method: only test crane start (highest is too slow)
                print(f"\nTesting {method} method (crane start):")
                solver = WordleSolver(word_list)
                solved, attempts = solver.solve(target, guess_method=method, start_strategy="crane")
                if solved:
                    print(f"✓ Solved in {attempts} guesses!")
                else:
                    print(f"✗ Failed to solve after {attempts} guesses.")

            else:  # frequency method
                # Frequency method: test all start strategies
                for strategy in start_strategies:
                    print(f"\nTesting {method} method ({strategy} start):")
                    solver = WordleSolver(word_list)
                    solved, attempts = solver.solve(target, guess_method=method, start_strategy=strategy)
                    if solved:
                        print(f"✓ Solved in {attempts} guesses!")
                    else:
                        print(f"✗ Failed to solve after {attempts} guesses.")

if __name__ == "__main__":
    main()