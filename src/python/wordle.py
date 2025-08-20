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
        # Check green letters (correct position)
        for i in range(self.word_length):
            if feedback[i] == 'G' and word[i] != guess[i]:
                return False

        # Check yellow letters (correct letter, wrong position)
        for i in range(self.word_length):
            if feedback[i] == 'Y':
                # The letter must be in the word somewhere
                if guess[i] not in word:
                    return False
                # But not in the same position as the guess
                if word[i] == guess[i]:
                    return False

        # Check gray letters (letter not in word)
        for i in range(self.word_length):
            if feedback[i] == 'X':
                # The letter should not appear in the word at all
                # BUT only if it's not marked as yellow or green elsewhere
                letter = guess[i]
                # Check if this letter appears as green or yellow elsewhere in this guess
                appears_elsewhere = False
                for j in range(self.word_length):
                    if j != i and guess[j] == letter and feedback[j] in ['G', 'Y']:
                        appears_elsewhere = True
                        break

                if not appears_elsewhere and letter in word:
                    return False

        return True

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

    def choose_guess_entropy(self) -> str:
        """Choose a guess based on maximum entropy."""
        if not self.possible_words:
            print("    No possible words left, using random from full list")
            return random.choice(self.word_list)
        if len(self.guesses) == 0:
            return "crane"  # Hardcoded first guess for simplicity

        # Calculate entropy for all words and find the best ones
        word_entropies = []
        for guess in self.possible_words:
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
            print(f"    Tie between {len(top_words)} words with entropy {max_entropy:.3f}: {top_words}")
            # For entropy ties, just pick the first one (they're all equally good)
            return top_words[0]
        else:
            print(f"    Best word: '{top_words[0]}' with entropy {max_entropy:.3f}")
            return top_words[0]

    def choose_guess_frequency(self) -> str:
        """Choose a guess based on letter frequency in each position."""
        if not self.possible_words:
            print("    No possible words left, using random from full list")
            return random.choice(self.word_list)
        if len(self.guesses) == 0:
            return "crane"  # Hardcoded first guess

        # Calculate letter frequencies for each position
        freq = [Counter() for _ in range(self.word_length)]
        for word in self.possible_words:
            for i, char in enumerate(word):
                freq[i][char] += 1

        # Score each word based on letter frequencies
        word_scores = []
        for word in self.possible_words:
            score = 0
            for i, char in enumerate(word):
                score += freq[i][char]  # Sum frequency of each letter in its position
            word_scores.append((word, score))

        # Find the maximum score
        max_score = max(score for _, score in word_scores)

        # Get all words with the maximum score
        top_words = [word for word, score in word_scores if score == max_score]

        # Handle ties and print results
        if len(top_words) > 1:
            print(f"    Tie between {len(top_words)} words with frequency score {max_score}: {top_words}")

            # Use entropy to break ties
            best_guess = None
            best_entropy = -1

            for guess in top_words:
                pattern_counts = Counter()
                for possible_target in self.possible_words:
                    feedback = self.get_feedback(guess, possible_target)
                    pattern_counts[feedback] += 1

                total_words = len(self.possible_words)
                entropy = 0
                for count in pattern_counts.values():
                    probability = count / total_words
                    entropy -= probability * math.log2(probability) if probability > 0 else 0

                if entropy > best_entropy:
                    best_entropy = entropy
                    best_guess = guess

            print(f"    Selected '{best_guess}' based on entropy ({best_entropy:.3f})")
            return best_guess
        else:
            print(f"    Best word: '{top_words[0]}' with frequency score {max_score}")
            return top_words[0]

    def choose_guess_likelihood(self) -> str:
        """Choose a guess based on letter likelihood in each position with tie-breaking."""
        if not self.possible_words:
            print("    No possible words left, using random from full list")
            return random.choice(self.word_list)
        if len(self.guesses) == 0:
            return "crane"  # Hardcoded first guess

        # Step 1: Calculate likelihood matrix (5x26 array of probabilities)
        total_words = len(self.possible_words)
        likelihood_matrix = []

        for pos in range(self.word_length):
            letter_counts = Counter()
            for word in self.possible_words:
                letter_counts[word[pos]] += 1

            # Convert counts to likelihoods (probabilities)
            position_likelihoods = {}
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                position_likelihoods[letter] = letter_counts.get(letter, 0) / total_words

            likelihood_matrix.append(position_likelihoods)

        # Step 2: Score each word based on cumulative likelihood
        word_scores = {}
        for word in self.possible_words:
            score = 0
            for i, char in enumerate(word):
                score += likelihood_matrix[i][char]
            word_scores[word] = score

        # Step 3: Find words with the highest score
        max_score = max(word_scores.values())
        top_words = [word for word, score in word_scores.items() if score == max_score]

        # Step 4: Handle ties
        if len(top_words) > 1:
            print(f"    Tie between {len(top_words)} words with score {max_score:.3f}: {top_words}")

            # Use entropy to break ties
            best_guess = None
            best_entropy = -1

            for guess in top_words:
                pattern_counts = Counter()
                for possible_target in self.possible_words:
                    feedback = self.get_feedback(guess, possible_target)
                    pattern_counts[feedback] += 1

                entropy = 0
                for count in pattern_counts.values():
                    probability = count / total_words
                    entropy -= probability * math.log2(probability) if probability > 0 else 0

                if entropy > best_entropy:
                    best_entropy = entropy
                    best_guess = guess

            print(f"    Selected '{best_guess}' based on entropy ({best_entropy:.3f})")
            return best_guess
        else:
            print(f"    Best word: '{top_words[0]}' with score {max_score:.3f}")
            return top_words[0]

    def solve(self, target: str, guess_method: str = "random") -> Tuple[bool, int]:
        """Attempt to solve Wordle for the given target word using specified guess method.
        guess_method: 'random', 'entropy', 'frequency', or 'likelihood'.
        Returns (solved, number_of_guesses)."""
        self.possible_words = self.word_list.copy()
        self.guesses = []
        self.feedbacks = []

        # Select the guessing method
        if guess_method == "random":
            choose_func = self.choose_guess_random
        elif guess_method == "entropy":
            choose_func = self.choose_guess_entropy
        elif guess_method == "frequency":
            choose_func = self.choose_guess_frequency
        elif guess_method == "likelihood":
            choose_func = self.choose_guess_likelihood
        else:
            raise ValueError("Invalid guess_method. Use 'random', 'entropy', 'frequency', or 'likelihood'.")

        for attempt in range(self.max_guesses):
            guess = choose_func()
            if not guess:
                return False, attempt

            feedback = self.get_feedback(guess, target)
            self.guesses.append(guess)
            self.feedbacks.append(feedback)

            print(f"Guess {attempt + 1}: {guess} -> {feedback} (Method: {guess_method})")

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

    # Test each guessing method with multiple target words
    test_words = ["smile", "crane", "house", "grape", "stone"]
    methods = ["random", "entropy", "frequency", "likelihood"]

    for target in test_words:
        print(f"\n{'='*50}")
        print(f"Testing target word: {target.upper()}")
        print(f"{'='*50}")

        for method in methods:
            print(f"\nTesting {method} method:")
            solver = WordleSolver(word_list)
            solved, attempts = solver.solve(target, guess_method=method)
            if solved:
                print(f"✓ Solved in {attempts} guesses!")
            else:
                print(f"✗ Failed to solve after {attempts} guesses.")

if __name__ == "__main__":
    main()