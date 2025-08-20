import random
from collections import Counter
from typing import List, Tuple

class WordleSolver:
    def __init__(self, word_list: List[str], word_length: int = 5, max_guesses: int = 6):
        """Initialize the solver (like a constructor in C++)."""
        self.word_list = word_list  # List of possible words (like a Fortran array)
        self.word_length = word_length  # Length of words (default 5 for Wordle)
        self.max_guesses = max_guesses  # Max attempts allowed
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
        for i in range(self.word_length):
            if feedback[i] == 'G' and word[i] != guess[i]:
                return False
            if feedback[i] == 'Y' and (guess[i] not in word or word[i] == guess[i]):
                return False
            if feedback[i] == 'X' and guess[i] in word:
                return False
        return True

    def filter_words(self, guess: str, feedback: str) -> None:
        """Filter possible words based on guess and feedback."""
        self.possible_words = [
            word for word in self.possible_words
            if self.is_valid_word(word, guess, feedback)
        ]

    def choose_guess(self) -> str:
        """Choose the next guess (simple heuristic: random from possible words)."""
        if not self.possible_words:
            return None
        if len(self.guesses) == 0:
            return "crane"  # Common first guess in Wordle (hardcoded for simplicity)
        return random.choice(self.possible_words)

    def solve(self, target: str) -> Tuple[bool, int]:
        """Attempt to solve Wordle for the given target word.
        Returns (solved, number_of_guesses)."""
        self.possible_words = self.word_list.copy()
        self.guesses = []
        self.feedbacks = []

        for attempt in range(self.max_guesses):
            guess = self.choose_guess()
            if not guess:
                return False, attempt

            feedback = self.get_feedback(guess, target)
            self.guesses.append(guess)
            self.feedbacks.append(feedback)

            print(f"Guess {attempt + 1}: {guess} -> {feedback}")

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

    # Create solver instance
    solver = WordleSolver(word_list)

    # Test with a target word
    target = "smile"  # Change this to test different words
    solved, attempts = solver.solve(target)

    if solved:
        print(f"Solved in {attempts} guesses!")
    else:
        print(f"Failed to solve after {attempts} guesses.")

if __name__ == "__main__":
    main()