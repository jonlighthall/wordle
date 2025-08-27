import random
import math
import os
import datetime
from collections import Counter
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
from ..core.wordle_utils import get_feedback, calculate_entropy, has_unique_letters, is_valid_word, load_words, filter_words_unique_letters, filter_wordle_appropriate, should_prefer_isograms, remove_word_from_list, save_words_to_file, get_word_information_score

# Import wordfreq for real-world word frequency data
try:
    from wordfreq import word_frequency
    WORDFREQ_AVAILABLE = True
except ImportError:
    WORDFREQ_AVAILABLE = False
    print("Warning: wordfreq not available. Install with 'pip install wordfreq' for improved scoring.")

# Get the repository root directory (3 levels up from this file)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
LOGS_DIR = os.path.join(REPO_ROOT, 'logs')

# Default starting words for algorithms
# To change the default starting words, modify these constants:
DEFAULT_ENTROPY_STARTER = "tares"      # Used by entropy and adaptive hybrid algorithms
DEFAULT_LETTFREQ_STARTER = "cares"    # Used by letter frequency algorithm and random fallback
DEFAULT_INFORMATION_STARTER = "enzym"  # Used by information algorithm (balances letter frequency and diversity)
DEFAULT_STARTER = "crane"              # Used by ultra-efficient and some fallback cases

# Popular Wordle starting words that users often choose
POPULAR_WORDS = ["crane", "slate", "trace", "stare", "audio", "ratio", "penis"]

def get_word_frequency_score(word: str, lang: str = "en") -> float:
    """Get real-world frequency score for a word using wordfreq library.

    Returns a score from 0.0 to 1.0 where 1.0 is most common.
    If wordfreq is not available, returns 0.5 as neutral score.
    """
    if not WORDFREQ_AVAILABLE:
        return 0.5

    try:
        # Get frequency (ranges from ~1e-7 for very rare words to ~0.01 for very common)
        freq = word_frequency(word, lang)

        # Scale to 0-1 range using log scale to handle the wide range
        # Most common words are around 1e-2, rare words around 1e-7
        if freq > 0:
            # Use log scale: log10(1e-7) = -7, log10(1e-2) = -2
            log_freq = math.log10(freq)
            # Scale from [-7, -2] to [0, 1]
            score = max(0.0, min(1.0, (log_freq + 7) / 5))
            return score
        else:
            return 0.0  # Word not found in frequency data
    except Exception:
        return 0.5  # Fallback score

def score_words_by_frequency(words: List[str]) -> List[Tuple[str, float]]:
    """Score a list of words by their real-world frequency.

    Returns list of (word, frequency_score) tuples sorted by frequency score descending.
    """
    scored_words = []
    for word in words:
        score = get_word_frequency_score(word)
        scored_words.append((word, score))

    # Sort by frequency score descending (most common first)
    scored_words.sort(key=lambda x: x[1], reverse=True)
    return scored_words

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'  # Reset to default color

def calculate_entropy_for_word(args):
    """Helper function for parallel entropy calculation.

    Args:
        args: Tuple of (guess_word, possible_words_list)

    Returns:
        Tuple of (guess_word, entropy_value)
    """
    guess, possible_words = args
    pattern_counts = Counter()

    for possible_target in possible_words:
        feedback = get_feedback(guess, possible_target)
        pattern_counts[feedback] += 1

    total_words = len(possible_words)
    entropy = 0
    for count in pattern_counts.values():
        probability = count / total_words
        entropy -= probability * math.log2(probability) if probability > 0 else 0

    return (guess, entropy)





def get_universal_optimal_starter(method: str = "entropy", strategy: str = "general") -> str:
    """Get universal optimal starting words based on extensive pre-analysis.

    These are computed offline based on performance against large datasets,
    not dependent on the current target word (which is unknown at start).
    """

    # Updated optimal starters based on the latest analysis results
    # These were determined by analyzing reduction percentages and success rates

    optimal_starters = {
        "entropy": {
            "general": DEFAULT_ENTROPY_STARTER,      # Best overall entropy and proven track record
            "aggressive": "slate",   # Maximum elimination for difficult words
            "balanced": "adieu",     # Good balance covering vowels
            "conservative": DEFAULT_LETTFREQ_STARTER  # Current best performing starter
        },
        "frequency": {
            "general": DEFAULT_LETTFREQ_STARTER,      # Best letter frequency-based performance
            "common": "cares",       # Focus on most common letters
            "balanced": "arose",     # Balance of frequency and position
            "vowel_focus": "adieu"   # When expecting vowel-heavy targets
        },
        "information": {
            "general": DEFAULT_INFORMATION_STARTER,  # Best for hybrid information approach
            "balanced": "cares",                     # Good information + entropy balance
            "precise": "stare"                       # High precision for end-game
        },
        "smart_hybrid": {
            "general": DEFAULT_STARTER,      # Proven best all-around starter
            "challenging": "slate",  # For difficult word sets
            "balanced": "adieu"      # Vowel coverage for diverse targets
        }
    }

    return optimal_starters.get(method, {}).get(strategy, DEFAULT_STARTER)



def analyze_word_reduction(results_by_algorithm: dict) -> dict:
    """Analyze word list reduction effectiveness for each algorithm."""
    analysis = {}

    for algorithm_key, algorithm_results in results_by_algorithm.items():
        if not algorithm_results:
            continue

        # Collect reduction statistics
        all_reductions = []
        step_reductions = {}  # Track reduction at each step number

        for result in algorithm_results:
            word_sizes = result['word_list_sizes']

            # Calculate percentage reductions at each step
            for step in range(1, len(word_sizes)):
                if word_sizes[step-1] > 0:
                    reduction_pct = (word_sizes[step-1] - word_sizes[step]) / word_sizes[step-1] * 100

                    if step not in step_reductions:
                        step_reductions[step] = []
                    step_reductions[step].append(reduction_pct)
                    all_reductions.append(reduction_pct)

        # Calculate statistics
        if all_reductions:
            avg_reduction = sum(all_reductions) / len(all_reductions)
            max_reduction = max(all_reductions)

            # Calculate average reduction by step
            avg_by_step = {}
            for step, reductions in step_reductions.items():
                avg_by_step[step] = sum(reductions) / len(reductions)

            analysis[algorithm_key] = {
                'avg_reduction_per_step': avg_reduction,
                'max_reduction_seen': max_reduction,
                'avg_by_step': avg_by_step,
                'total_reductions': len(all_reductions)
            }

    return analysis

def analyze_algorithm_performance_by_step(results_by_algorithm: dict) -> dict:
    """Analyze which algorithm performs better at each step of the game."""
    step_performance = {}

    # Get entropy and frequency results
    entropy_results = results_by_algorithm.get("entropy (fixed)", [])
    frequency_results = results_by_algorithm.get("frequency (fixed)", [])

    if not entropy_results or not frequency_results:
        return {}

    # Analyze each step (1-6 for standard Wordle)
    for step in range(1, 7):
        entropy_reductions = []
        frequency_reductions = []

        # Collect reduction data for this step
        for result in entropy_results:
            word_sizes = result['word_list_sizes']
            if step < len(word_sizes) and word_sizes[step-1] > 0:
                reduction_pct = (word_sizes[step-1] - word_sizes[step]) / word_sizes[step-1] * 100
                entropy_reductions.append(reduction_pct)

        for result in frequency_results:
            word_sizes = result['word_list_sizes']
            if step < len(word_sizes) and word_sizes[step-1] > 0:
                reduction_pct = (word_sizes[step-1] - word_sizes[step]) / word_sizes[step-1] * 100
                frequency_reductions.append(reduction_pct)

        # Calculate averages for this step
        if entropy_reductions and frequency_reductions:
            entropy_avg = sum(entropy_reductions) / len(entropy_reductions)
            frequency_avg = sum(frequency_reductions) / len(frequency_reductions)

            # Determine which is better and calculate weights based on actual advantage
            if entropy_avg > frequency_avg:
                better_algorithm = "entropy"
                advantage = entropy_avg - frequency_avg
                # More aggressive weighting: 0.1-0.2 advantage = +0.1 weight, 0.3+ advantage = +0.2 weight
                if advantage >= 3.0:  # 3%+ advantage gets strong weighting
                    entropy_weight = 0.7
                elif advantage >= 1.0:  # 1-3% advantage gets moderate weighting
                    entropy_weight = 0.6
                else:  # <1% advantage stays balanced
                    entropy_weight = 0.5
                frequency_weight = 1.0 - entropy_weight
            else:
                better_algorithm = "frequency"
                advantage = frequency_avg - entropy_avg
                # Apply same logic for frequency advantages
                if advantage >= 3.0:
                    frequency_weight = 0.7
                elif advantage >= 1.0:
                    frequency_weight = 0.6
                else:
                    frequency_weight = 0.5
                entropy_weight = 1.0 - frequency_weight

            step_performance[step] = {
                'entropy_avg_reduction': entropy_avg,
                'frequency_avg_reduction': frequency_avg,
                'better_algorithm': better_algorithm,
                'advantage': advantage,
                'entropy_weight': entropy_weight,
                'frequency_weight': frequency_weight
            }

    return step_performance

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

    # Create filename with current date in the logs directory
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    filename = os.path.join(LOGS_DIR, f"failed_words_{date_str}.txt")

    # Format the entry
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} | {word.upper()} | {method}"
    if strategy != "fixed":
        entry += f" ({strategy})"
    entry += "\n"

    try:
        # Ensure logs directory exists
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(filename, "a") as f:
            f.write(entry)
        print(f"    Failed word logged to {filename}")
    except Exception as e:
        print(f"    Warning: Could not log failed word to {filename}: {e}")

    # Also save to simple words_missing.txt file
    try:
        with open(os.path.join(DATA_DIR, "words_missing.txt"), "a") as f:
            f.write(f"{word.upper()}\n")
        print(f"    Failed word added to words_missing.txt")
    except Exception as e:
        print(f"    Warning: Could not add word to words_missing.txt: {e}")

def write_challenging_word(word: str):
    """Write a word that took more than 6 guesses to solve."""
    try:
        with open(os.path.join(DATA_DIR, "words_challenging.txt"), "a") as f:
            f.write(f"{word.upper()}\n")
        print(f"    Challenging word added to words_challenging.txt")
    except Exception as e:
        print(f"    Warning: Could not add word to words_challenging.txt: {e}")

def write_solver_state_after_6(target: str, method: str, strategy: str, possible_words: List[str]):
    """Write solver state when it exceeds 6 guesses (Wordle limit)."""
    import datetime

    # Create filename with current date in the logs directory
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    filename = os.path.join(LOGS_DIR, f"solver_state_after_6_{date_str}.log")

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
        # Ensure logs directory exists
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(filename, "a") as f:
            f.write(entry)
        print(f"    Solver state after 6 guesses logged to {filename}")
    except Exception as e:
        print(f"    Warning: Could not log solver state to {filename}: {e}")

class WordleSolver:
    def __init__(self, word_list: List[str], word_length: int = 5, max_guesses: int = 20, word_file_path: str = None, hard_mode: bool = False):
        """Initialize the solver (like a constructor in C++)."""
        self.word_list = word_list  # List of possible words (like a Fortran array)
        self.word_length = word_length  # Length of words (default 5 for Wordle)
        self.max_guesses = max_guesses  # Max attempts allowed (increased for automated testing)
        self.possible_words = word_list.copy()  # Working copy of word list
        self.guesses = []  # Store guesses made
        self.feedbacks = []  # Store feedback for each guess
        self.word_file_path = word_file_path  # Path to word file for saving when words are removed
        self.hard_mode = hard_mode  # Force solver to use revealed information

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

    def filter_words(self, guess: str, feedback: str, verbose: bool = True) -> None:
        """Filter possible words based on guess and feedback."""
        old_count = len(self.possible_words)
        self.possible_words = [
            word for word in self.possible_words
            if is_valid_word(word, guess, feedback)
        ]
        new_count = len(self.possible_words)
        if verbose:
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
            return DEFAULT_LETTFREQ_STARTER
        return random.choice(self.possible_words)

    def choose_guess_entropy(self, use_optimal_start: bool = False) -> str:
        """Choose a guess based on maximum entropy with enhanced early termination."""
        if not self.possible_words:
            print("    No possible words left, using random from full list")
            return random.choice(self.word_list)

        # Handle first guess
        if len(self.guesses) == 0:
            if not use_optimal_start:
                return get_universal_optimal_starter("entropy", "general")  # Use improved starter
            else:
                print("    Computing optimal entropy-based first guess from full word list...")

        # Enhanced early termination: switch to direct guessing sooner
        num_possible = len(self.possible_words)
        attempt = len(self.guesses)

        # More aggressive early termination based on attempt number
        if attempt >= 3 and num_possible <= 4:
            print(f"    Entropy: attempt {attempt+1}, {num_possible} words left, switching to direct guessing")
            return self.possible_words[0]
        elif attempt >= 2 and num_possible <= 2:
            print(f"    Entropy: attempt {attempt+1}, {num_possible} words left, switching to direct guessing")
            return self.possible_words[0]
        elif num_possible <= 1:
            return self.possible_words[0] if self.possible_words else random.choice(self.word_list)

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

        if total_words_to_check > 100:  # Only show progress for longer computations
            print(f"    Computing entropy for {total_words_to_check} words: ", end="", flush=True)

        # Use parallel processing for large computations
        if total_words_to_check > 50:  # Use parallel processing for 50+ words
            # Prepare arguments for parallel processing
            args_list = [(guess, self.possible_words) for guess in search_space]

            # Use number of CPU cores, but cap at reasonable limit
            num_processes = min(cpu_count(), max(2, total_words_to_check // 20))

            try:
                with Pool(processes=num_processes) as pool:
                    word_entropies = pool.map(calculate_entropy_for_word, args_list)

                if total_words_to_check > 100:
                    print(" done (parallel)")

            except Exception as e:
                # Fall back to sequential processing if parallel fails
                print(f" (parallel failed: {e}, using sequential)")
                word_entropies = []
                progress_interval = max(1, total_words_to_check // 10)

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

                if total_words_to_check > 100:
                    print(" done")
        else:
            # Use sequential processing for smaller computations
            progress_interval = max(1, total_words_to_check // 10)

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
                return DEFAULT_LETTFREQ_STARTER  # Optimal letter frequency-based starting word
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
                return DEFAULT_STARTER  # Default fallback

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

    def choose_guess_information(self, use_optimal_start: bool = False) -> str:
        """Choose a guess based on information content and position frequencies.

        This method uses the information score directly without double-counting entropy.
        The information score already balances frequency analysis and diversity.
        """
        if not self.possible_words:
            print("    No possible words left, using random from full list")
            return random.choice(self.word_list)

        # Handle first guess
        if len(self.guesses) == 0:
            return get_universal_optimal_starter("information", "general")

        # If very few possibilities left, just guess one of them
        if len(self.possible_words) <= 2:
            return self.possible_words[0]

        # For subsequent guesses, use possible words for search
        search_space = self.possible_words.copy()

        # Prefer isograms when it makes sense
        if should_prefer_isograms(self.possible_words, len(self.guesses)):
            unique_search_space = self.filter_words_unique_letters(search_space)
            if unique_search_space and len(unique_search_space) >= len(search_space) * 0.3:
                search_space = unique_search_space
                print(f"    Preferring isograms: filtered to {len(search_space)} words with unique letters")

        # Calculate information scores (no entropy mixing to avoid double counting)
        word_scores = []

        # Add progress indicator for long computations
        total_words_to_check = len(search_space)

        if total_words_to_check > 100:
            print(f"    Computing information scores for {total_words_to_check} words: ", end="", flush=True)

        # Use parallel processing for large computations
        if total_words_to_check > 50:
            # Use information score only (no entropy mixing)
            args_list = [(word, self.possible_words) for word in search_space]
            num_processes = min(cpu_count(), max(2, total_words_to_check // 20))

            try:
                with Pool(processes=num_processes) as pool:
                    # Get information scores directly without entropy mixing
                    results = pool.map(lambda args: (args[0], get_word_information_score(args[0], args[1])), args_list)
                    word_scores = [(word, info_score, info_score, 0.0) for word, info_score in results]

                if total_words_to_check > 100:
                    print(" done (parallel)")

            except Exception as e:
                print(f" (parallel failed: {e}, using sequential)")
                word_scores = []
                for word in search_space:
                    info_score = get_word_information_score(word, self.possible_words)
                    word_scores.append((word, info_score, info_score, 0.0))

                if total_words_to_check > 100:
                    print(" done")
        else:
            # Sequential processing
            for word in search_space:
                info_score = get_word_information_score(word, self.possible_words)
                word_scores.append((word, info_score, info_score, 0.0))

        # Find the best word based on information score only
        best_word, best_combined, best_info, _ = max(word_scores, key=lambda x: x[1])

        print(f"    Best word: '{best_word}' with info score {best_info:.3f} from possible words")
        return best_word

    def is_valid_hard_mode_guess(self, guess: str) -> bool:
        """Check if a guess is valid in hard mode (must use all revealed information)."""
        if not self.hard_mode or not self.guesses:
            return True

        # Check against each previous guess/feedback pair
        for prev_guess, prev_feedback in zip(self.guesses, self.feedbacks):
            for i, (g_char, f_char) in enumerate(zip(prev_guess, prev_feedback)):
                if f_char == 'G':  # Green letter must be in same position
                    if guess[i] != g_char:
                        return False
                elif f_char == 'Y':  # Yellow letter must be somewhere else in the word
                    if g_char not in guess or guess[i] == g_char:
                        return False

        return True



    def choose_guess_ultra_efficient(self) -> str:
        """Ultra-efficient algorithm that prioritizes speed over theoretical optimality.

        This algorithm combines multiple strategies:
        - Uses entropy algorithm for maximum elimination when many words remain
        - Uses frequency-based tie-breaking (inherited from entropy calculations)
        - Employs aggressive early termination for speed
        - Switches to direct guessing much earlier than pure entropy
        """
        if not self.possible_words:
            print("    No possible words left, using random from full list")
            return random.choice(self.word_list)

        num_possible = len(self.possible_words)
        attempt = len(self.guesses)

        # First guess: use proven optimal starter
        if attempt == 0:
            starter = get_universal_optimal_starter("entropy", "aggressive")  # "slate" - unique starter
            return starter

        # Very early termination for small word lists
        if num_possible <= 3:
            print(f"    Ultra efficient: {num_possible} words left, guessing directly")
            return self.possible_words[0]

        # Smart threshold adjustment based on attempt number
        # Switch to direct guessing much earlier
        if attempt >= 3 and num_possible <= 8:
            print(f"    Ultra efficient: attempt {attempt+1}, {num_possible} words left, guessing directly")
            return self.possible_words[0]

        if attempt >= 2 and num_possible <= 5:
            print(f"    Ultra efficient: attempt {attempt+1}, {num_possible} words left, guessing directly")
            return self.possible_words[0]

        # Use pure entropy for maximum elimination when many words remain
        if num_possible > 20:
            print(f"    Ultra efficient: {num_possible} words remaining, using entropy strategy")
            return self.choose_guess_entropy(False)

        # Mid-game: prefer words from possible list to maximize chance of lucky guess
        elif num_possible > 8:
            print(f"    Ultra efficient: {num_possible} words remaining, preferring possible words")
            # Mix of entropy and direct guessing - choose from possible words
            search_space = self.possible_words[:min(20, len(self.possible_words))]

            best_word = None
            best_entropy = -1

            for word in search_space:
                entropy = calculate_entropy(word, self.possible_words)
                if entropy > best_entropy:
                    best_entropy = entropy
                    best_word = word

            print(f"    Best possible word: '{best_word}' with entropy {best_entropy:.3f}")
            return best_word

        # Late game: direct guessing
        else:
            print(f"    Ultra efficient: {num_possible} words remaining, guessing directly")
            return self.possible_words[0]

    def choose_guess_adaptive_hybrid(self) -> str:
        """Adaptive hybrid algorithm that dynamically weights entropy and frequency based on game stage.

        Uses historical performance data to determine optimal weighting at each step:
        - Early game: Typically favors entropy for maximum elimination
        - Mid game: Balanced weighting based on remaining words
        - Late game: May favor frequency for precision or direct guessing
        """
        if not self.possible_words:
            print("    No possible words left, using random from full list")
            return random.choice(self.word_list)

        num_possible = len(self.possible_words)
        attempt = len(self.guesses)

        # First guess: use proven optimal starter
        if attempt == 0:
            # Use a different starter than entropy to avoid duplicates
            starter = get_universal_optimal_starter("entropy", "balanced")  # "adieu" instead of "tares"
            return starter

        # Very small word lists: direct guessing
        if num_possible <= 2:
            print(f"    Adaptive hybrid: {num_possible} words left, guessing directly")
            return self.possible_words[0]

        # Define step-based weights based on actual performance data from multiple analyses
        # Consolidated evidence shows entropy performs better from step 2 onwards
        step_weights = {
            1: {'entropy': 0.5, 'frequency': 0.5},  # Consistently close, frequency slight edge
            2: {'entropy': 0.7, 'frequency': 0.3},  # Entropy advantage appears consistently
            3: {'entropy': 0.6, 'frequency': 0.4},  # Entropy maintains advantage
            4: {'entropy': 0.6, 'frequency': 0.4},  # Entropy continues to lead
            5: {'entropy': 0.7, 'frequency': 0.3},  # Entropy shows stronger advantage late game
            6: {'entropy': 0.6, 'frequency': 0.4},  # Extrapolated based on entropy trend
        }

        current_step = attempt + 1
        if current_step not in step_weights:
            # Default to balanced for steps beyond 6
            weights = {'entropy': 0.5, 'frequency': 0.5}
        else:
            weights = step_weights[current_step]

        print(f"    Adaptive hybrid: step {current_step}, using entropy weight {weights['entropy']:.1f}, frequency weight {weights['frequency']:.1f}")

        # Calculate scores for both algorithms
        search_space = self.possible_words.copy()

        # Prefer isograms when it makes sense
        if should_prefer_isograms(self.possible_words, len(self.guesses)):
            unique_search_space = self.filter_words_unique_letters(search_space)
            if unique_search_space:
                search_space = unique_search_space
                print(f"    Preferring isograms: filtered to {len(search_space)} words with unique letters")

        word_scores = []

        # Calculate frequency scores
        total_words = len(self.possible_words)
        freq = [Counter() for _ in range(self.word_length)]

        for word in search_space:
            for i, char in enumerate(word):
                freq[i][char] += 1

        # Score each word with weighted combination
        for word in search_space:
            # Frequency score (normalized)
            freq_score = sum(freq[i][word[i]] for i in range(self.word_length))
            freq_score_normalized = freq_score / len(search_space)

            # Entropy score
            pattern_counts = Counter()
            for possible_target in self.possible_words:
                feedback = get_feedback(word, possible_target)
                pattern_counts[feedback] += 1

            entropy = 0
            for count in pattern_counts.values():
                probability = count / total_words
                entropy -= probability * math.log2(probability) if probability > 0 else 0

            # Normalize entropy (approximate max entropy is log2(total_words))
            max_possible_entropy = math.log2(total_words) if total_words > 1 else 1
            entropy_normalized = entropy / max_possible_entropy

            # Weighted combination
            combined_score = (weights['entropy'] * entropy_normalized +
                            weights['frequency'] * freq_score_normalized)

            word_scores.append((word, combined_score, entropy, freq_score))

        # Find the best word
        best_word, best_combined, best_entropy, best_freq = max(word_scores, key=lambda x: x[1])

        print(f"    Best word: '{best_word}' with combined score {best_combined:.3f} (entropy: {best_entropy:.3f}, freq: {best_freq})")
        return best_word

    def choose_guess_smart_hybrid(self) -> str:
        """Smart hybrid approach that adapts strategy based on game state."""
        if not self.possible_words:
            print("    No possible words left, using random from full list")
            return random.choice(self.word_list)

        num_possible = len(self.possible_words)
        attempt = len(self.guesses)

        # First guess: use universal optimal starting word
        if attempt == 0:
            # Use universal starter - not dependent on target word knowledge
            optimal_start = get_universal_optimal_starter("smart_hybrid", "general")  # "crane"
            print(f"    Smart hybrid: using universal optimal start '{optimal_start}'")
            return optimal_start

        # Dynamic threshold adjustment based on attempt number
        # Later in the game, switch to direct guessing sooner
        direct_guess_threshold = max(2, 4 - attempt)  # 2 at attempt 2+, 4 at attempt 0, 3 at attempt 1
        entropy_threshold = max(20, 80 - (attempt * 20))  # Decrease entropy threshold as attempts increase
        info_threshold = max(6, 15 - (attempt * 3))   # Decrease info threshold as attempts increase

        # Final stages: if very few words left, just guess from possibilities
        if num_possible <= direct_guess_threshold:
            print(f"    Smart hybrid: {num_possible} words left, guessing directly")
            return self.possible_words[0]

        # Early-mid game: use entropy for maximum elimination
        if num_possible > entropy_threshold:
            print(f"    Smart hybrid: {num_possible} words remaining, using entropy strategy")
            return self.choose_guess_entropy(False)

        # Mid-late game: use information method for balanced approach
        elif num_possible > info_threshold:
            print(f"    Smart hybrid: {num_possible} words remaining, using information strategy")
            return self.choose_guess_information(False)

        # Late game: use frequency for precision targeting
        else:
            print(f"    Smart hybrid: {num_possible} words remaining, using frequency strategy")
            return self.choose_guess_frequency(start_strategy="fixed")





    def solve_automated(self, target: str, guess_algorithm: str = "random", start_strategy: str = "fixed") -> dict:
        """Solve with detailed tracking for automated analysis.
        Returns dictionary with solving details including word list reduction tracking."""
        self.possible_words = self.word_list.copy()
        self.guesses = []
        self.feedbacks = []

        # Track word list sizes after each guess
        word_list_sizes = [len(self.possible_words)]  # Initial size

        # Select the guessing algorithm
        if guess_algorithm == "random":
            choose_func = lambda: self.choose_guess_random()
        elif guess_algorithm == "entropy":
            choose_func = lambda: self.choose_guess_entropy(False)
        elif guess_algorithm == "frequency":
            choose_func = lambda: self.choose_guess_frequency(start_strategy=start_strategy)
        elif guess_algorithm == "information":
            choose_func = lambda: self.choose_guess_information(False)
        elif guess_algorithm == "smart_hybrid":
            choose_func = lambda: self.choose_guess_smart_hybrid()
        elif guess_algorithm == "ultra_efficient":
            choose_func = lambda: self.choose_guess_ultra_efficient()
        elif guess_algorithm == "adaptive_hybrid":
            choose_func = lambda: self.choose_guess_adaptive_hybrid()
        else:
            raise ValueError("Invalid guess_algorithm. Use 'random', 'entropy', 'frequency', 'information', 'smart_hybrid', 'ultra_efficient', or 'adaptive_hybrid'.")

        solved = False
        attempts = 0

        for attempt in range(self.max_guesses):
            guess = choose_func()
            if not guess:
                break

            feedback = get_feedback(guess, target)
            self.guesses.append(guess)
            self.feedbacks.append(feedback)
            attempts = attempt + 1

            if feedback == 'G' * self.word_length:
                solved = True
                break

            self.filter_words(guess, feedback)
            word_list_sizes.append(len(self.possible_words))

            # Log solver state if we just completed the 6th guess without solving
            if attempt + 1 == 6 and feedback != 'G' * self.word_length:
                write_solver_state_after_6(target, guess_algorithm, start_strategy, self.possible_words)

        return {
            'solved': solved,
            'attempts': attempts,
            'word_list_sizes': word_list_sizes,
            'guesses': self.guesses.copy(),
            'feedbacks': self.feedbacks.copy(),
            'algorithm': guess_algorithm,
            'strategy': start_strategy
        }

    def solve(self, target: str, guess_algorithm: str = "random", start_strategy: str = "fixed") -> Tuple[bool, int]:
        """Attempt to solve Wordle for the given target word using specified guess algorithm.
        guess_algorithm: 'random', 'entropy', 'frequency', or 'information'.
        start_strategy: For frequency algorithm: 'fixed', 'random', 'highest', 'lowest'.
        Returns (solved, number_of_guesses)."""
        self.possible_words = self.word_list.copy()
        self.guesses = []
        self.feedbacks = []

        # Select the guessing algorithm
        if guess_algorithm == "random":
            choose_func = lambda: self.choose_guess_random()
        elif guess_algorithm == "entropy":
            choose_func = lambda: self.choose_guess_entropy(False)  # Never use optimal start for entropy
        elif guess_algorithm == "frequency":
            choose_func = lambda: self.choose_guess_frequency(start_strategy=start_strategy)
        elif guess_algorithm == "information":
            choose_func = lambda: self.choose_guess_information(False)
        elif guess_algorithm == "adaptive_hybrid":
            choose_func = lambda: self.choose_guess_adaptive_hybrid()
        else:
            raise ValueError("Invalid guess_algorithm. Use 'random', 'entropy', 'frequency', 'information', or 'adaptive_hybrid'.")

        for attempt in range(self.max_guesses):
            guess = choose_func()
            if not guess:
                return False, attempt

            feedback = get_feedback(guess, target)
            self.guesses.append(guess)
            self.feedbacks.append(feedback)

            algorithm_desc = f"{guess_algorithm}"
            if attempt == 0 and start_strategy != "fixed":
                algorithm_desc += f" ({start_strategy} start)"
            print(f"Guess {attempt + 1}: {guess} -> {feedback} (Algorithm: {algorithm_desc})")

            if feedback == 'G' * self.word_length:
                return True, attempt + 1

            self.filter_words(guess, feedback)

            # Log solver state if we just completed the 6th guess without solving
            if attempt + 1 == 6 and feedback != 'G' * self.word_length:
                write_solver_state_after_6(target, guess_algorithm, start_strategy, self.possible_words)

        return False, self.max_guesses

def calculate_word_scores(word: str, possible_words: List[str], search_space: List[str]) -> dict:
    """Calculate entropy, frequency, and information scores for a word for display purposes."""
    if not possible_words:
        return {'entropy': 0.0, 'frequency': 0, 'likelihood': 0.0, 'information': 0.0}

    # Calculate entropy score
    pattern_counts = Counter()
    for possible_target in possible_words:
        feedback = get_feedback(word, possible_target)
        pattern_counts[feedback] += 1

    total_words = len(possible_words)
    entropy = 0
    for count in pattern_counts.values():
        probability = count / total_words
        entropy -= probability * math.log2(probability) if probability > 0 else 0

    # Calculate frequency score
    freq = [Counter() for _ in range(5)]  # Assuming 5-letter words
    for search_word in search_space:
        for i, char in enumerate(search_word):
            freq[i][char] += 1

    freq_score = sum(freq[i][word[i]] for i in range(len(word)))
    likelihood_score = freq_score / len(search_space) if search_space else 0

    # Calculate information score
    info_score = get_word_information_score(word, possible_words)

    # Calculate real-world frequency score
    wordfreq_score = get_word_frequency_score(word)

    return {
        'entropy': entropy,
        'frequency': freq_score,
        'likelihood': likelihood_score,
        'information': info_score,
        'wordfreq': wordfreq_score
    }

def get_valid_popular_words(possible_words: List[str], guessed_words: List[str]) -> List[str]:
    """Get popular words that are still valid and haven't been guessed yet."""
    valid_popular = []

    # Convert all to lowercase for comparison
    guessed_words_lower = [word.lower() for word in guessed_words]

    for word in POPULAR_WORDS:
        word_lower = word.lower()

        # Skip if already guessed or suggested
        if word_lower in guessed_words_lower:
            continue

        # Check if word is still possible based on current game state
        if word in possible_words:
            valid_popular.append(word)

    return valid_popular

def interactive_mode():
    """Interactive mode where user sees suggestions from all algorithms and chooses."""
    print("\n" + "="*60)
    print("🎯 INTERACTIVE WORDLE SOLVER - MULTI-ALGORITHM MODE")
    print("="*60)
    print("🤖 You'll see suggestions from all algorithms at each step.")
    print("📊 Each suggestion includes entropy and frequency scores.")
    print("🎮 Choose the word you want to play, or enter your own.")
    print("="*60)

    # Load word list
    word_file_path = os.path.join(DATA_DIR, "words_alpha5.txt")
    word_list = load_words(word_file_path)
    if not word_list:
        print("Word file not found, using fallback list")
        word_list = ["crane", "house", "smile", "grape", "stone", "flame", "lakes"]
        word_file_path = None
    else:
        print(f"Loaded {len(word_list)} words from file")

    # Initialize multiple solvers (one for each algorithm)
    algorithms = {
        'entropy': 'Entropy',
        'frequency': 'Letter-Frequency',
        'information': 'Information',
        'ultra_efficient': 'Ultra-Efficient',
        'adaptive_hybrid': 'Adaptive-Hybrid'
    }

    solvers = {}
    for alg_key in algorithms.keys():
        solvers[alg_key] = WordleSolver(word_list, word_file_path=word_file_path)

    # Choose target word mode
    print("\nChoose your game mode:")
    print("1. TEST     - Specify the target word (for testing)")
    print("2. PRACTICE - Pick a random word for me to solve")
    print("3. WORDLE   - Play against real Wordle website (get AI suggestions)")

    while True:
        try:
            target_choice = input("\nEnter choice (1-3) [default: 3]: ").strip()

            # Make option 3 (real Wordle) the default if no input is provided
            if not target_choice:
                target_choice = "3"

            if target_choice in ["1", "2", "3"]:
                break
            else:
                print("Invalid choice. Please enter 1-3.")
        except KeyboardInterrupt:
            # Exit gracefully to "Thanks for playing!" message
            return

    # Track if user interrupted with Ctrl+C to skip "play again" prompt
    interrupted = False

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

        # Multi-algorithm interactive solving mode
        print(f"\n🎯 Target word: {target.upper()}")
        print("🤖 AI algorithms will suggest words at each step...\n")

        try:
            play_multi_algorithm_game(solvers, algorithms, target, mode="automated")
        except KeyboardInterrupt:
            # Exit immediately to "Thanks for playing!" message
            print("Thanks for playing! 👋")
            interrupted = True

    elif target_choice == "2":
        # Random target word
        target = random.choice(word_list)
        print(f"\n🎯 Random target word selected.")
        print("🤖 AI algorithms will suggest words at each step...\n")

        try:
            result = play_multi_algorithm_game(solvers, algorithms, target, mode="automated")
            print(f"\n🎯 The target word was: {target.upper()}")
        except KeyboardInterrupt:
            # Exit immediately to "Thanks for playing!" message
            print("Thanks for playing! 👋")
            interrupted = True

    else:
        # Manual feedback mode (real Wordle)
        print(f"\n🎮 REAL WORDLE MODE - MULTI-ALGORITHM ASSISTANT")
        print("="*60)
        print("🌐 Play on the real Wordle website and get suggestions from all algorithms.")
        print("📝 Feedback format: G=Green (correct), Y=Yellow (wrong position), X=Gray (not in word)")
        print("❌ Special: R=Rejected (if Wordle doesn't accept the word)")
        print("📘 Example: CRANE -> XYGXX means C=gray, R=yellow, A=green, N=gray, E=gray")
        print("🔄 Rejected words don't count toward your 6-guess limit.")
        print("="*60)

        try:
            play_multi_algorithm_game(solvers, algorithms, None, mode="manual")
        except KeyboardInterrupt:
            # Exit immediately to "Thanks for playing!" message
            print("Thanks for playing! 👋")
            interrupted = True

    # Skip "play again" prompt if user interrupted with Ctrl+C
    if not interrupted:
        # Ask if user wants to play again
        print(f"\nWould you like to play again? (y/n): ", end="")
        try:
            if input().strip().lower().startswith('y'):
                interactive_mode()
        except KeyboardInterrupt:
            pass

    print("Thanks for playing! 👋")

def play_multi_algorithm_game(solvers: dict, algorithms: dict, target: str = None, mode: str = "automated"):
    """Play a game showing suggestions from all algorithms."""

    attempt = 0
    max_attempts = 6  # Standard Wordle limit

    print(f"\n{'🎮 GAME START' if mode == 'manual' else '🤖 SOLVING'}")
    print("="*60)

    while attempt < max_attempts:
        attempt += 1
        print(f"\n📍 GUESS {attempt}")
        print("-" * 40)

        # Get suggestions from all algorithms
        suggestions = {}
        word_to_algorithms = {}  # Track which algorithms suggest each word

        for alg_key, alg_name in algorithms.items():
            solver = solvers[alg_key]

            if not solver.possible_words:
                suggestion = "No words left!"
                scores = {'entropy': 0, 'frequency': 0, 'likelihood': 0, 'information': 0}
            else:
                # Get suggestion based on algorithm
                if alg_key == 'entropy':
                    suggestion = solver.choose_guess_entropy(False)
                elif alg_key == 'frequency':
                    suggestion = solver.choose_guess_frequency(start_strategy="fixed")
                elif alg_key == 'information':
                    suggestion = solver.choose_guess_information(False)
                elif alg_key == 'ultra_efficient':
                    suggestion = solver.choose_guess_ultra_efficient()
                elif alg_key == 'adaptive_hybrid':
                    suggestion = solver.choose_guess_adaptive_hybrid()
                else:
                    suggestion = solver.choose_guess_random()

                # Calculate scores for display
                search_space = solver.possible_words.copy()
                if should_prefer_isograms(solver.possible_words, len(solver.guesses)):
                    isogram_space = solver.filter_words_unique_letters(search_space)
                    if isogram_space:
                        search_space = isogram_space

                scores = calculate_word_scores(suggestion, solver.possible_words, search_space)

            suggestions[alg_key] = {
                'word': suggestion,
                'name': alg_name,
                'scores': scores,
                'remaining': len(solver.possible_words)
            }

            # Track word-to-algorithm mapping for duplicate detection
            if suggestion != "No words left!":
                word_lower = suggestion.lower()
                if word_lower not in word_to_algorithms:
                    word_to_algorithms[word_lower] = []
                word_to_algorithms[word_lower].append((alg_key, alg_name))

        # Handle duplicates by finding alternative suggestions for conflicted algorithms
        unique_suggestions = {}
        used_words = set()

        for alg_key, data in suggestions.items():
            word = data['word']
            word_lower = word.lower()

            if word == "No words left!":
                unique_suggestions[alg_key] = data
            elif word_lower not in used_words:
                # First algorithm to suggest this word - keep it with original algorithm name
                unique_suggestions[alg_key] = data
                used_words.add(word_lower)
            else:
                # This word is already used - find the next best suggestion for this algorithm
                solver = solvers[alg_key]
                alternative_found = False

                if solver.possible_words:
                    # Simple approach: find available words not already used
                    available_words = [w for w in solver.possible_words if w.lower() not in used_words]
                    if available_words:
                        # Take the first available word as alternative
                        alternative_word = available_words[0]
                        search_space = solver.possible_words.copy()
                        if should_prefer_isograms(solver.possible_words, len(solver.guesses)):
                            isogram_space = solver.filter_words_unique_letters(search_space)
                            if isogram_space:
                                search_space = isogram_space
                        alt_scores = calculate_word_scores(alternative_word, solver.possible_words, search_space)
                        unique_suggestions[alg_key] = {
                            'word': alternative_word,
                            'name': data['name'],
                            'scores': alt_scores,
                            'remaining': len(solver.possible_words)
                        }
                        used_words.add(alternative_word.lower())
                        alternative_found = True

                if not alternative_found:
                    # Fallback: mark as no unique suggestion available
                    unique_suggestions[alg_key] = {
                        'word': f"(duplicate)",
                        'name': data['name'],
                        'scores': {'entropy': 0, 'frequency': 0, 'likelihood': 0, 'information': 0},
                        'remaining': len(solver.possible_words) if solver.possible_words else 0
                    }

        # Get valid popular words (excluding already suggested algorithm words and previously guessed words)
        solver_ref = list(solvers.values())[0]  # Use first solver as reference for game state
        guessed_words = solver_ref.guesses if hasattr(solver_ref, 'guesses') else []
        algorithm_words = [data['word'].lower() for data in unique_suggestions.values() if data['word'] not in ["No words left!", "(duplicate)"]]

        valid_popular = get_valid_popular_words(solver_ref.possible_words, guessed_words + algorithm_words)

        # Consolidate all suggestions into one list
        all_suggestions = []

        # Add algorithm suggestions (excluding "No words left!" and "(duplicate)" entries)
        for alg_key, data in unique_suggestions.items():
            # Skip entries that have no valid suggestions
            if data['word'] not in ["No words left!", "(duplicate)"]:
                all_suggestions.append({
                    'word': data['word'],
                    'type': data['name'],
                    'scores': data['scores'],
                    'remaining': data['remaining'],
                    'source': 'algorithm'
                })

        # Add popular words with their scores
        if valid_popular:
            for word in valid_popular:
                # Calculate scores for this popular word
                search_space = solver_ref.possible_words.copy()
                if should_prefer_isograms(solver_ref.possible_words, len(solver_ref.guesses)):
                    isogram_space = solver_ref.filter_words_unique_letters(search_space)
                    if isogram_space:
                        search_space = isogram_space

                scores = calculate_word_scores(word, solver_ref.possible_words, search_space)

                all_suggestions.append({
                    'word': word,
                    'type': 'Popular',
                    'scores': scores,
                    'remaining': len(solver_ref.possible_words),
                    'source': 'popular'
                })

        # Sort all suggestions by composite score (wordfreq + letter frequency, if available)
        def get_sort_key(suggestion):
            scores = suggestion['scores']
            # Use wordfreq as primary, letter frequency as secondary if available
            wordfreq_score = scores.get('wordfreq', 0)
            lettfreq_score = scores.get('frequency', 0) / 1000.0 if scores.get('frequency', 0) > 0 else 0
            # Weighted combination: 70% real-world frequency, 30% letter frequency
            return wordfreq_score * 0.7 + lettfreq_score * 0.3

        all_suggestions.sort(key=get_sort_key, reverse=True)

        # Display consolidated and sorted suggestions
        print("🎯 Word Suggestions (sorted by real-world frequency):")
        print(f"     {'Word':<6} {'Type':<18} {'Entropy':<7} {'LtFq':<5} {'Like':<5} {'Info':<5} {'WF':<5}")
        print("-" * 79)

        for i, suggestion in enumerate(all_suggestions, 1):
            word = suggestion['word']
            word_type = suggestion['type']
            scores = suggestion['scores']

            print(f"{i:2d}. {word.upper():<6} {word_type:<18} "
                  f"{scores['entropy']:<7.2f} {scores['frequency']:<5} "
                  f"{scores['likelihood']:<5.2f} {scores.get('information', 0):<5.2f} "
                  f"{scores.get('wordfreq', 0):<5.2f}")

        total_choices = len(all_suggestions)

        # Get user choice
        print(f"\n📝 Options:")
        print(f"• Enter 1-{total_choices} to select a suggestion")
        print("• Enter a 5-letter word to use your own guess")

        while True:
            try:
                user_input = input(f"\n{Colors.YELLOW}Your choice:{Colors.RESET} ").strip().lower()

                # Check if it's a number (selection)
                if user_input.isdigit() and 1 <= int(user_input) <= total_choices:
                    choice_num = int(user_input)

                    # Select from consolidated list
                    suggestion = all_suggestions[choice_num - 1]
                    chosen_word = suggestion['word']
                    chosen_alg = suggestion['type']

                    if chosen_word in ["No words left!", "(duplicate)"]:
                        print("❌ That suggestion has no valid words available!")
                        continue

                    print(f"✅ Using {chosen_alg} suggestion: {chosen_word.upper()}")
                    break

                # Check if it's a valid 5-letter word
                elif len(user_input) == 5 and user_input.isalpha():
                    chosen_word = user_input
                    chosen_alg = "User Choice"
                    print(f"✅ Using your word: {chosen_word.upper()}")
                    break

                else:
                    print(f"❌ Please enter 1-{total_choices} or a 5-letter word.")

            except KeyboardInterrupt:
                # Show solution if we're in automated mode with a target
                if mode == "automated" and target:
                    print(f"\n🎯 The target word was: {target.upper()}")
                # Re-raise to be caught by interactive_mode for immediate exit
                raise

        # Handle the guess based on mode
        if mode == "automated" and target:
            # Automated mode - calculate feedback
            feedback = get_feedback(chosen_word, target)
            print(f"\n🎯 {chosen_word.upper()} → {feedback}")

            # Check if solved
            if feedback == 'G' * 5:
                print(f"\n🎉 Solved in {attempt} guesses using {chosen_alg}!")
                return {'solved': True, 'attempts': attempt, 'algorithm': chosen_alg}

        else:
            # Manual mode - get feedback from user
            while True:
                try:
                    feedback_input = input(f"\n🌐 Enter Wordle feedback for '{chosen_word.upper()}' (5 chars: G/Y/X, or 'R' if rejected): ").strip().upper()

                    if feedback_input == 'R':
                        print(f"❌ Word '{chosen_word.upper()}' was rejected by Wordle")
                        # Remove word from all solvers using shared approach
                        first_solver = list(solvers.values())[0]
                        print(f"    Removing '{chosen_word}' from word lists (rejected by Wordle)")
                        updated_word_list = remove_word_from_list(first_solver.word_list, chosen_word)
                        updated_possible_words = remove_word_from_list(first_solver.possible_words, chosen_word)
                        print(f"    Word lists updated: {len(updated_word_list)} total words, {len(updated_possible_words)} possible words")

                        # Apply the same updated lists to all solvers
                        for solver in solvers.values():
                            solver.word_list = updated_word_list.copy()
                            solver.possible_words = updated_possible_words.copy()

                        # Save updated word list to file if we have a file path
                        if first_solver.word_file_path:
                            if save_words_to_file(first_solver.word_file_path, updated_word_list):
                                print(f"    Updated word list saved to {first_solver.word_file_path}")
                            else:
                                print(f"    Warning: Could not save updated word list to {first_solver.word_file_path}")

                        print("🔄 Getting new suggestions...")
                        attempt -= 1  # Don't count rejected words
                        break

                    elif len(feedback_input) == 5 and all(c in 'GYX' for c in feedback_input):
                        feedback = feedback_input
                        print(f"📝 Feedback recorded: {chosen_word.upper()} → {feedback}")

                        # Check if solved
                        if feedback == "GGGGG":
                            print(f"\n🎉 Congratulations! You solved it in {attempt} guesses!")
                            print(f"🏆 Winning algorithm: {chosen_alg}")
                            return {'solved': True, 'attempts': attempt, 'algorithm': chosen_alg}
                        break

                    else:
                        print("❌ Please enter exactly 5 characters (G/Y/X) or 'R' for rejected.")

                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    return {'solved': False, 'attempts': attempt, 'algorithm': 'Interrupted'}

        # If word was rejected, continue to next iteration
        if mode == "manual" and feedback_input == 'R':
            continue

        # Update all solvers with the guess and feedback
        # Filter the word list once and apply to all solvers for efficiency
        first_solver = list(solvers.values())[0]
        old_count = len(first_solver.possible_words)

        # Perform filtering once
        filtered_words = [
            word for word in first_solver.possible_words
            if is_valid_word(word, chosen_word, feedback)
        ]
        new_count = len(filtered_words)
        print(f"    Filtered from {old_count} to {new_count} possible words")

        # Apply the same filtered list and game state to all solvers
        for solver in solvers.values():
            solver.guesses.append(chosen_word)
            solver.feedbacks.append(feedback)
            solver.possible_words = filtered_words.copy()  # Share the same filtered result

        # Show remaining words count
        remaining_counts = {alg: len(solver.possible_words) for alg, solver in solvers.items()}
        print(f"\n📊 Words remaining: {remaining_counts}")

        # Check if any solver has no words left
        if any(count == 0 for count in remaining_counts.values()):
            print("⚠️  Some algorithms have no possible words left! Check your feedback.")

    # Game over
    print(f"\n😞 Game over! Used all {max_attempts} guesses.")
    return {'solved': False, 'attempts': max_attempts, 'algorithm': 'None'}

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
                automated_testing()
                return
            else:
                print("Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            return

def automated_testing():
    """Run automated testing with user-selected word file."""
    print("\n🤖 AUTOMATED TESTING MODE")
    print("="*60)

    # Load word list from file
    word_list = load_words(os.path.join(DATA_DIR, "words_alpha5.txt"))
    if not word_list:
        print("Word file not found, using fallback list")
        word_list = ["crane", "house", "smile", "grape", "stone", "flame", "lakes"]
    else:
        print(f"Loaded {len(word_list)} words from file")

    # Prompt user for which test word file to use
    print("\nSelect which word file to test against:")
    print("1. All past Wordle words (words_past5_date.txt)")
    print("2. Failed/missing words (words_missing.txt)")
    print("3. Challenging words (words_challenging.txt)")

    test_file_options = {
        "1": ("words_past5_date.txt", "all past Wordle words"),
        "2": ("words_missing.txt", "failed/missing words"),
        "3": ("words_challenging.txt", "challenging words")
    }

    while True:
        try:
            file_choice = input("\nEnter your choice (1-3): ").strip()
            if file_choice in test_file_options:
                test_filename, description = test_file_options[file_choice]
                break
            else:
                print("Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            return

    # Load the selected test word file
    test_words_file = os.path.join(DATA_DIR, test_filename)
    test_words_from_file = load_words(test_words_file)

    if test_words_from_file:
        print(f"Found {description} file: {len(test_words_from_file)} words")

        # Prompt user for how many words to test
        while True:
            try:
                user_input = input(f"How many words to test? (number, 'all', or Enter for default 10): ").strip().lower()

                if not user_input:  # Empty input (just pressed Enter)
                    num_words = min(10, len(test_words_from_file))
                    break
                elif user_input == "all":
                    num_words = len(test_words_from_file)
                    break
                else:
                    num_words = int(user_input)
                    if num_words <= 0:
                        print("Please enter a positive number.")
                        continue
                    elif num_words > len(test_words_from_file):
                        print(f"Only {len(test_words_from_file)} words available. Using all {len(test_words_from_file)} words.")
                        num_words = len(test_words_from_file)
                    break
            except ValueError:
                print("Please enter a valid number, 'all', or press Enter for default.")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                return

        test_words = test_words_from_file[:num_words]
        print(f"Using {len(test_words)} {description}: {[w.upper() for w in test_words]}")
    else:
        print(f"{description.capitalize()} file not found, using default test words")
        test_words = ["smile", "house", "grape"]  # Using fewer words for manageable output
        print(f"Using default test words: {[w.upper() for w in test_words]}")

    algorithms = ["entropy", "frequency", "information", "ultra_efficient", "adaptive_hybrid"]

    # Track results for summary
    results = {}

    for target in test_words:
        print(f"\n{'='*80}")
        print(f"Testing target word: {target.upper()}")
        print(f"{'='*80}")

        for algorithm in algorithms:
            print(f"\n{'-'*60}")
            print(f"Algorithm: {algorithm.upper()}")
            print(f"{'-'*60}")

            if algorithm == "entropy":
                # Entropy algorithm: only test fixed start
                print(f"\nTesting {algorithm} algorithm (fixed start):")
                solver = WordleSolver(word_list)
                result = solver.solve_automated(target, guess_algorithm=algorithm, start_strategy="fixed")
                print(format_result(result['solved'], result['attempts']))

                # Log failed words
                if not result['solved']:
                    write_failed_word(target, algorithm, "fixed")
                # Log challenging words (solved but > 6 guesses)
                elif result['solved'] and result['attempts'] > 6:
                    write_challenging_word(target)

                # Track results
                key = f"{algorithm} (fixed)"
                if key not in results:
                    results[key] = []
                results[key].append(result)

            elif algorithm == "frequency":
                # Frequency algorithm: only test fixed start
                print(f"\nTesting {algorithm} algorithm (fixed start):")
                solver = WordleSolver(word_list)
                result = solver.solve_automated(target, guess_algorithm=algorithm, start_strategy="fixed")
                print(format_result(result['solved'], result['attempts']))

                # Log failed words
                if not result['solved']:
                    write_failed_word(target, algorithm, "fixed")
                # Log challenging words (solved but > 6 guesses)
                elif result['solved'] and result['attempts'] > 6:
                    write_challenging_word(target)

                # Track results
                key = f"{algorithm} (fixed)"
                if key not in results:
                    results[key] = []
                results[key].append(result)

            elif algorithm == "ultra_efficient":
                # Ultra efficient algorithm: speed optimized
                print(f"\nTesting {algorithm} algorithm (speed optimized):")
                solver = WordleSolver(word_list)
                result = solver.solve_automated(target, guess_algorithm=algorithm, start_strategy="fixed")
                print(format_result(result['solved'], result['attempts']))

                # Log failed words
                if not result['solved']:
                    write_failed_word(target, algorithm, "speed")
                # Log challenging words (solved but > 6 guesses)
                elif result['solved'] and result['attempts'] > 6:
                    write_challenging_word(target)

                # Track results
                key = f"{algorithm} (speed)"
                if key not in results:
                    results[key] = []
                results[key].append(result)

            elif algorithm == "adaptive_hybrid":
                # Adaptive hybrid algorithm: dynamic entropy/frequency weighting
                print(f"\nTesting {algorithm} algorithm (dynamic weighting):")
                solver = WordleSolver(word_list)
                result = solver.solve_automated(target, guess_algorithm=algorithm, start_strategy="fixed")
                print(format_result(result['solved'], result['attempts']))

                # Log failed words
                if not result['solved']:
                    write_failed_word(target, algorithm, "adaptive")
                # Log challenging words (solved but > 6 guesses)
                elif result['solved'] and result['attempts'] > 6:
                    write_challenging_word(target)

                # Track results
                key = f"{algorithm} (adaptive)"
                if key not in results:
                    results[key] = []
                results[key].append(result)

    # Print summary of results
    print(f"\n{'='*80}")
    print("📊 PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Tested {len(test_words)} words: {[w.upper() for w in test_words]}")
    print()

    # Analyze word reduction effectiveness
    reduction_analysis = analyze_word_reduction(results)

    # Calculate and display statistics for each algorithm
    for algorithm_key in sorted(results.keys()):
        algorithm_results = results[algorithm_key]
        total_tests = len(algorithm_results)
        successful_solves = sum(1 for result in algorithm_results if result['solved'])
        failed_solves = total_tests - successful_solves

        # Count wins (solved within 6 guesses) and losses (exceeded 6 guesses or failed)
        wins = sum(1 for result in algorithm_results if result['solved'] and result['attempts'] <= 6)
        exceeded_limit = sum(1 for result in algorithm_results if result['attempts'] > 6)

        if successful_solves > 0:
            successful_attempts = [result['attempts'] for result in algorithm_results if result['solved']]
            avg_attempts = sum(successful_attempts) / len(successful_attempts)
            min_attempts = min(successful_attempts)
            max_attempts = max(successful_attempts)
        else:
            avg_attempts = 0
            min_attempts = 0
            max_attempts = 0

        # Color code the algorithm name based on overall performance
        if wins == total_tests and avg_attempts <= 4:
            color = Colors.GREEN
        elif wins == total_tests and avg_attempts <= 6:
            color = Colors.YELLOW
        else:
            color = Colors.RED

        print(f"{color}{algorithm_key:20}{Colors.RESET} | ", end="")
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

    # Add word reduction analysis
    if reduction_analysis:
        print(f"\n{'='*80}")
        print("🔄 WORD LIST REDUCTION ANALYSIS")
        print(f"{'='*80}")
        print("Shows how effectively each method reduces the possible word list")
        print()

        # Sort methods by average reduction effectiveness
        sorted_methods = sorted(reduction_analysis.items(),
                              key=lambda x: x[1]['avg_reduction_per_step'],
                              reverse=True)

        for method_key, analysis in sorted_methods:
            avg_reduction = analysis['avg_reduction_per_step']
            max_reduction = analysis['max_reduction_seen']

            # Color code based on reduction effectiveness
            if avg_reduction >= 75:
                color = Colors.GREEN
            elif avg_reduction >= 50:
                color = Colors.YELLOW
            else:
                color = Colors.RED

            print(f"{color}{method_key:20}{Colors.RESET} | ", end="")
            print(f"Avg reduction: {avg_reduction:5.1f}% | ", end="")
            print(f"Max reduction: {max_reduction:5.1f}% | ", end="")

            # Show step-by-step breakdown for first few steps
            step_info = []
            for step in sorted(analysis['avg_by_step'].keys())[:3]:  # Show first 3 steps
                step_avg = analysis['avg_by_step'][step]
                step_info.append(f"Step {step}: {step_avg:.1f}%")

            print(f"({', '.join(step_info)})")

        # Add step-by-step analysis for entropy vs frequency
        step_analysis = analyze_algorithm_performance_by_step(results)
        if step_analysis:
            print(f"\n{'='*60}")
            print("📈 STEP-BY-STEP PERFORMANCE ANALYSIS")
            print(f"{'='*60}")
            print("Entropy vs Frequency algorithm effectiveness by game step")
            print("(Helps determine optimal weighting for adaptive hybrid algorithms)")
            print()

            print(f"{'Step':<4} | {'Entropy':<8} | {'Frequency':<9} | {'Better':<9} | {'Weights'}")
            print("-" * 55)

            for step in sorted(step_analysis.keys()):
                data = step_analysis[step]
                entropy_avg = data['entropy_avg_reduction']
                freq_avg = data['frequency_avg_reduction']
                better = data['better_algorithm']
                entropy_weight = data['entropy_weight']
                freq_weight = data['frequency_weight']

                # Color code the better algorithm
                if better == "entropy":
                    better_display = f"{Colors.GREEN}entropy{Colors.RESET}"
                else:
                    better_display = f"{Colors.YELLOW}frequency{Colors.RESET}"

                print(f"{step:<4} | {entropy_avg:6.1f}%  | {freq_avg:7.1f}%  | {better_display:<17} | E:{entropy_weight:.1f} F:{freq_weight:.1f}")

            print(f"\n💡 Adaptive Hybrid Insights:")

            # Analyze patterns
            early_entropy_count = sum(1 for step in [1, 2] if step in step_analysis and step_analysis[step]['better_algorithm'] == 'entropy')
            late_freq_count = sum(1 for step in [4, 5, 6] if step in step_analysis and step_analysis[step]['better_algorithm'] == 'frequency')

            if early_entropy_count >= 1:
                print(f"   • Entropy typically better early game")
            if late_freq_count >= 2:
                print(f"   • Frequency typically better late game")

            # Find crossover
            crossover = next((step for step in sorted(step_analysis.keys()) if step_analysis[step]['better_algorithm'] == 'frequency'), None)
            if crossover:
                print(f"   • Strategy transition recommended at step {crossover}")

        print(f"\n📈 {Colors.GREEN}Best decimator{Colors.RESET}: ", end="")
        best_method, best_analysis = sorted_methods[0]
        print(f"{best_method} (avg {best_analysis['avg_reduction_per_step']:.1f}% reduction per step)")

        print(f"\n💡 Analysis shows which method reduces possibilities fastest at each step.")
        print(f"   Higher percentages = better at eliminating impossible words quickly.")

if __name__ == "__main__":
    main()