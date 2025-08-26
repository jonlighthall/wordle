import random
import math
import os
import datetime
from collections import Counter
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
from ..core.wordle_utils import get_feedback, calculate_entropy, has_unique_letters, is_valid_word, load_words, filter_words_unique_letters, filter_wordle_appropriate, should_prefer_isograms, remove_word_from_list, save_words_to_file, get_word_information_score

# Get the repository root directory (3 levels up from this file)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
LOGS_DIR = os.path.join(REPO_ROOT, 'logs')

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

def calculate_information_for_word(args):
    """Helper function for parallel information score calculation.

    Args:
        args: Tuple of (guess_word, possible_words_list)

    Returns:
        Tuple of (guess_word, combined_score, info_score, entropy_score)
    """
    word, possible_words = args

    # Calculate both metrics independently
    info_score = get_word_information_score(word, possible_words)
    entropy_score = calculate_entropy(word, possible_words)

    # Use the appropriate metric based on game state (not double counting)
    num_words = len(possible_words)

    if num_words > 50:
        # Early game: Entropy is superior for elimination
        combined_score = entropy_score
        method_used = "entropy"
    elif num_words > 10:
        # Mid game: Choose based on which score is higher
        # This lets each metric compete fairly without double counting
        if entropy_score > info_score:
            combined_score = entropy_score
            method_used = "entropy"
        else:
            combined_score = info_score
            method_used = "information"
    else:
        # End game: Information score better for precision
        combined_score = info_score
        method_used = "information"

    return (word, combined_score, info_score, entropy_score)

def detect_word_patterns(possible_words: List[str]) -> dict:
    """Detect common patterns in remaining words to optimize strategy."""
    if not possible_words:
        return {}

    patterns = {
        'common_prefixes': Counter(),
        'common_suffixes': Counter(),
        'repeated_letters': 0,
        'vowel_positions': Counter(),
        'consonant_clusters': Counter()
    }

    for word in possible_words:
        # Prefixes and suffixes
        if len(word) >= 2:
            patterns['common_prefixes'][word[:2]] += 1
            patterns['common_suffixes'][word[-2:]] += 1

        # Repeated letters
        if len(set(word)) < len(word):
            patterns['repeated_letters'] += 1

        # Vowel positions
        for i, char in enumerate(word):
            if char in 'aeiou':
                patterns['vowel_positions'][i] += 1

        # Common consonant clusters
        for i in range(len(word) - 1):
            if word[i] not in 'aeiou' and word[i+1] not in 'aeiou':
                patterns['consonant_clusters'][word[i:i+2]] += 1

    return patterns

def get_universal_optimal_starter(method: str = "entropy", strategy: str = "general") -> str:
    """Get universal optimal starting words based on extensive pre-analysis.

    These are computed offline based on performance against large datasets,
    not dependent on the current target word (which is unknown at start).
    """

    # Updated optimal starters based on the latest analysis results
    # These were determined by analyzing reduction percentages and success rates

    optimal_starters = {
        "entropy": {
            "general": "crane",      # Best overall entropy and proven track record
            "aggressive": "slate",   # Maximum elimination for difficult words
            "balanced": "adieu",     # Good balance covering vowels
            "conservative": "tares"  # Current best performing starter
        },
        "frequency": {
            "general": "roate",      # Best frequency-based performance
            "common": "stare",       # Focus on most common letters
            "balanced": "arose",     # Balance of frequency and position
            "vowel_focus": "adieu"   # When expecting vowel-heavy targets
        },
        "information": {
            "general": "slate",      # Best for hybrid information approach
            "balanced": "cares",     # Good information + entropy balance
            "precise": "stare"       # High precision for end-game
        },
        "smart_hybrid": {
            "general": "crane",      # Proven best all-around starter
            "challenging": "slate",  # For difficult word sets
            "balanced": "adieu"      # Vowel coverage for diverse targets
        }
    }

    return optimal_starters.get(method, {}).get(strategy, "crane")

def get_adaptive_starter_by_context(context: str = "general") -> str:
    """Get optimal starter based on game context/iteration.

    This allows for varying starters based on different scenarios:
    - Testing different word types
    - Sequential games
    - Known characteristics of target set
    """

    context_starters = {
        "general": "tares",        # Best overall performance
        "challenging": "roate",    # For known difficult words
        "common_words": "arose",   # For everyday vocabulary
        "past_wordles": "slate",   # Optimized for historical Wordle words
        "missing_words": "lynch",  # For words that failed previous solvers
        "vowel_heavy": "adieu",    # When targets likely have many vowels
        "consonant_heavy": "lynch" # When targets likely consonant-heavy
    }

    return context_starters.get(context, "tares")

def analyze_word_reduction(results_by_method: dict) -> dict:
    """Analyze word list reduction effectiveness for each method."""
    analysis = {}

    for method_key, method_results in results_by_method.items():
        if not method_results:
            continue

        # Collect reduction statistics
        all_reductions = []
        step_reductions = {}  # Track reduction at each step number

        for result in method_results:
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

            analysis[method_key] = {
                'avg_reduction_per_step': avg_reduction,
                'max_reduction_seen': max_reduction,
                'avg_by_step': avg_by_step,
                'total_reductions': len(all_reductions)
            }

    return analysis

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
            return "roate"  # Better first guess than crane
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
                return "roate"  # Optimal frequency-based starting word (better than cares)
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

    def choose_guess_pattern_aware(self) -> str:
        """Pattern-aware method that uses word structure analysis for optimization."""
        if not self.possible_words:
            print("    No possible words left, using random from full list")
            return random.choice(self.word_list)

        num_possible = len(self.possible_words)
        attempt = len(self.guesses)

        # First guess: use proven optimal starter
        if attempt == 0:
            starter = get_universal_optimal_starter("smart_hybrid", "general")
            print(f"    Pattern aware: using proven starter '{starter}'")
            return starter

        # Aggressive early termination
        if num_possible <= 2:
            print(f"    Pattern aware: {num_possible} words left, guessing directly")
            return self.possible_words[0]

        # Analyze remaining word patterns
        patterns = detect_word_patterns(self.possible_words)

        # Very aggressive termination if patterns suggest it
        if attempt >= 3 and num_possible <= 6:
            print(f"    Pattern aware: attempt {attempt+1}, {num_possible} words left, guessing directly")
            return self.possible_words[0]
        elif attempt >= 2 and num_possible <= 3:
            print(f"    Pattern aware: attempt {attempt+1}, {num_possible} words left, guessing directly")
            return self.possible_words[0]

        # Pattern-based strategy selection
        repeated_ratio = patterns.get('repeated_letters', 0) / num_possible if num_possible > 0 else 0

        if num_possible > 15:
            # Early game: use entropy for maximum elimination
            print(f"    Pattern aware: {num_possible} words remaining, using entropy strategy")
            return self.choose_guess_entropy(False)
        elif repeated_ratio > 0.5 and num_possible > 6:
            # Many words with repeated letters - use frequency approach
            print(f"    Pattern aware: {num_possible} words with {repeated_ratio:.1%} repeated letters, using frequency strategy")
            return self.choose_guess_frequency(start_strategy="fixed")
        else:
            # Mid-late game: prefer words from possible list for lucky guesses
            print(f"    Pattern aware: {num_possible} words remaining, preferring possible words")
            # Choose best entropy word from possible words only
            best_word = None
            best_entropy = -1

            search_words = self.possible_words[:min(15, len(self.possible_words))]
            for word in search_words:
                entropy = calculate_entropy(word, self.possible_words)
                if entropy > best_entropy:
                    best_entropy = entropy
                    best_word = word

            print(f"    Best possible word: '{best_word}' with entropy {best_entropy:.3f}")
            return best_word

    def choose_guess_ultra_efficient(self) -> str:
        """Ultra-efficient method that prioritizes speed over theoretical optimality."""
        if not self.possible_words:
            print("    No possible words left, using random from full list")
            return random.choice(self.word_list)

        num_possible = len(self.possible_words)
        attempt = len(self.guesses)

        # First guess: use proven optimal starter
        if attempt == 0:
            starter = get_universal_optimal_starter("smart_hybrid", "general")
            print(f"    Ultra efficient: using proven starter '{starter}'")
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
            optimal_start = get_universal_optimal_starter("entropy", "general")
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

    def choose_guess_smart_hybrid_adaptive(self, test_context: str = "general") -> str:
        """Enhanced smart hybrid that can vary starting words based on context."""
        if not self.possible_words:
            print("    No possible words left, using random from full list")
            return random.choice(self.word_list)

        num_possible = len(self.possible_words)
        attempt = len(self.guesses)

        # First guess: use context-aware starting word
        if attempt == 0:
            # Vary starting word based on test context
            optimal_start = get_adaptive_starter_by_context(test_context)
            print(f"    Smart hybrid adaptive: using context '{test_context}' starter '{optimal_start}'")
            return optimal_start

        # Rest of the logic remains the same as smart_hybrid
        direct_guess_threshold = max(2, 4 - attempt)
        entropy_threshold = max(20, 80 - (attempt * 20))
        info_threshold = max(6, 15 - (attempt * 3))

        if num_possible <= direct_guess_threshold:
            print(f"    Smart hybrid adaptive: {num_possible} words left, guessing directly")
            return self.possible_words[0]

        if num_possible > entropy_threshold:
            print(f"    Smart hybrid adaptive: {num_possible} words remaining, using entropy strategy")
            return self.choose_guess_entropy(False)
        elif num_possible > info_threshold:
            print(f"    Smart hybrid adaptive: {num_possible} words remaining, using information strategy")
            return self.choose_guess_information(False)
        else:
            print(f"    Smart hybrid adaptive: {num_possible} words remaining, using frequency strategy")
            return self.choose_guess_frequency(start_strategy="fixed")

    def solve_automated_with_context(self, target: str, guess_method: str = "smart_hybrid_adaptive", test_context: str = "general") -> dict:
        """Solve with context-aware starting words for testing variation."""
        self.possible_words = self.word_list.copy()
        self.guesses = []
        self.feedbacks = []

        # Track word list sizes after each guess
        word_list_sizes = [len(self.possible_words)]

        # Select the guessing method with context awareness
        if guess_method == "smart_hybrid_adaptive":
            choose_func = lambda: self.choose_guess_smart_hybrid_adaptive(test_context)
        else:
            # Fall back to regular methods
            return self.solve_automated(target, guess_method, "fixed")

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
                write_solver_state_after_6(target, guess_method, test_context, self.possible_words)

        return {
            'solved': solved,
            'attempts': attempts,
            'word_list_sizes': word_list_sizes,
            'guesses': self.guesses.copy(),
            'feedbacks': self.feedbacks.copy(),
            'method': guess_method,
            'strategy': test_context
        }

    def solve_automated(self, target: str, guess_method: str = "random", start_strategy: str = "fixed") -> dict:
        """Solve with detailed tracking for automated analysis.
        Returns dictionary with solving details including word list reduction tracking."""
        self.possible_words = self.word_list.copy()
        self.guesses = []
        self.feedbacks = []

        # Track word list sizes after each guess
        word_list_sizes = [len(self.possible_words)]  # Initial size

        # Select the guessing method
        if guess_method == "random":
            choose_func = lambda: self.choose_guess_random()
        elif guess_method == "entropy":
            choose_func = lambda: self.choose_guess_entropy(False)
        elif guess_method == "frequency":
            choose_func = lambda: self.choose_guess_frequency(start_strategy=start_strategy)
        elif guess_method == "information":
            choose_func = lambda: self.choose_guess_information(False)
        elif guess_method == "smart_hybrid":
            choose_func = lambda: self.choose_guess_smart_hybrid()
        elif guess_method == "ultra_efficient":
            choose_func = lambda: self.choose_guess_ultra_efficient()
        else:
            raise ValueError("Invalid guess_method. Use 'random', 'entropy', 'frequency', 'information', 'smart_hybrid', or 'ultra_efficient'.")

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
                write_solver_state_after_6(target, guess_method, start_strategy, self.possible_words)

        return {
            'solved': solved,
            'attempts': attempts,
            'word_list_sizes': word_list_sizes,
            'guesses': self.guesses.copy(),
            'feedbacks': self.feedbacks.copy(),
            'method': guess_method,
            'strategy': start_strategy
        }

    def solve(self, target: str, guess_method: str = "random", start_strategy: str = "fixed") -> Tuple[bool, int]:
        """Attempt to solve Wordle for the given target word using specified guess method.
        guess_method: 'random', 'entropy', 'frequency', or 'information'.
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
        elif guess_method == "information":
            choose_func = lambda: self.choose_guess_information(False)
        else:
            raise ValueError("Invalid guess_method. Use 'random', 'entropy', 'frequency', or 'information'.")

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
    word_file_path = os.path.join(DATA_DIR, "words_alpha5.txt")
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
    print("4. Information-based (hybrid approach)")
    print("5. Smart Hybrid (adaptive strategy)")

    while True:
        try:
            method_choice = input("\nEnter choice (1-5): ").strip()
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
            elif method_choice == "4":
                guess_method = "information"
                break
            elif method_choice == "5":
                guess_method = "smart_hybrid"
                break
            else:
                print("Invalid choice. Please enter 1-5.")
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

    methods = ["entropy", "frequency", "ultra_efficient"]

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

            if method == "entropy":
                # Entropy method: only test fixed start
                print(f"\nTesting {method} method (fixed start):")
                solver = WordleSolver(word_list)
                result = solver.solve_automated(target, guess_method=method, start_strategy="fixed")
                print(format_result(result['solved'], result['attempts']))

                # Log failed words
                if not result['solved']:
                    write_failed_word(target, method, "fixed")
                # Log challenging words (solved but > 6 guesses)
                elif result['solved'] and result['attempts'] > 6:
                    write_challenging_word(target)

                # Track results
                key = f"{method} (fixed)"
                if key not in results:
                    results[key] = []
                results[key].append(result)

            elif method == "frequency":
                # Frequency method: only test fixed start
                print(f"\nTesting {method} method (fixed start):")
                solver = WordleSolver(word_list)
                result = solver.solve_automated(target, guess_method=method, start_strategy="fixed")
                print(format_result(result['solved'], result['attempts']))

                # Log failed words
                if not result['solved']:
                    write_failed_word(target, method, "fixed")
                # Log challenging words (solved but > 6 guesses)
                elif result['solved'] and result['attempts'] > 6:
                    write_challenging_word(target)

                # Track results
                key = f"{method} (fixed)"
                if key not in results:
                    results[key] = []
                results[key].append(result)

            elif method == "ultra_efficient":
                # Ultra efficient method: speed optimized
                print(f"\nTesting {method} method (speed optimized):")
                solver = WordleSolver(word_list)
                result = solver.solve_automated(target, guess_method=method, start_strategy="fixed")
                print(format_result(result['solved'], result['attempts']))

                # Log failed words
                if not result['solved']:
                    write_failed_word(target, method, "speed")
                # Log challenging words (solved but > 6 guesses)
                elif result['solved'] and result['attempts'] > 6:
                    write_challenging_word(target)

                # Track results
                key = f"{method} (speed)"
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

    # Calculate and display statistics for each method
    for method_key in sorted(results.keys()):
        method_results = results[method_key]
        total_tests = len(method_results)
        successful_solves = sum(1 for result in method_results if result['solved'])
        failed_solves = total_tests - successful_solves

        # Count wins (solved within 6 guesses) and losses (exceeded 6 guesses or failed)
        wins = sum(1 for result in method_results if result['solved'] and result['attempts'] <= 6)
        exceeded_limit = sum(1 for result in method_results if result['attempts'] > 6)

        if successful_solves > 0:
            successful_attempts = [result['attempts'] for result in method_results if result['solved']]
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

        print(f"\n📈 {Colors.GREEN}Best decimator{Colors.RESET}: ", end="")
        best_method, best_analysis = sorted_methods[0]
        print(f"{best_method} (avg {best_analysis['avg_reduction_per_step']:.1f}% reduction per step)")

        print(f"\n💡 Analysis shows which method reduces possibilities fastest at each step.")
        print(f"   Higher percentages = better at eliminating impossible words quickly.")

if __name__ == "__main__":
    main()