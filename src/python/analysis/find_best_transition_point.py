#!/usr/bin/env python3
"""
Analysis script to determine the optimal point in the game to prioritize known words
over maximum information gain strategies.

This script tests different transition strategies:
1. Pure entropy (never prioritize known words)
2. Pure wordfreq (always prioritize known words)
3. Hybrid strategies that switch at different game stages
4. Dynamic strategies based on remaining word count

This is a simplified standalone version that implements basic solving logic
to avoid complex import dependencies.
"""

import os
import sys
import math
import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.wordle_utils import load_words, get_feedback, is_valid_word, calculate_entropy
from core.common_utils import DATA_DIR, ProgressReporter, load_word_list_with_fallback, get_word_frequency_score


class SimpleWordleSolver:
    """Simplified Wordle solver for transition analysis."""
    
    def __init__(self, word_list: List[str]):
        self.word_list = word_list
        self.possible_words = word_list.copy()
        self.guesses = []
        self.feedbacks = []
    
    def reset(self):
        """Reset solver state for new game."""
        self.possible_words = self.word_list.copy()
        self.guesses = []
        self.feedbacks = []
    
    def filter_words(self, guess: str, feedback: str):
        """Filter possible words based on guess and feedback."""
        self.possible_words = [
            word for word in self.possible_words
            if is_valid_word(word, guess, feedback)
        ]
    
    def choose_entropy_guess(self) -> str:
        """Choose guess using entropy strategy."""
        if len(self.guesses) == 0:
            return "tares"  # Optimal entropy starter
        
        if not self.possible_words:
            return random.choice(self.word_list)
        
        if len(self.possible_words) <= 2:
            return self.possible_words[0]
        
        # Use entropy calculation for larger sets
        best_word = None
        best_entropy = -1
        
        search_space = self.possible_words[:min(50, len(self.possible_words))]  # Limit for speed
        
        for word in search_space:
            entropy = calculate_entropy(word, self.possible_words)
            if entropy > best_entropy:
                best_entropy = entropy
                best_word = word
        
        return best_word if best_word else self.possible_words[0]
    
    def choose_wordfreq_guess(self) -> str:
        """Choose guess prioritizing real-world word frequency."""
        if len(self.guesses) == 0:
            return "about"  # Common word starter
        
        if not self.possible_words:
            return random.choice(self.word_list)
        
        # Score by word frequency
        scored_words = []
        for word in self.possible_words:
            freq_score = get_word_frequency_score(word)
            scored_words.append((word, freq_score))
        
        # Sort by frequency score (higher = more common)
        scored_words.sort(key=lambda x: x[1], reverse=True)
        return scored_words[0][0] if scored_words else self.possible_words[0]


class TransitionPointAnalyzer:
    """Analyze optimal transition points from entropy to wordfreq strategies."""
    
    def __init__(self, word_list: List[str], test_words: List[str]):
        self.word_list = word_list
        self.test_words = test_words
    
    def test_strategy(self, strategy_name: str, strategy_func: callable, max_test_words: int = 50) -> Dict:
        """Test a specific strategy against test words."""
        results = {
            'strategy': strategy_name,
            'solved': 0,
            'failed': 0,
            'total_attempts': 0,
            'attempt_distribution': defaultdict(int),
            'solve_rate': 0.0,
            'avg_attempts': 0.0
        }
        
        test_subset = self.test_words[:max_test_words]
        progress_reporter = ProgressReporter(len(test_subset), report_interval=10)
        
        for i, target in enumerate(test_subset):
            progress_reporter.report_progress(i, "words")
            
            solver = SimpleWordleSolver(self.word_list)
            solved = False
            attempts = 0
            
            for attempt in range(10):  # Max 10 attempts
                guess = strategy_func(solver, attempt)
                if not guess:
                    break
                
                feedback = get_feedback(guess, target)
                solver.guesses.append(guess)
                solver.feedbacks.append(feedback)
                attempts = attempt + 1
                
                if feedback == 'G' * 5:
                    solved = True
                    break
                
                solver.filter_words(guess, feedback)
            
            # Record results
            if solved:
                results['solved'] += 1
                results['attempt_distribution'][attempts] += 1
            else:
                results['failed'] += 1
                results['attempt_distribution']['failed'] += 1
            
            results['total_attempts'] += attempts
        
        progress_reporter.final_report("words")
        
        # Calculate statistics
        total_tests = len(test_subset)
        results['solve_rate'] = results['solved'] / total_tests
        results['avg_attempts'] = results['total_attempts'] / total_tests if total_tests > 0 else 0
        results['total_tests'] = total_tests
        
        return results
    
    def create_transition_strategy(self, transition_step: int):
        """Create a strategy that switches from entropy to wordfreq at specified step."""
        def strategy_func(solver: SimpleWordleSolver, attempt: int) -> str:
            if attempt < transition_step:
                return solver.choose_entropy_guess()
            else:
                return solver.choose_wordfreq_guess()
        return strategy_func
    
    def create_count_strategy(self, word_threshold: int):
        """Create a strategy that switches based on remaining word count."""
        def strategy_func(solver: SimpleWordleSolver, attempt: int) -> str:
            if len(solver.possible_words) > word_threshold:
                return solver.choose_entropy_guess()
            else:
                return solver.choose_wordfreq_guess()
        return strategy_func
    
    def run_analysis(self) -> Dict:
        """Run comprehensive transition point analysis."""
        print("🧪 KNOWN WORD TRANSITION ANALYSIS")
        print("=" * 60)
        print("Testing when to prioritize known words over entropy...")
        print()
        
        strategies = {}
        
        # Pure strategies
        print("📊 Testing baseline strategies...")
        
        strategies['pure_entropy'] = self.test_strategy(
            "Pure Entropy",
            lambda solver, attempt: solver.choose_entropy_guess()
        )
        
        strategies['pure_wordfreq'] = self.test_strategy(
            "Pure WordFreq", 
            lambda solver, attempt: solver.choose_wordfreq_guess()
        )
        
        # Step-based transitions
        print("\n📊 Testing step-based transitions...")
        for step in [2, 3, 4]:
            strategy_name = f"Switch at Step {step}"
            strategy_func = self.create_transition_strategy(step)
            strategies[f'step_{step}'] = self.test_strategy(strategy_name, strategy_func)
        
        # Count-based transitions
        print("\n📊 Testing count-based transitions...")
        for count in [10, 25, 50]:
            strategy_name = f"Switch at ≤{count} words"
            strategy_func = self.create_count_strategy(count)
            strategies[f'count_{count}'] = self.test_strategy(strategy_name, strategy_func)
        
        return strategies
    
    def display_results(self, strategies: Dict):
        """Display analysis results."""
        print("\n" + "=" * 80)
        print("📈 TRANSITION ANALYSIS RESULTS")
        print("=" * 80)
        
        # Sort by solve rate
        sorted_strategies = sorted(
            strategies.items(), 
            key=lambda x: x[1]['solve_rate'], 
            reverse=True
        )
        
        print(f"{'Strategy':<25} {'Solve Rate':<12} {'Avg Attempts':<15}")
        print("-" * 55)
        
        for strategy_key, results in sorted_strategies:
            strategy_name = results['strategy']
            solve_rate = results['solve_rate']
            avg_attempts = results['avg_attempts']
            
            print(f"{strategy_name:<25} {solve_rate*100:>8.1f}%    {avg_attempts:>8.2f}")
        
        # Key findings
        print("\n" + "=" * 80)
        print("🔍 KEY FINDINGS")
        print("=" * 80)
        
        best_strategy = sorted_strategies[0]
        best_name = best_strategy[1]['strategy']
        best_solve_rate = best_strategy[1]['solve_rate']
        
        print(f"🏆 Best Strategy: {best_name}")
        print(f"   Solve Rate: {best_solve_rate*100:.1f}%")
        print(f"   Average Attempts: {best_strategy[1]['avg_attempts']:.2f}")
        
        # Compare baselines
        entropy_results = strategies.get('pure_entropy', {})
        wordfreq_results = strategies.get('pure_wordfreq', {})
        
        if entropy_results and wordfreq_results:
            entropy_rate = entropy_results['solve_rate']
            wordfreq_rate = wordfreq_results['solve_rate']
            
            print("\n📊 Baseline Comparison:")
            print(f"   Pure Entropy: {entropy_rate*100:.1f}% ({entropy_results['avg_attempts']:.2f} avg)")
            print(f"   Pure WordFreq: {wordfreq_rate*100:.1f}% ({wordfreq_results['avg_attempts']:.2f} avg)")
            
            if entropy_rate > wordfreq_rate:
                print(f"   → Entropy leads by {(entropy_rate - wordfreq_rate)*100:.1f} percentage points")
            else:
                print(f"   → WordFreq leads by {(wordfreq_rate - entropy_rate)*100:.1f} percentage points")
        
        # Transition recommendations
        step_strategies = {k: v for k, v in strategies.items() if k.startswith('step_')}
        count_strategies = {k: v for k, v in strategies.items() if k.startswith('count_')}
        
        if step_strategies:
            best_step = max(step_strategies.items(), key=lambda x: x[1]['solve_rate'])
            step_num = best_step[0].split('_')[1]
            print(f"\n🎯 Best Step Transition: Switch at step {step_num}")
            print(f"   Performance: {best_step[1]['solve_rate']*100:.1f}% solve rate")
        
        if count_strategies:
            best_count = max(count_strategies.items(), key=lambda x: x[1]['solve_rate'])
            count_num = best_count[0].split('_')[1]
            print(f"\n📊 Best Count Transition: Switch at ≤{count_num} words")
            print(f"   Performance: {best_count[1]['solve_rate']*100:.1f}% solve rate")


def main():
    """Main function to run the transition point analysis."""
    print("🔍 KNOWN WORD TRANSITION POINT ANALYSIS")
    print("=" * 60)
    
    # Load word lists
    word_list = load_word_list_with_fallback("words_alpha5.txt")
    print(f"📚 Loaded {len(word_list)} total words")
    
    # Load test words
    test_file = os.path.join(DATA_DIR, "words_challenging.txt")
    test_words = load_words(test_file)
    
    if not test_words:
        print("⚠️  Challenging words not found, using past Wordle words")
        test_file = os.path.join(DATA_DIR, "words_past5_date.txt")
        test_words = load_words(test_file)
        
        if not test_words:
            print("⚠️  Past Wordle words not found, using word list subset")
            test_words = word_list[:100]
    
    print(f"🎯 Loaded {len(test_words)} challenging words for testing")
    
    # Run analysis
    analyzer = TransitionPointAnalyzer(word_list, test_words)
    strategies = analyzer.run_analysis()
    analyzer.display_results(strategies)
    
    print("\n✅ Analysis complete!")
    print(f"📊 Tested {len(strategies)} different strategies")


if __name__ == "__main__":
    main()
