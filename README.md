# Wordle Analysis Project

A comprehensive Wordle analysis and solver suite written in Python and Fortran.

## Repository Structure

```
wordle/
├── run_wordle.py          # Main entry point - run this from the repository root
├── cleanup_word_files.py  # Word file cleanup utility - removes duplicates
├── OPTIMIZATION.md        # Performance optimization guide and recommendations
├── bin/                   # Compiled executables
│   ├── filter-list        # Fortran utility
│   └── wordle             # Fortran Wordle solver
├── src/
│   ├── fortran/           # Fortran source files
│   │   ├── filter-list.f90
│   │   └── wordle.f90
│   └── python/
│       ├── core/          # Core library modules
│       │   └── wordle_utils.py
│       ├── analysis/      # Analysis scripts
│       │   ├── find_best_entropy.py
│       │   └── find_best_letter_frequency.py
│       └── cli/           # Command-line interface
│           └── wordle.py
├── tests/                 # Test files
│   ├── test_attempt_counting.py
│   ├── test_feedback.py
│   ├── test_interactive_rejection.py
│   └── test_rejection.py
├── logs/                  # Log files and solver states
├── data/                  # Word lists and data files
│   ├── words_alpha.txt
│   ├── words_alpha5.txt
│   ├── words_alpha5_100.txt
│   ├── words_past5_alpha.txt
│   ├── words_past5_date.txt
│   └── words_past5_rev_date.txt
├── obj/                   # Build artifacts
└── makefile              # Build configuration for Fortran
```

## Usage

### Quick Start

From the repository root directory, run:

```bash
python3 run_wordle.py
```

This will present you with a menu of options:

1. **Interactive Wordle Solver** - Play Wordle with AI assistance
2. **Automated Testing** - Run comprehensive solver testing
3. **Find Best Entropy Words** - Analyze words by information entropy
4. **Find Best Frequency Words** - Analyze words by letter frequency
5. **Run Tests** - Execute the test suite

### Utilities

```bash
# Clean word files (remove duplicates)
python3 cleanup_word_files.py
python3 cleanup_word_files.py --dry-run  # Preview changes
python3 cleanup_word_files.py --files words_missing.txt  # Clean specific file
```

For detailed optimization recommendations, see `OPTIMIZATION.md`.

### Running Individual Components

You can also run components directly:

```bash
# Interactive solver
python3 -m src.python.cli.wordle

# Analysis tools
python3 -m src.python.analysis.find_best_entropy
python3 -m src.python.analysis.find_best_letter_frequency
python3 -m src.python.analysis.find_best_wordfreq
python3 -m src.python.analysis.find_best_transition_point  # NEW: Transition analysis

# Tests
python3 -m tests.test_feedback
```

### Fortran Components

To compile and run the Fortran components:

```bash
# Compile
make

# Run filter utility
./bin/filter-list

# Run Fortran solver
./bin/wordle
```

## Features

- **Multiple Solving Strategies**:
  - **Entropy-based**: Maximum information gain approach
  - **Frequency-based**: Letter frequency and real-world word usage
  - **Information-based**: Balanced information theory approach
  - **Adaptive Hybrid**: Dynamic weighting of strategies by game stage
  - **Optimal Transition**: 🆕 Empirically optimized strategy (entropy→wordfreq at ≤50 words)
  - **Ultra Efficient**: Speed-optimized approach
- **Interactive Mode**: Get AI assistance while playing Wordle
- **Automated Testing**: Comprehensive testing against historical Wordle words
- **Strategy Analysis**: 🆕 Transition point analysis to determine optimal switching strategies
- **Word Analysis**: Find optimal starting words and strategies
- **Fortran Implementation**: High-performance alternative solver
- **Extensible Design**: Modular structure for easy enhancement

## File Paths

All file paths are now relative to the repository root, making the project portable. The Python modules automatically detect the repository structure and construct appropriate paths.

## Contributing

When adding new features:
- Place core utilities in `src/python/core/`
- Place analysis scripts in `src/python/analysis/`
- Place CLI tools in `src/python/cli/`
- Place tests in `tests/`
- Place logs in `logs/`
- Use relative paths from the repository root
