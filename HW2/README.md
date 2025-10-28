# Multi-Armed Bandit Experiment

This project implements and compares different Multi-Armed Bandit algorithms for Gaussian reward distributions. The implementation explores various epsilon-greedy decay strategies and compares them against Thompson Sampling to identify the most effective exploration-exploitation strategy.

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- loguru==0.7.2
- numpy==1.24.3
- matplotlib==3.7.1

## Running the Experiment

Execute the main script:

```bash
python Bandit.py
```

The experiment tests a 4-armed bandit with Gaussian rewards:
- **Arm 0**: μ = 1, σ² = 1
- **Arm 1**: μ = 2, σ² = 1
- **Arm 2**: μ = 3, σ² = 1
- **Arm 3**: μ = 4, σ² = 1 (optimal arm)

**Total trials**: 20,000 per algorithm

## Reproducibility

The code is fully reproducible:
- **Fixed random seeds**: EpsilonGreedy uses seed=7, ThompsonSampling uses seed=11
- **No absolute paths**: All file paths are relative to the project directory
- **Same output**: Running the experiment multiple times produces identical results

## Output

The experiment generates:

1. **Plots** (saved to `img/` folder):
   - `plot1_learning_curves.png`: Learning curves showing arm estimate convergence (linear & log scale)
   - `plot2_cumulative_rewards_and_regrets.png`: Cumulative rewards and regrets comparison
   - `plot3_epsilon_decay_function_tests.png`: Comparison of different epsilon decay strategies
   - `plot4_different_version_comparison.png`: Comprehensive comparison of different Epsilon Greedy decay versions vs Thompson Sampling

2. **CSV Results** (saved to `report/` folder):
   - `epsilon_greedy_results.csv`: Detailed results for best epsilon strategy
   - `thompson_sampling_results.csv`: Detailed results for Thompson Sampling

3. **Logs** (saved to `logs/` folder):
   - `bandit_experiment_YYYYMMDD_HHMMSS.log`: Complete execution log with timestamp and debug information (shows locally, ignored in GitHub)

## Algorithms Implemented

### Epsilon-Greedy
Balances exploration and exploitation using decaying epsilon parameter:
- With probability ε(t): explore (choose random arm)
- With probability 1-ε(t): exploit (choose arm with highest estimated mean)

**Four epsilon decay strategies tested:**
1. **Inverse (1/t)**: ε(t) = 1/t — Simple, theoretically grounded decay
2. **Exponential**: ε(t) = ε₀ × α^t — Fast decay (ε₀=1.0, α=0.9999)
3. **Linear**: ε(t) = max(ε₀ - k·t, ε_min) — Controlled linear reduction (ε₀=1.0, k=0.0001, ε_min=0.01)
4. **Logarithmic**: ε(t) = a / log(b·t + c) — Slow decay (a=1, b=1, c=2)

### Thompson Sampling
Bayesian approach that naturally balances exploration and exploitation:
- Uses Normal-Normal conjugate prior model with known precision (τ=1.0)
- Samples from posterior distribution of each arm's mean
- Selects arm with highest sampled value
- Updates posterior using Bayesian inference after each observation

## Experiment Workflow

The experiment runs in two phases:

### Phase 1: Epsilon Decay Strategy Comparison
- Tests 4 different epsilon decay strategies (Inverse, Exponential, Linear, Logarithmic)
- Runs 20,000 trials for each strategy with fixed random seed (seed=7)
- Identifies the best-performing strategy based on total reward
- Generates `plot3_epsilon_decay_function_tests.png` showing cumulative rewards and regrets

### Phase 2: Final Comparison
- Runs the best epsilon strategy from Phase 1 against Thompson Sampling (seed=11)
- Generates detailed visualizations:
  - `plot1_learning_curves.png`: Arm estimate convergence (linear & log scales)
  - `plot2_cumulative_rewards_and_regrets.png`: Performance comparison
  - `plot4_different_version_comparison.png`: Comprehensive multi-panel comparison
- Saves trial-by-trial results to CSV files
- Logs all statistics and metrics to timestamped log file

## Results

The experiment shows that:
- **Best Epsilon Strategy**: Inverse (1/t) decay outperforms exponential, linear, and logarithmic alternatives
- **Overall Performance**: Thompson Sampling slightly outperforms the best epsilon-greedy variant
- **Convergence**: Both methods successfully identify and exploit the optimal arm

See `report/conclusions.md` for detailed analysis and visualizations.

## Code Structure

- `Bandit.py`: Main implementation
- `requirements.txt`: Python dependencies with specific versions
- `img/`: Generated visualizations (4 plots)
- `report/`: CSV files with detailed trial-by-trial results
- `logs/`: Timestamped execution logs with debug information

## Code Optimizations

The implementation follows best practices:
- **Helper Functions**: Centralized `format_axis()`, `save_plot()`, `log_section()`, `get_avg_reward()` to eliminate redundancy
- **Comprehensive Documentation**: Docstrings for all classes, methods, and helper functions
- **Consistent Formatting**: Standardized logging, plotting, and calculation patterns