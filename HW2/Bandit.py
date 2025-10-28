from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from datetime import datetime

# Configure logger to save to file in logs folder
# NOTE: All paths in this file are RELATIVE paths for reproducibility across environments
log_filename = f"logs/bandit_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.add(log_filename, level="DEBUG")


def format_axis(ax, title, xlabel, ylabel, legend=True, grid=True):
    """Helper to format matplotlib axes consistently
    
    :param ax: Matplotlib axis object
    :param title: Plot title
    :param xlabel: X-axis label
    :param ylabel: Y-axis label
    :param legend: Whether to show legend (Default: True)
    :param grid: Whether to show grid (Default: True)
    """
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()
    if grid:
        ax.grid(True, alpha=0.3)

def save_plot(path, message=None):
    """Helper to save plot and log consistently
    
    :param path: Path to save plot
    :param message: Optional custom log message
    """
    plt.savefig(path, dpi=300, bbox_inches='tight')
    logger.info(message or f"Saved {path}")
    plt.close()

def log_section(title):
    """Helper to log section headers consistently
    
    :param title: Section title
    """
    logger.info("\n" + "=" * 80)
    logger.info(title)
    logger.info("=" * 80)


def get_avg_reward(bandit):
    """Helper to calculate average reward consistently
    
    :param bandit: Bandit instance with total_reward and total_count
    :returns: Average reward per trial
    :rtype: float
    """
    return bandit.total_reward / bandit.total_count if bandit.total_count > 0 else 0.0

def get_epsilon_strategy_info(bandit):
    """Get epsilon strategy information as a formatted string

    :param bandit: EpsilonGreedy instance with epsilon_strategy attribute
    :returns: Formatted epsilon decay formula with parameters
    :rtype: str

    """
    if bandit.epsilon_strategy == "inverse":
        return "ε(t) = 1/t"
    elif bandit.epsilon_strategy == "exponential":
        return f"ε(t) = {bandit.epsilon_0}×{bandit.alpha}^t"
    elif bandit.epsilon_strategy == "linear":
        return f"ε(t) = max({bandit.epsilon_0} - {bandit.k}×t, {bandit.epsilon_min})"
    elif bandit.epsilon_strategy == "logarithmic":
        return f"ε(t) = {bandit.a}/log({bandit.b}×t + {bandit.c:.2f})"
    return "Unknown"

def calculate_metrics(bandit, algorithm_name):
    """Calculate performance metrics for a bandit algorithm

    :param bandit: Bandit instance with experiment results
    :param algorithm_name: Name of the algorithm for labeling
    :returns: Dictionary containing all calculated metrics including:
            - avg_reward: Average reward per trial
            - optimal_mean: True optimal mean reward
            - total_regret: Total regret accumulated
            - avg_regret: Average regret per trial
            - cumulative_rewards: Array of cumulative rewards over time
            - cumulative_regrets: Array of cumulative regrets over time
    :rtype: dict

    """
    metrics = {}
    metrics['avg_reward'] = get_avg_reward(bandit)
    metrics['optimal_mean'] = max(bandit.true_means)
    metrics['optimal_reward'] = metrics['optimal_mean'] * bandit.total_count
    metrics['total_regret'] = metrics['optimal_reward'] - bandit.total_reward
    metrics['avg_regret'] = metrics['total_regret'] / bandit.total_count if bandit.total_count > 0 else 0
    metrics['cumulative_rewards'] = np.cumsum(bandit.rewards_history)
    metrics['optimal_rewards'] = np.arange(1, len(bandit.rewards_history) + 1) * metrics['optimal_mean']
    metrics['cumulative_regrets'] = metrics['optimal_rewards'] - metrics['cumulative_rewards']
    metrics['algorithm_name'] = algorithm_name
    return metrics

def save_results_to_csv(bandit, filename, algorithm_name):
    """Save bandit experiment results to CSV file

    :param bandit: Bandit instance with experiment results
    :param filename: Path to output CSV file (relative path)
    :param algorithm_name: Name of the algorithm for CSV labeling

    """
    metrics = calculate_metrics(bandit, algorithm_name)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Trial', 'Bandit', 'Reward', 'Cumulative_Reward', 'Cumulative_Regret', 'Algorithm'])
        
        for i in range(len(bandit.rewards_history)):
            writer.writerow([
                i + 1,
                bandit.chosen_arms_history[i],
                f"{bandit.rewards_history[i]:.4f}",
                f"{metrics['cumulative_rewards'][i]:.4f}",
                f"{metrics['cumulative_regrets'][i]:.4f}",
                algorithm_name
            ])
    
    logger.info(f"Results saved to {filename}")

class Bandit(ABC):
    """Abstract base class for Multi-Armed Bandit algorithms
    
    This class defines the interface that all bandit algorithms must implement.
    Concrete implementations include EpsilonGreedy and ThompsonSampling.
    
    All subclasses must implement:
        - __init__: Initialize the bandit with reward parameters
        - __repr__: String representation
        - pull: Select an arm and receive reward
        - update: Update internal state after receiving reward
        - experiment: Run full experiment for N trials
        - report: Generate and save results

    """
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        """
        Initialize the bandit algorithm
        
        Args:
            p: List of true mean rewards for each arm
        """
        pass

    @abstractmethod
    def __repr__(self):
        """Return string representation of the bandit
        
        :returns: Human-readable description of the bandit instance
        :rtype: str
        """
        pass

    @abstractmethod
    def pull(self):
        """Select an arm and receive a reward

        :returns: (chosen_arm_index, reward_value)

        :rtype: tuple

        """
        pass

    @abstractmethod
    def update(self):
        """Update internal statistics after pulling an arm

        :returns: None

        """
        pass

    @abstractmethod
    def experiment(self):
        """Run the full bandit experiment for specified number of trials
        
        Executes the pull-update loop for all trials and logs progress.

        :returns: None

        """
        pass

    @abstractmethod
    def report(self):
        """Generate and save experiment results
        
        Saves results to CSV file and logs performance metrics including:
        - Average reward and regret
        - Total reward and regret
        - Arm pull statistics

        :returns: None

        """
        pass
    
    def cumulative_reward(self):
        """Get cumulative rewards over time
        
        :returns: Array of cumulative rewards at each trial
        :rtype: numpy.ndarray
        """
        return np.cumsum(self.rewards_history)
    
    def cumulative_regret(self):
        """Get cumulative regrets over time
        
        :returns: Array of cumulative regrets at each trial
        :rtype: numpy.ndarray
        """
        optimal_mean = max(self.true_means)
        optimal_rewards = np.arange(1, len(self.rewards_history) + 1) * optimal_mean
        return optimal_rewards - self.cumulative_reward()

#--------------------------------------#

class Visualization():
    """Visualization class for bandit algorithm comparisons
    
    Provides methods to create comprehensive plots comparing:
    - Learning curves (estimated means over time)
    - Cumulative rewards and regrets
    - Algorithm performance comparisons
    
    Supports both Epsilon-Greedy and Thompson Sampling visualizations
    with automatic parameter annotation in plot titles and legends.

    """

    def plot1(self, eg_bandit=None, ts_bandit=None):
        """Visualize the performance of each bandit: linear and log scale

        :param eg_bandit: EpsilonGreedy bandit instance (Default value = None)
        :param ts_bandit: ThompsonSampling bandit instance (Default value = None)

        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Epsilon-Greedy plots
        if eg_bandit is not None:
            arm_estimates = np.array(eg_bandit.arm_estimates)
            T, K = arm_estimates.shape
            
            # Get epsilon strategy info
            epsilon_info = f"Strategy: {get_epsilon_strategy_info(eg_bandit)}"
            
            # Linear scale
            ax = axes[0, 0]
            for k in range(K):
                ax.plot(arm_estimates[:, k], label=f"Arm {k} (true={eg_bandit.true_means[k]})")
            format_axis(ax, f"Epsilon-Greedy: Estimated Means (Linear Scale)\n{epsilon_info}", 
                       "Trials", "Estimated Mean Reward")
            
            # Log scale
            ax = axes[0, 1]
            for k in range(K):
                ax.plot(arm_estimates[:, k], label=f"Arm {k} (true={eg_bandit.true_means[k]})")
            ax.set_xscale("log")
            format_axis(ax, f"Epsilon-Greedy: Estimated Means (Log Scale)\n{epsilon_info}",
                       "Trials (log scale)", "Estimated Mean Reward")
        
        # Thompson Sampling plots
        if ts_bandit is not None:
            arm_post_means = np.array(ts_bandit.arm_post_means)
            T, K = arm_post_means.shape
            
            # Linear scale
            ax = axes[1, 0]
            for k in range(K):
                ax.plot(arm_post_means[:, k], label=f"Arm {k} (true={ts_bandit.true_means[k]})")
            format_axis(ax, f"Thompson Sampling: Posterior Means (Linear Scale)\nPrecision τ={ts_bandit.tau:.4f}",
                       "Trials", "Posterior Mean")
            
            # Log scale
            ax = axes[1, 1]
            for k in range(K):
                ax.plot(arm_post_means[:, k], label=f"Arm {k} (true={ts_bandit.true_means[k]})")
            ax.set_xscale("log")
            format_axis(ax, f"Thompson Sampling: Posterior Means (Log Scale)\nPrecision τ={ts_bandit.tau:.4f}",
                       "Trials (log scale)", "Posterior Mean")
        
        plt.tight_layout()
        save_plot("img/plot1_learning_curves.png")
    
    def plot2(self, eg_bandit=None, ts_bandit=None):
        """Compare E-greedy and Thompson sampling cumulative rewards and regrets

        :param eg_bandit: EpsilonGreedy bandit instance (Default value = None)
        :param ts_bandit: ThompsonSampling bandit instance (Default value = None)

        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Build algorithm labels with parameters
        eg_label = "Epsilon-Greedy"
        ts_label = "Thompson Sampling"
        
        if eg_bandit is not None:
            eg_label = f"Epsilon-Greedy ({get_epsilon_strategy_info(eg_bandit)})"
        
        if ts_bandit is not None:
            ts_label = f"Thompson Sampling (τ={ts_bandit.tau})"
        
        # Cumulative Rewards
        ax = axes[0]
        if eg_bandit is not None:
            ax.plot(eg_bandit.cumulative_reward(), label=eg_label, linewidth=2)
        if ts_bandit is not None:
            ax.plot(ts_bandit.cumulative_reward(), label=ts_label, linewidth=2)
        format_axis(ax, "Cumulative Rewards Comparison", "Trials", "Cumulative Reward")
        ax.legend(fontsize=9)
        
        # Cumulative Regrets
        ax = axes[1]
        if eg_bandit is not None:
            ax.plot(eg_bandit.cumulative_regret(), label=eg_label, linewidth=2)
        if ts_bandit is not None:
            ax.plot(ts_bandit.cumulative_regret(), label=ts_label, linewidth=2)
        format_axis(ax, "Cumulative Regrets Comparison", "Trials", "Cumulative Regret")
        ax.legend(fontsize=9)
        
        plt.tight_layout()
        save_plot("img/plot2_cumulative_rewards_and_regrets.png")

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """Epsilon-Greedy Algorithm with Gaussian Rewards and Decaying Exploration
    - Supports multiple epsilon decay strategies
    - Each arm gives continuous rewards from Normal(mean, variance=1)

    """
    
    def __init__(self, p, trials=20000, epsilon_strategy="inverse", epsilon_params=None):
        """
        Initialize the Epsilon-Greedy bandit with Gaussian rewards
        
        Args:
            p: list of true mean rewards for each arm (Bandit_Reward=[1,2,3,4])
            trials: number of trials to run (default 20000)
            epsilon_strategy: strategy for epsilon decay
                - "inverse": ε(t) = 1/t (default)
                - "exponential": ε(t) = ε₀ * αᵗ
                - "linear": ε(t) = max(ε₀ − k·t, ε_min)
                - "logarithmic": ε(t) = a / log(b·t + c)
            epsilon_params: dict of parameters for the chosen strategy
        """
        self.true_means = list(p)  # True mean rewards (for simulation)
        self.n_arms = len(p)
        self.trials = trials
        self.tau = 1.0  # Known precision (variance = 1.0)
        self.variance = 1.0 / self.tau
        
        # Epsilon strategy configuration
        self.epsilon_strategy = epsilon_strategy
        self.epsilon_params = epsilon_params or {}
        self._setup_epsilon_params()
        
        # Track statistics for each arm
        self.counts = [0] * self.n_arms  # Number of times each arm was pulled
        self.sum_rewards = [0.0] * self.n_arms  # Sum of rewards per arm
        self.values = [0.0] * self.n_arms  # Estimated mean reward for each arm
        
        # Track history
        self.total_reward = 0.0
        self.total_count = 0
        self.rewards_history = []
        self.chosen_arms_history = []
        self.arm_estimates = []  # Track estimated means over time for visualization
        self.epsilon_history = []  # Track epsilon values over time
        
        # Initialize RNG with fixed seed for reproducibility across runs
        # Seed=7 ensures same random sequences in any environment
        self.rng = np.random.default_rng(7)
        
        logger.info(f"Initialized EpsilonGreedy with {self.n_arms} arms (Gaussian rewards, strategy={epsilon_strategy})")
    
    def _setup_epsilon_params(self):
        """Setup default parameters for different epsilon decay strategies
        
        Configures parameters based on the selected epsilon_strategy:
        - inverse: No parameters needed (ε(t) = 1/t)
        - exponential: epsilon_0, alpha for exponential decay
        - linear: epsilon_0, k, epsilon_min for linear decay with floor
        - logarithmic: a, b, c for logarithmic decay

        """
        if self.epsilon_strategy == "inverse":
            # ε(t) = 1/t - no additional params needed
            pass
        elif self.epsilon_strategy == "exponential":
            # ε(t) = ε₀ * αᵗ - exponential decay starting from epsilon_0
            self.epsilon_0 = self.epsilon_params.get("epsilon_0", 1.0)
            self.alpha = self.epsilon_params.get("alpha", 0.9995)
        elif self.epsilon_strategy == "linear":
            # ε(t) = max(ε₀ − k·t, ε_min) - linear decay with minimum floor
            self.epsilon_0 = self.epsilon_params.get("epsilon_0", 1.0)
            self.k = self.epsilon_params.get("k", 0.0001)
            self.epsilon_min = self.epsilon_params.get("epsilon_min", 0.01)
        elif self.epsilon_strategy == "logarithmic":
            # ε(t) = a / log(b·t + c) - logarithmic decay
            self.a = self.epsilon_params.get("a", 1.0)
            self.b = self.epsilon_params.get("b", 1.0)
            self.c = self.epsilon_params.get("c", math.e)
    
    def __repr__(self):
        """String representation of the bandit
        
        :returns: Human-readable description
        :rtype: str
        """
        return f"EpsilonGreedy(arms={self.n_arms}, strategy={self.epsilon_strategy}, trials={self.trials})"
    
    def _epsilon(self, t):
        """Calculate epsilon value for time step t based on chosen strategy

        :param t: Current trial number (1-indexed)
        :returns: Epsilon value for exploration probability at time t
        :rtype: float

        """
        if self.epsilon_strategy == "inverse":
            # ε(t) = 1/t - standard inverse decay
            return 1.0 / max(1, t)
        elif self.epsilon_strategy == "exponential":
            # ε(t) = ε₀ * αᵗ - exponential decay
            return self.epsilon_0 * (self.alpha ** t)
        elif self.epsilon_strategy == "linear":
            # ε(t) = max(ε₀ − k·t, ε_min) - linear decay with floor
            return max(self.epsilon_0 - self.k * t, self.epsilon_min)
        elif self.epsilon_strategy == "logarithmic":
            # ε(t) = a / log(b·t + c) - logarithmic decay
            return self.a / math.log(self.b * t + self.c)
        else:
            # Default fallback to inverse strategy
            return 1.0 / max(1, t)
    
    def pull(self):
        """Select an arm using epsilon-greedy strategy with decaying epsilon

        :returns: (chosen_arm_index, reward_value)

        :rtype: tuple

        """
        t = self.total_count + 1
        eps = self._epsilon(t)
        
        # Exploration: choose random arm with probability ε(t)
        if self.rng.random() < eps:
            chosen_arm = int(self.rng.integers(0, self.n_arms))
            logger.debug(f"Trial {t}: Exploring (ε={eps:.3f}): randomly chose arm {chosen_arm}")
        # Exploitation: choose best known arm
        else:
            # Choose arm with highest estimated mean (ties broken randomly)
            max_val = max(self.values)
            best_arms = [i for i, v in enumerate(self.values) if v == max_val]
            chosen_arm = int(self.rng.choice(best_arms))
            logger.debug(f"Trial {t}: Exploiting (ε={eps:.3f}): chose best arm {chosen_arm} with value {self.values[chosen_arm]:.3f}")
        
        # Simulate pulling the arm: Gaussian reward with true mean and variance=1
        reward = float(self.rng.normal(loc=self.true_means[chosen_arm], scale=math.sqrt(self.variance)))
        
        return chosen_arm, reward
    
    def update(self, chosen_arm, reward):
        """Update statistics after receiving a reward

        :param chosen_arm: index of the arm that was pulled
        :param reward: continuous reward received
        :returns: None

        """
        # Update counts and sums
        self.counts[chosen_arm] += 1
        self.sum_rewards[chosen_arm] += reward
        self.total_count += 1
        self.total_reward += reward
        
        # Update estimated mean using sample average
        self.values[chosen_arm] = self.sum_rewards[chosen_arm] / self.counts[chosen_arm]
        
        # Store history
        self.rewards_history.append(reward)
        self.chosen_arms_history.append(chosen_arm)
        self.arm_estimates.append(self.values.copy())
        self.epsilon_history.append(self._epsilon(self.total_count))
        
        logger.debug(f"Updated arm {chosen_arm}: new mean estimate={self.values[chosen_arm]:.3f}, count={self.counts[chosen_arm]}")
    
    def experiment(self, n_trials=None):
        """Run the experiment for n_trials

        :param n_trials: number of trials to run (uses self.trials if None) (Default value = None)
        :returns: None

        """
        if n_trials is None:
            n_trials = self.trials
        
        logger.info(f"Starting Epsilon-Greedy experiment with {n_trials} trials (Gaussian rewards)")
        
        for trial in range(n_trials):
            chosen_arm, reward = self.pull()
            self.update(chosen_arm, reward)
            
            if (trial + 1) % 2000 == 0:
                logger.info(f"Trial {trial + 1}/{n_trials}: Avg reward = {get_avg_reward(self):.3f}")
        
        logger.info(f"Experiment completed! Total reward: {self.total_reward:.2f}")
    
    def report(self, filename="report/epsilon_greedy_results.csv"):
        """Save results to CSV and log statistics

        :param filename: name of the CSV file to save results (Default value = "report/epsilon_greedy_results.csv")
        :returns: None

        """
        # Save to CSV using helper function
        save_results_to_csv(self, filename, "EpsilonGreedy")
        
        # Calculate metrics
        metrics = calculate_metrics(self, "EpsilonGreedy")
        
        # Log statistics
        log_section("EPSILON-GREEDY RESULTS (Gaussian Rewards)")
        logger.info(f"Average Reward: {metrics['avg_reward']:.4f}")
        logger.info(f"Average Regret: {metrics['avg_regret']:.4f}")
        logger.info(f"Total Reward: {self.total_reward:.2f}")
        logger.info(f"Total Regret: {metrics['total_regret']:.2f}")
        logger.info(f"Total Trials: {self.total_count}")
        logger.info(f"Arm Pulls: {self.counts}")
        logger.info(f"Estimated Means: {[f'{v:.3f}' for v in self.values]}")
        logger.info(f"True Means: {[f'{v:.3f}' for v in self.true_means]}")

#--------------------------------------#

class ThompsonSampling(Bandit):
    """Thompson Sampling for Gaussian rewards with KNOWN precision.
    
    Conjugate model:
        Prior: μ ~ Normal(μ0, 1/τ0)
        Likelihood (reward): x | μ ~ Normal(μ, 1/τ) with τ KNOWN
        Posterior: μ | data ~ Normal(μ_n, 1/τ_n)
            where τ_n = τ0 + n·τ
            and μ_n = (τ0·μ0 + τ·∑x) / τ_n

    """
    
    def __init__(self, p, trials=20000):
        """
        Initialize Thompson Sampling bandit with Gaussian rewards
        
        Args:
            p: list of true mean rewards for each arm (Bandit_Reward=[1,2,3,4])
            trials: number of trials to run (default 20000)
        """
        self.true_means = list(p)
        self.n_arms = len(p)
        self.trials = trials
        self.tau = 1.0  # Known precision
        self.variance = 1.0 / self.tau
        
        # Bayesian statistics for each arm
        self.counts = [0] * self.n_arms
        self.sum_rewards = [0.0] * self.n_arms
        
        # Prior parameters (diffuse/non-informative)
        self.mu0 = 0.0
        self.tau0 = 1e-6
        
        # Posterior parameters (updated with data)
        self.mu_post = [self.mu0] * self.n_arms
        self.tau_post = [self.tau0] * self.n_arms
        
        # Track history
        self.total_reward = 0.0
        self.total_count = 0
        self.rewards_history = []
        self.chosen_arms_history = []
        self.arm_post_means = []  # Track posterior means over time
        
        # Initialize RNG with fixed seed for reproducibility across runs
        # Seed=11 (different from EpsilonGreedy) to ensure independent random sequences
        self.rng = np.random.default_rng(11)
        
        logger.info(f"Initialized ThompsonSampling with {self.n_arms} arms (Gaussian, known precision τ={self.tau})")
    
    def __repr__(self):
        """String representation of the bandit
        
        :returns: Human-readable description
        :rtype: str
        """
        return f"ThompsonSampling(arms={self.n_arms}, Gaussian, tau={self.tau}, trials={self.trials})"
    
    def pull(self):
        """Select an arm using Thompson Sampling
        Sample from each arm's posterior and choose the best

        :returns: (chosen_arm_index, reward_value)

        :rtype: tuple

        """
        # Sample from each arm's posterior distribution
        samples = []
        for k in range(self.n_arms):
            var_mu = 1.0 / self.tau_post[k]
            sample = self.rng.normal(loc=self.mu_post[k], scale=math.sqrt(var_mu))
            samples.append(sample)
        
        # Choose arm with highest sample (ties broken randomly)
        max_sample = max(samples)
        best_arms = [i for i, s in enumerate(samples) if s == max_sample]
        chosen_arm = int(self.rng.choice(best_arms))
        
        logger.debug(f"Trial {self.total_count + 1}: Sampled {[f'{s:.3f}' for s in samples]}, chose arm {chosen_arm}")
        
        # Pull the chosen arm and get reward
        reward = float(self.rng.normal(loc=self.true_means[chosen_arm], scale=math.sqrt(self.variance)))
        
        return chosen_arm, reward
    
    def update(self, chosen_arm, reward):
        """Update posterior distribution using Bayesian update (conjugate prior)

        :param chosen_arm: index of the arm that was pulled
        :param reward: continuous reward received
        :returns: None

        """
        # Update counts and sums
        self.counts[chosen_arm] += 1
        self.sum_rewards[chosen_arm] += reward
        n = self.counts[chosen_arm]
        
        # Bayesian update for Normal-Normal conjugate model with known precision
        self.tau_post[chosen_arm] = self.tau0 + self.tau * n
        self.mu_post[chosen_arm] = (self.tau0 * self.mu0 + self.tau * self.sum_rewards[chosen_arm]) / self.tau_post[chosen_arm]
        
        # Update totals
        self.total_count += 1
        self.total_reward += reward
        
        # Store history
        self.rewards_history.append(reward)
        self.chosen_arms_history.append(chosen_arm)
        self.arm_post_means.append(self.mu_post.copy())
        
        logger.debug(f"Updated arm {chosen_arm}: posterior mean={self.mu_post[chosen_arm]:.3f}, count={n}")
    
    def experiment(self, n_trials=None):
        """Run the Thompson Sampling experiment

        :param n_trials: number of trials to run (uses self.trials if None) (Default value = None)
        :returns: None

        """
        if n_trials is None:
            n_trials = self.trials
        
        logger.info(f"Starting Thompson Sampling experiment with {n_trials} trials (Gaussian, known precision)")
        
        for trial in range(n_trials):
            chosen_arm, reward = self.pull()
            self.update(chosen_arm, reward)
            
            if (trial + 1) % 2000 == 0:
                logger.info(f"Trial {trial + 1}/{n_trials}: Avg reward = {get_avg_reward(self):.3f}")
        
        logger.info(f"Experiment completed! Total reward: {self.total_reward:.2f}")
    
    def report(self, filename="report/thompson_sampling_results.csv"):
        """Save results to CSV and log statistics

        :param filename: name of the CSV file to save results (Default value = "report/thompson_sampling_results.csv")
        :returns: None

        """
        # Save to CSV using helper function
        save_results_to_csv(self, filename, "ThompsonSampling")
        
        # Calculate metrics
        metrics = calculate_metrics(self, "ThompsonSampling")
        
        # Log statistics
        log_section("THOMPSON SAMPLING RESULTS (Gaussian Rewards)")
        logger.info(f"Average Reward: {metrics['avg_reward']:.4f}")
        logger.info(f"Average Regret: {metrics['avg_regret']:.4f}")
        logger.info(f"Total Reward: {self.total_reward:.2f}")
        logger.info(f"Total Regret: {metrics['total_regret']:.2f}")
        logger.info(f"Total Trials: {self.total_count}")
        logger.info(f"Arm Pulls: {self.counts}")
        logger.info(f"Posterior Means: {[f'{v:.3f}' for v in self.mu_post]}")
        logger.info(f"True Means: {[f'{v:.3f}' for v in self.true_means]}")

def comparison(eg_bandit, ts_bandit):
    """Compare the performances of Epsilon-Greedy and Thompson Sampling VISUALLY

    :param eg_bandit: EpsilonGreedy bandit instance (after experiment)
    :param ts_bandit: ThompsonSampling bandit instance (after experiment)

    """
    logger.info("Creating comparison visualizations...")
    
    vis = Visualization()
    
    # Plot 1: Learning curves for both algorithms
    vis.plot1(eg_bandit=eg_bandit, ts_bandit=ts_bandit)
    
    # Plot 2: Cumulative rewards and regrets comparison
    vis.plot2(eg_bandit=eg_bandit, ts_bandit=ts_bandit)
    
    logger.info("Comparison visualizations created successfully!")

def compare_epsilon_strategies(bandit_reward, trials=20000):
    """Compare different epsilon decay strategies

    :param bandit_reward: list of true mean rewards for each arm
    :param trials: number of trials per experiment (Default value = 20000)
    :returns: Dictionary of strategy names to bandit instances
    :rtype: dict

    """
    log_section("COMPARING EPSILON DECAY STRATEGIES")
    
    strategies = {
        "Inverse (1/t)": {
            "strategy": "inverse",
            "params": {}
        },
        "Exponential (ε₀*α^t)": {
            "strategy": "exponential",
            "params": {"epsilon_0": 1.0, "alpha": 0.9995}
        },
        "Linear (max(ε₀-k*t, ε_min))": {
            "strategy": "linear",
            "params": {"epsilon_0": 1.0, "k": 0.00005, "epsilon_min": 0.01}
        },
        "Logarithmic (a/log(b*t+c))": {
            "strategy": "logarithmic",
            "params": {"a": 2.0, "b": 1.0, "c": math.e}
        }
    }
    
    results = {}
    
    # Run experiments with each strategy
    for name, config in strategies.items():
        logger.info(f"\n### Testing: {name} ###")
        bandit = EpsilonGreedy(
            p=bandit_reward,
            trials=trials,
            epsilon_strategy=config["strategy"],
            epsilon_params=config["params"]
        )
        bandit.experiment()
        results[name] = bandit
        
        # Log summary
        avg_reward = get_avg_reward(bandit)
        optimal_mean = max(bandit.true_means)
        total_regret = (optimal_mean * bandit.total_count) - bandit.total_reward
        logger.info(f"Strategy: {name}")
        logger.info(f"  Avg Reward: {avg_reward:.4f}")
        logger.info(f"  Total Regret: {total_regret:.2f}")
        logger.info(f"  Arm Pulls: {bandit.counts}")
    
    # Create comparison plots
    _plot_epsilon_comparison(results, trials)
    
    # Find and report best strategy
    best_strategy = max(results.items(), key=lambda x: x[1].total_reward)
    log_section(f"BEST EPSILON STRATEGY: {best_strategy[0]}")
    logger.info(f"Total Reward: {best_strategy[1].total_reward:.2f}")
    
    return results

def _plot_epsilon_comparison(results, trials):
    """Create comprehensive comparison plots for different epsilon strategies

    :param results: dict of strategy names to bandit instances
    :param trials: number of trials

    """
    # Create 2x2 grid with only the 4 main plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Epsilon decay curves
    ax = axes[0, 0]
    for name, bandit in results.items():
        ax.plot(bandit.epsilon_history, label=name, linewidth=2)
    ax.set_title("Epsilon Decay Curves", fontsize=14, fontweight='bold')
    ax.set_xlabel("Trial")
    ax.set_ylabel("Epsilon (ε)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Epsilon decay curves (log scale)
    ax = axes[0, 1]
    for name, bandit in results.items():
        ax.plot(bandit.epsilon_history, label=name, linewidth=2)
    ax.set_title("Epsilon Decay Curves (Log Scale)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Trial (log scale)")
    ax.set_ylabel("Epsilon (ε)")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative rewards
    ax = axes[1, 0]
    for name, bandit in results.items():
        ax.plot(bandit.cumulative_reward(), label=name, linewidth=2)
    ax.set_title("Cumulative Rewards", fontsize=14, fontweight='bold')
    ax.set_xlabel("Trial")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative regrets
    ax = axes[1, 1]
    for name, bandit in results.items():
        ax.plot(bandit.cumulative_regret(), label=name, linewidth=2)
    ax.set_title("Cumulative Regrets", fontsize=14, fontweight='bold')
    ax.set_xlabel("Trial")
    ax.set_ylabel("Cumulative Regret")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot("img/plot3_epsilon_decay_function_tests.png", 
             "Saved epsilon comparison plot: img/plot3_epsilon_decay_function_tests.png")

def _create_final_comparison_plot(epsilon_results, ts_bandit):
    """Create a final comparison plot including all epsilon strategies and Thompson Sampling

    :param epsilon_results: dict of epsilon strategy results
    :param ts_bandit: Thompson Sampling bandit instance

    """
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Add main title with precision info
    fig.suptitle(f"Complete Algorithm Comparison | Precision τ={ts_bandit.tau}", 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: Cumulative rewards comparison
    ax = fig.add_subplot(gs[0, 0])
    for name, bandit in epsilon_results.items():
        ax.plot(bandit.cumulative_reward(), label=name, linewidth=2, alpha=0.7)
    ax.plot(ts_bandit.cumulative_reward(), label=f"Thompson Sampling (τ={ts_bandit.tau})", 
            linewidth=2.5, color='red', linestyle='--')
    format_axis(ax, "Cumulative Rewards - All Algorithms", "Trial", "Cumulative Reward")
    ax.legend(loc='best', fontsize=8)
    
    # Plot 2: Cumulative regrets comparison
    ax = fig.add_subplot(gs[0, 1])
    for name, bandit in epsilon_results.items():
        ax.plot(bandit.cumulative_regret(), label=name, linewidth=2, alpha=0.7)
    ax.plot(ts_bandit.cumulative_regret(), label=f"Thompson Sampling (τ={ts_bandit.tau})", 
            linewidth=2.5, color='red', linestyle='--')
    format_axis(ax, "Cumulative Regrets - All Algorithms", "Trial", "Cumulative Regret")
    ax.legend(loc='best', fontsize=8)
    
    # Plot 3: Average reward comparison (bar chart)
    ax = fig.add_subplot(gs[1, 0])
    names = list(epsilon_results.keys()) + ["Thompson Sampling"]
    avg_rewards = [get_avg_reward(bandit) for bandit in epsilon_results.values()]
    avg_rewards.append(get_avg_reward(ts_bandit))
    
    colors = ['steelblue'] * len(epsilon_results) + ['red']
    bars = ax.bar(range(len(names)), avg_rewards, color=colors, alpha=0.7)
    ax.set_title("Average Reward - All Algorithms", fontsize=13, fontweight='bold')
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Average Reward")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, avg_rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Total regret comparison (bar chart)
    ax = fig.add_subplot(gs[1, 1])
    optimal_mean = max(ts_bandit.true_means)
    total_regrets = [(optimal_mean * bandit.total_count) - bandit.total_reward 
                     for bandit in epsilon_results.values()]
    total_regrets.append((optimal_mean * ts_bandit.total_count) - ts_bandit.total_reward)
    
    colors = ['steelblue'] * len(epsilon_results) + ['red']
    bars = ax.bar(range(len(names)), total_regrets, color=colors, alpha=0.7)
    ax.set_title("Total Regret - All Algorithms", fontsize=13, fontweight='bold')
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Total Regret")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, total_regrets):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 5: Add a text summary box with parameters
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    
    # Build summary text with parameters for each strategy
    summary_text = "Algorithm Parameters:\n" + "="*80 + "\n\n"
    for name, bandit in epsilon_results.items():
        params = get_epsilon_strategy_info(bandit)
        metrics = calculate_metrics(bandit, name)
        summary_text += f"• {name}: {params}\n  → Avg Reward: {metrics['avg_reward']:.4f}, Total Regret: {metrics['total_regret']:.2f}\n\n"
    
    summary_text += f"• Thompson Sampling: Bayesian with known precision τ={ts_bandit.tau}\n"
    ts_metrics = calculate_metrics(ts_bandit, "ThompsonSampling")
    summary_text += f"  → Avg Reward: {ts_metrics['avg_reward']:.4f}, Total Regret: {ts_metrics['total_regret']:.2f}"
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            family='monospace')
    
    save_plot("img/plot4_different_version_comparison.png", 
             "Saved final comparison plot: img/plot4_different_version_comparison.png")

if __name__=='__main__':
    """
    Main execution block for Multi-Armed Bandit experiments
    
    This script runs in two phases:
    1. Compare different epsilon decay strategies for Epsilon-Greedy
    2. Compare best epsilon strategy against Thompson Sampling
    
    All results are saved to:
    - img/: visualization plots
    - report/: CSV files with detailed results
    - logs/: execution logs with timestamp
    """
    
    # Test logger levels
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
    
    logger.info("\n" + "="*80)
    logger.info("MULTI-ARMED BANDIT EXPERIMENT: GAUSSIAN REWARDS")
    logger.info("="*80)
    
    # Assignment defaults
    Bandit_Reward = [1, 2, 3, 4]  # True mean rewards for each arm
    NumberOfTrials = 20000
    
    # STEP 1: Compare different epsilon decay strategies
    log_section("STEP 1: COMPARING EPSILON DECAY STRATEGIES")
    epsilon_results = compare_epsilon_strategies(Bandit_Reward, trials=NumberOfTrials)
    
    # Find the best epsilon strategy
    best_epsilon_name, best_epsilon_bandit = max(epsilon_results.items(), 
                                                   key=lambda x: x[1].total_reward)
    
    # STEP 2: Run full comparison with best epsilon strategy vs Thompson Sampling
    log_section("STEP 2: FINAL COMPARISON - BEST EPSILON VS THOMPSON SAMPLING")
    
    # Create new instance with best epsilon strategy
    logger.info(f"\n### EPSILON-GREEDY (Best Strategy: {best_epsilon_name}) ###")
    eg_bandit_final = EpsilonGreedy(
        p=Bandit_Reward, 
        trials=NumberOfTrials,
        epsilon_strategy=best_epsilon_bandit.epsilon_strategy,
        epsilon_params=best_epsilon_bandit.epsilon_params
    )
    eg_bandit_final.experiment()
    eg_bandit_final.report(filename="report/epsilon_greedy_results.csv")
    
    # Create and run Thompson Sampling
    logger.info("\n### THOMPSON SAMPLING ###")
    ts_bandit = ThompsonSampling(p=Bandit_Reward, trials=NumberOfTrials)
    ts_bandit.experiment()
    ts_bandit.report(filename="report/thompson_sampling_results.csv")
    
    # Compare the two algorithms visually
    logger.info("\n### FINAL COMPARISON ###")
    comparison(eg_bandit_final, ts_bandit)
    
    # Create summary comparison plot
    _create_final_comparison_plot(epsilon_results, ts_bandit)
    
    log_section("EXPERIMENT COMPLETE")
    logger.info(f"Best Epsilon Strategy: {best_epsilon_name}")
    logger.info(f"Best Epsilon Reward: {best_epsilon_bandit.total_reward:.2f}")
    logger.info(f"Thompson Sampling Reward: {ts_bandit.total_reward:.2f}")
