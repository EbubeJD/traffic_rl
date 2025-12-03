"""
Traffic reward function for RL-based traffic signal control.

The reward is designed to minimize congestion and waiting times at intersections.
"""

from typing import Dict


def compute_traffic_reward(
    group_metrics: Dict[str, float],
    phase_changed: bool,
    config: Dict[str, float]
) -> float:
    """
    Compute dense reward for traffic signal control.

    The reward is more negative when traffic conditions are worse (higher queues,
    longer wait times). This encourages the agent to minimize congestion.

    Args:
        group_metrics: Dictionary containing aggregated metrics across all traffic lights:
            - 'total_queue_ema': Sum of queue_ema across all lights (vehicles)
            - 'total_avg_wait': Sum of avg_wait across all lights (seconds)
            - 'total_num_long_wait': Sum of num_long_wait_60s across all lights (count)
        phase_changed: Boolean indicating if phase was switched this step
        config: Dictionary with reward weights:
            - 'alpha': Weight for queue length (default 1.0)
            - 'beta': Weight for average wait time (default 0.5)
            - 'gamma': Weight for long waiters (default 2.0)
            - 'delta': Penalty for phase switches (default 0.1)

    Returns:
        Scalar reward (typically in range [-50, 0] for low congestion)

    Example:
        >>> metrics = {'total_queue_ema': 5.0, 'total_avg_wait': 10.0, 'total_num_long_wait': 2}
        >>> config = {'alpha': 1.0, 'beta': 0.5, 'gamma': 2.0, 'delta': 0.1}
        >>> reward = compute_traffic_reward(metrics, phase_changed=False, config=config)
        >>> # reward = -(1.0*5.0 + 0.5*10.0 + 2.0*2) = -14.0
    """
    # Extract metrics
    total_queue_ema = group_metrics.get('total_queue_ema', 0.0)
    total_avg_wait = group_metrics.get('total_avg_wait', 0.0)
    total_num_long_wait = group_metrics.get('total_num_long_wait', 0)

    # Extract weights
    alpha = config.get('alpha', 1.0)
    beta = config.get('beta', 0.5)
    gamma = config.get('gamma', 2.0)
    delta = config.get('delta', 0.1)

    # Compute base reward (negative congestion)
    reward = -(
        alpha * total_queue_ema +
        beta * total_avg_wait +
        gamma * total_num_long_wait
    )

    # Apply phase switch penalty
    if phase_changed:
        reward -= delta

    return float(reward)


# Default reward configuration
DEFAULT_REWARD_CONFIG = {
    "alpha": 1.0,    # Queue length weight (vehicles)
    "beta": 0.5,     # Average wait weight (seconds)
    "gamma": 2.0,    # Long wait penalty (equity focus)
    "delta": 0.1,    # Phase switch penalty (discourage thrashing)
}
