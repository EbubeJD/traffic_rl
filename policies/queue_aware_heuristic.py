"""
Queue-aware heuristic policy for traffic signal control.

Implements a greedy max-queue policy that adapts to traffic conditions:
- Keeps phase green when queue exists
- Switches to red when queue clears AND minimum green time elapsed
- Enforces minimum green time to prevent thrashing

Usage:
    from policies import MaxQueuePolicy

    policy = MaxQueuePolicy(min_green_time=10.0, queue_threshold=0.5, dt=2.0)
    action = policy.act(obs, info)
"""

import numpy as np


class MaxQueuePolicy:
    """
    Max-queue heuristic policy for single-lane traffic control.

    Rules:
    1. If queue > threshold, keep phase green (action 0)
    2. If queue ≈ 0 AND min_green_time elapsed, switch to red (action 1)
    3. Enforce minimum green time to avoid rapid phase changes

    This policy is designed to serve queued vehicles while preventing
    thrashing (rapid phase switches that reduce throughput).

    Args:
        min_green_time: Minimum duration (seconds) to hold green before allowing switch
        queue_threshold: Queue count threshold (vehicles) to keep green
        dt: Decision interval (seconds) - should match env step interval
    """

    def __init__(self, min_green_time=10.0, queue_threshold=0.5, dt=2.0):
        """Initialize policy with timing parameters."""
        self.min_green_time = min_green_time
        self.queue_threshold = queue_threshold
        self.dt = dt

        # State tracking
        self.current_phase = 0  # Start with green
        self.time_in_phase = 0.0

    def act(self, obs: np.ndarray, info: dict = None) -> int:
        """
        Select action based on queue state.

        Args:
            obs: Observation array (9-dim for TrafficEnv)
                 - obs[0]: queue (normalized by MAX_QUEUE=20)
                 - obs[-1]: current_phase (0 or 1)
            info: Optional info dict with unnormalized metrics
                 - info["queue_count"]: raw queue count (vehicles)

        Returns:
            action: 0 (green) or 1 (red)
        """
        # Extract queue count
        if info and "queue_count" in info:
            # Use unnormalized queue from info dict (more accurate)
            queue = info["queue_count"]
        else:
            # Denormalize from observation (obs[0] * MAX_QUEUE)
            queue = obs[0] * 20.0

        # Current phase from observation
        current_phase_obs = int(obs[-1]) if len(obs) > 0 else self.current_phase

        # Decision logic: should we be green or red?
        if queue > self.queue_threshold:
            desired_phase = 0  # Green (serve queue)
        else:
            desired_phase = 1  # Red (no queue, yield to other directions)

        # Enforce minimum green time
        if desired_phase != self.current_phase:
            # Want to change phase
            if self.time_in_phase >= self.min_green_time:
                # Minimum time met, allow change
                self.current_phase = desired_phase
                self.time_in_phase = 0.0
            # else: minimum time not met, stay in current phase
        else:
            # No change desired, continue current phase
            pass

        # Update timer
        self.time_in_phase += self.dt

        return self.current_phase

    def reset(self):
        """Reset policy state for new episode."""
        self.current_phase = 0
        self.time_in_phase = 0.0


# Unit tests
if __name__ == "__main__":
    print("="*80)
    print("MAX-QUEUE POLICY UNIT TESTS")
    print("="*80)

    policy = MaxQueuePolicy(min_green_time=10.0, queue_threshold=0.5, dt=2.0)

    # Test 1: Queue builds up, should stay green
    print("\n[Test 1] Queue builds up -> should stay green")
    policy.reset()
    for step in range(6):  # 12 seconds
        # Queue = 2 vehicles (normalized: 2/20 = 0.1)
        obs = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.5, 0.0])
        action = policy.act(obs)
        print(f"  Step {step+1} (t={policy.time_in_phase:.1f}s): action={action}, phase={policy.current_phase}")
        assert action == 0, f"Expected green (0), got {action}"

    print("  [PASS] Stayed green with queue")

    # Test 2: Queue clears, but min green not met -> stay green
    print("\n[Test 2] Queue clears early -> should stay green (min not met)")
    policy.reset()
    for step in range(3):  # 6 seconds (< 10s min)
        obs_empty = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.3, 0.0])
        action = policy.act(obs_empty)
        print(f"  Step {step+1} (t={policy.time_in_phase:.1f}s): action={action}, phase={policy.current_phase}")
        assert action == 0, f"Expected green (0), got {action}"

    print("  [PASS] Stayed green until min time")

    # Test 3: Queue clears, min green met -> switch to red
    print("\n[Test 3] Queue clears, min green met -> should switch to red")
    policy.reset()
    # Advance to 10s
    for _ in range(5):
        obs = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.5, 0.0])
        policy.act(obs)

    # Now queue clears
    obs_empty = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.3, 0.0])
    action = policy.act(obs_empty)
    print(f"  After min green (t={policy.time_in_phase:.1f}s): action={action}, phase={policy.current_phase}")
    assert action == 1, f"Expected red (1), got {action}"

    print("  [PASS] Switched to red when queue cleared")

    # Test 4: Queue builds again -> switch back to green
    print("\n[Test 4] Queue builds again -> should switch back to green")
    # Policy is now red, advance 10s
    for _ in range(5):
        obs_empty = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.3, 1.0])
        policy.act(obs_empty)

    # Queue builds
    obs_queue = np.array([0.15, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.5, 1.0])
    action = policy.act(obs_queue)
    print(f"  After min red (t={policy.time_in_phase:.1f}s): action={action}, phase={policy.current_phase}")
    assert action == 0, f"Expected green (0), got {action}"

    print("  [PASS] Switched back to green when queue built")

    # Test 5: Using info dict for queue
    print("\n[Test 5] Queue from info dict -> should work")
    policy.reset()
    obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # All zeros
    info = {"queue_count": 3.0}  # But info has queue
    action = policy.act(obs, info)
    print(f"  obs queue=0, info queue=3.0: action={action}")
    assert action == 0, f"Expected green (0) based on info, got {action}"

    print("  [PASS] Used queue from info dict")

    print("\n" + "="*80)
    print("[SUCCESS] ALL TESTS PASSED")
    print("="*80)
