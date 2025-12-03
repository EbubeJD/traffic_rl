# Traffic RL - MVP Quick Start Guide

This guide walks you through setting up and running the MVP (Minimum Viable Product) for RL-based traffic signal control in CARLA.

## Prerequisites

1. **CARLA Server**: Running on `localhost:2000` with Town10HD_Opt map
2. **Python Packages**:
   ```bash
   pip install gymnasium stable-baselines3 torch numpy
   ```
3. **ROI Annotations**: Ensure ROI annotations exist for GROUP_INDEX=0 traffic lights in `outputs/Town10HD_Opt/tl_*/roi.json`

## Step-by-Step Setup

### Step 1: Identify Traffic Light Configuration

First, run the helper script to identify stable IDs for your traffic lights:

```bash
python tools/print_tl_info.py
```

**Expected Output:**
- List of traffic lights in GROUP_INDEX=0
- Stable IDs (e.g., `road42_lane5_s1234`)
- Location and rotation information
- A phase configuration template

**Example Output:**
```
Traffic Light #1
  Stable ID:  road42_lane5_s1234
  Location:   x=123.45, y=67.89, z=0.50
  Rotation:   yaw=90.00
  ...

PHASE CONFIGURATION TEMPLATE
...
```

### Step 2: Configure Phases

1. Look at the traffic lights in CARLA to understand which approaches they control
2. Use the **yaw angle** to infer direction:
   - `0°` = North
   - `90°` = East
   - `180°` = South
   - `270°` = West

3. Edit `traffic_rl/rl/config.py` and fill in `INTERSECTION_0_PHASES`:

```python
INTERSECTION_0_PHASES = {
    "phase_0": {  # Example: North-South Green
        "road42_lane5_s1234": carla.TrafficLightState.Green,  # North
        "road42_lane6_s1235": carla.TrafficLightState.Green,  # South
        "road43_lane2_s2341": carla.TrafficLightState.Red,    # East
        "road43_lane3_s2342": carla.TrafficLightState.Red,    # West
    },
    "phase_1": {  # Example: East-West Green
        "road42_lane5_s1234": carla.TrafficLightState.Red,
        "road42_lane6_s1235": carla.TrafficLightState.Red,
        "road43_lane2_s2341": carla.TrafficLightState.Green,
        "road43_lane3_s2342": carla.TrafficLightState.Green,
    },
}
```

**Important:** Make sure phase_0 and phase_1 are complementary (lights green in one phase should be red in the other).

### Step 3: Test with Random Agent

Test the environment with a random agent to ensure everything works:

```bash
python traffic_rl/rl/test_random_agent.py --episodes 10
```

**Expected Behavior:**
- ✓ Environment creates successfully
- ✓ Passes `gymnasium.utils.env_checker.check_env()`
- ✓ Random agent runs for 10 episodes without crashing
- ✓ Rewards are negative (more negative = worse congestion)
- ✓ Metrics show realistic traffic patterns

**Typical Output:**
```
Episode 1/10
  Step  5: action=0, reward= -15.30, phase=0, queue=4.2, avg_wait=8.5s
  Step 10: action=1, reward= -18.50, phase=1, queue=5.1, avg_wait=10.2s
  ...

RANDOM AGENT TEST SUMMARY
Completed 10/10 episodes
Mean reward: -250.32
✓ Random agent test PASSED
```

### Step 4: Run Quick PPO Training

Train PPO for 50k-100k timesteps to verify learning:

```bash
python traffic_rl/rl/train_ppo.py --timesteps 50000
```

**What to Monitor:**
- Episode rewards should increase over time (become less negative)
- Check TensorBoard logs: `tensorboard --logdir=./runs/ppo_tensorboard`
- Checkpoints saved to `./checkpoints/`

**Training Progress:**
```
[ENV] Connected to CARLA
[ENV] Selected group 0 with 4 traffic light(s)
-----------------------------------
| rollout/           |            |
|    ep_len_mean     | 36         |
|    ep_rew_mean     | -245.3     |
| time/              |            |
|    total_timesteps | 2048       |
-----------------------------------
...
```

**Success Criteria:**
- PPO completes training without crashes
- Episode reward improves by >10% over random baseline
- Policy shows coherent behavior (not just random switching)

## MVP Success Checklist

- [ ] `print_tl_info.py` prints traffic light information
- [ ] `INTERSECTION_0_PHASES` configured in `config.py`
- [ ] Random agent test passes (10 episodes complete)
- [ ] Environment passes `check_env()`
- [ ] PPO training runs for 50k steps
- [ ] Episode rewards improve during training
- [ ] Checkpoints saved successfully

## Troubleshooting

### Error: "No ROI for TL {stable_id}"
- **Cause:** Missing ROI annotations for traffic lights
- **Fix:** Run ROI annotation tool (see main project README) or use existing annotations

### Error: "Failed to connect to CARLA"
- **Cause:** CARLA server not running
- **Fix:** Start CARLA server on port 2000: `CarlaUE4.sh -carla-rpc-port=2000`

### Error: "Phase configuration is empty"
- **Cause:** `INTERSECTION_0_PHASES` not filled in
- **Fix:** Complete Step 2 (Configure Phases)

### Warning: "Observer error"
- **Cause:** Camera or detection issues (usually safe to ignore during warm-up)
- **Fix:** Ensure YOLO model exists at path specified in `config.py`

### Training is very slow
- **Cause:** CARLA simulation overhead, single environment
- **Fix:** Normal for MVP. Future: parallelize environments, reduce episode length

### Reward doesn't improve
- **Possible causes:**
  1. Phase configuration is incorrect (lights don't actually control traffic flow)
  2. Traffic too light (increase `num_vehicles` in config)
  3. Need more training steps (try 100k-200k)
  4. Reward weights need tuning

## Next Steps After MVP

Once MVP is working:
1. ✅ Increase training to 500k timesteps
2. ✅ Implement detailed episode logging (`rl_logger.py`)
3. ✅ Add fixed-time baselines for comparison
4. ✅ Create evaluation scripts
5. ✅ Generate comparison plots and metrics
6. ✅ Write full documentation

## File Structure

```
traffic_rl-1/
├── traffic_rl/
│   ├── envs/
│   │   ├── carla_intersection_env.py   # Main Gym environment
│   │   └── phase_controller.py         # Phase management
│   ├── reward/
│   │   └── traffic_reward.py           # Reward function
│   └── rl/
│       ├── config.py                   # Configuration (EDIT THIS!)
│       ├── train_ppo.py                # PPO training script
│       └── test_random_agent.py        # Random agent test
├── tools/
│   └── print_tl_info.py                # Helper to identify stable IDs
└── README_MVP.md                        # This file
```

## Configuration Reference

Key parameters in `traffic_rl/rl/config.py`:

### Environment
- `episode_length_sec`: 180 (3 minutes) - shorter for faster iteration
- `decision_interval_sec`: 5 - agent decides every 5 seconds
- `min_green_time_sec`: 10 - minimum green time before switching
- `num_vehicles`: 40 - fixed traffic density

### Reward Weights
- `alpha`: 1.0 - queue length weight
- `beta`: 0.5 - average wait time weight
- `gamma`: 2.0 - long wait penalty (equity)
- `delta`: 0.1 - phase switch penalty

### PPO Hyperparameters
- `learning_rate`: 3e-4
- `n_steps`: 2048
- `batch_size`: 64
- `net_arch`: [64, 64] - two-layer MLP

## Expected Performance

**Random Agent Baseline:**
- Mean episode reward: ~ -250 to -300
- Mean queue length: ~5-8 vehicles
- Mean wait time: ~10-15 seconds

**PPO After 50k Steps:**
- Mean episode reward: ~ -180 to -220 (20-30% improvement)
- More stable queue lengths
- Reduced long wait times (60s+)

**PPO After 500k Steps (Full Training):**
- Mean episode reward: ~ -150 to -180 (40-50% improvement)
- Significant improvement over fixed-time controllers
- Adaptive behavior based on traffic patterns

---

**Questions or issues?** Check the troubleshooting section or refer to the detailed plan in `.claude/plans/`.
