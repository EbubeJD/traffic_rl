# Traffic RL Environment Diagnostic Summary

**Date:** 2025-12-06
**Tool:** `tools/diagnose_env.py`

## Results

### ✓ Working Components
- CARLA server connection: **SUCCESS**
- TrafficRunner initialization: **SUCCESS**
- Observer/camera spawn: **SUCCESS** (road5_lane-1_s10058)
- ROI detection: **FUNCTIONAL** (detected vehicles at steps 6-10, 76-82)
- Metric collection: **WORKING** (queue, wait, arrivals captured)

### ⚠ Issues Found

#### 1. Low Traffic Density (PRIMARY ISSUE)
**Symptom:** 80%+ of steps show queue=0, wait=0
**Impact:** Training data has 90%+ zeros → no learnable signal for BC

**Evidence from run:**
- Total steps: 100
- Steps with queue > 0: ~8 steps (8%)
- Steps with wait > 0: ~12 steps (12%)
- Most observations: all zeros except time_in_state and current_phase

**Root Cause:** Only 10 autopilot vehicles spawned, sparse arrivals at intersection

#### 2. Minor Config Issue
**Error:** `cannot import name 'YOLO_MODEL_PATH' from 'traffic_rl.config'`
**Impact:** Low (YOLO still works, just can't verify path in diagnostic)

## Recommendations

### Option 1: Increase Traffic (RECOMMENDED)
Modify `traffic_rl/config.py`:
```python
# Change from:
NUM_AUTOPILOT = 10

# To:
NUM_AUTOPILOT = 30  # or higher
```

**Expected improvement:**
- More vehicles → more arrivals → sustained queues
- Queue/wait zeros: 90% → <50%
- Better training data for BC

### Option 2: Run Longer Episodes
Current: 100 steps (~3 minutes simulation time)
Try: 200-300 steps to capture more traffic cycles

### Option 3: Use Traffic Demand File
If CARLA supports scenario files, use a high-demand traffic pattern

---

## Next Steps

### Step 1: Fix Traffic Density
```bash
# Edit config
# Set NUM_AUTOPILOT = 30

# Regenerate data with heuristic
python tools/generate_heuristic_data.py --policy max_queue --episodes 10 --steps 120
```

### Step 2: Verify Improvement
```bash
# Analyze new data
python tools/analyze_transitions.py --transitions outputs/Town10HD_Opt/tl_road5_lane-1_s10058/transitions.csv
```

**Success criteria:**
- Queue/wait zeros: <50% (vs current 90%)
- Action alternation rate: <50% (vs current 99%)
- Feature-action MI: >0.1 (vs current ~0.0)

### Step 3: Retrain BC
```bash
python tools/train_bc.py \
    --transitions outputs/Town10HD_Opt/tl_road5_lane-1_s10058/transitions.csv \
    --epochs 50 \
    --normalize \
    --out bc_policy_v2.pt
```

**Expected:** Validation accuracy >70% (vs current 50%)

---

## Conclusion

**Status:** Environment is **FUNCTIONAL** but underfed with traffic.

The 50% BC accuracy is NOT due to broken ROI/observers, but rather:
1. **Low traffic density** → 90% zero observations
2. **Naive fixed-time controller** → 99% action alternation

**Solution:** Increase `NUM_AUTOPILOT`, regenerate data with queue-aware heuristic, retrain BC.

---

**Tools created:** ✓ All 7 pipeline tools ready to use
**Next:** Increase traffic → regenerate data → retrain → evaluate
