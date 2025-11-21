# Meta-Episodic PID Tuning Implementation

## Overview

This implementation transforms the episodic PID tuning approach into a **Meta-Episodic MDP** with true sequential decision making. This addresses the fundamental RL issues with the previous one-shot episodic approach.

## Problem with Previous Approach

The original episodic approach had several RL theoretical issues:

1. **Not a True MDP**: One action → Done = Contextual Bandit (not MDP)
2. **Sample Inefficiency**: 1 sample per episode vs 500 in step-level
3. **Meaningless Initial Observation**: Zeros (no trajectory yet)
4. **No Sequential Dependency**: Agent can't learn from previous attempts

## Meta-Episodic MDP Solution

### Key Concept

Instead of:
- **Old**: 1 episode = 1 action → Done

We now have:
- **New**: 1 meta-episode = N episodes = N actions
- Each step = one full simulation with fixed PID
- Agent adapts PID based on history from previous episodes

### Architecture

```
Meta-Episode (10 episodes total)
├── Episode 1: Scenario 1 → Select PID₁ → Simulate → Observe (reward₁, hit₁)
├── Episode 2: Scenario 2 → Select PID₂ based on (PID₁, reward₁, hit₁) → Simulate → Observe (reward₂, hit₂)
├── Episode 3: Scenario 3 → Select PID₃ based on history[Episode 1-2] → Simulate
...
└── Episode 10: Scenario 10 → Select PID₁₀ based on history[Episode 5-9] → Done
```

### State Space (33D)

**Context (8D)**: Current scenario parameters
```python
[
    missile_x / map_size,           # Normalized [0, 1]
    missile_y / map_size,
    missile_vx / missile_speed,
    target_x / map_size,
    target_y / map_size,
    target_heading / (2π),
    maneuver_id / 3.0,              # 0-3 → [0, 1]
    difficulty                      # Initial distance / map_size
]
```

**History (25D)**: Last 5 episodes' summary
```python
# For each of 5 episodes: [Kp, Ki, Kd, reward, hit]
[
    Kp₁/10000, Ki₁/50, Kd₁/50, reward₁/100, hit₁,  # Episode t-5
    Kp₂/10000, Ki₂/50, Kd₂/50, reward₂/100, hit₂,  # Episode t-4
    Kp₃/10000, Ki₃/50, Kd₃/50, reward₃/100, hit₃,  # Episode t-3
    Kp₄/10000, Ki₄/50, Kd₄/50, reward₄/100, hit₄,  # Episode t-2
    Kp₅/10000, Ki₅/50, Kd₅/50, reward₅/100, hit₅   # Episode t-1
]
```

**Note**: We use **summary statistics**, NOT full trajectories:
- Full trajectory = 500 steps × 12 features × 5 episodes = 30,000D ❌
- Summary = (PID, reward, hit) × 5 episodes = 25D ✓

### Action Space (3D)

```python
[
    log_Kp ∈ [2.0, 4.0],  # Log scale: 10² to 10⁴ = 100 to 10,000
    Ki ∈ [0, 50],          # Linear scale
    Kd ∈ [0, 50]           # Linear scale
]
```

**Discretization**: Actions are discretized for interpretability
- Kp → nearest 100 (e.g., 1585 → 1600)
- Ki, Kd → nearest 5 (e.g., 7.3 → 5)

### Reward Function

Same as episodic approach (per-episode reward):
- **Hit**: +100 + time_bonus
- **Miss**: -50 - distance_penalty
- **Trajectory Quality**: -avg_distance/1000 - smoothness_penalty
- **Closing Velocity**: +avg_closing_velocity/1000

### Sequential Decision Making

The key innovation is **temporal dependency through history**:

1. **Episode 1**: Agent sees context, no history → Explores
2. **Episode 2**: Agent sees new context + (PID₁, reward₁, hit₁) → Adapts
3. **Episode 3**: Agent sees new context + history[1-2] → Further refines
4. ...
5. **Episode 10**: Agent sees new context + history[5-9] → Done

This creates a **true MDP** where:
- State depends on history
- Action affects future states (via history buffer)
- Sequential credit assignment

## Files Created

### 1. `src/meta_episodic_pid_env.py`

The core environment implementing Meta-Episodic MDP:

**Key Methods**:
- `reset()`: Start new meta-episode (10 episodes)
- `step(action)`: Run one episode with PID, update history
- `_sample_scenario()`: Generate random initial conditions
- `_get_observation()`: Construct (context + history) observation
- `_calculate_reward()`: Same episodic reward calculation

### 2. `train_meta_pid.py`

Training script for Meta-Episodic approach:

**Features**:
- RecurrentPPO (LSTM for sequence learning) - **Recommended**
- PPO (MLP baseline)
- SAC (off-policy alternative)
- SubprocVecEnv for parallel training (4-8 envs)
- Proper callbacks (checkpoints, evaluation)

**Usage**:
```bash
# Recommended: RecurrentPPO with LSTM
python train_meta_pid.py \
    --algorithm RecurrentPPO \
    --maneuver circular \
    --timesteps 50000 \
    --n_envs 8 \
    --episodes_per_meta 10 \
    --window_size 5

# For comparison: PPO without LSTM
python train_meta_pid.py \
    --algorithm PPO \
    --maneuver circular \
    --timesteps 50000 \
    --n_envs 8
```

**Training Metrics**:
- Total timesteps = total episodes
- Meta-episodes = timesteps / episodes_per_meta
- Example: 50K timesteps with 10 episodes_per_meta = 5K meta-episodes

### 3. `test_meta_env.py`

Quick test script to verify environment works:

```bash
python test_meta_env.py
```

Validates:
- ✓ Environment creation
- ✓ Observation space (33D)
- ✓ Action space (3D)
- ✓ Episode sequence
- ✓ History buffer
- ✓ Sequential dependency

## Training Plan for Study 1

### Goal
Compare **RL Meta-Episodic** vs **Ziegler-Nichols** vs **Genetic Algorithm** for non-adaptive PID tuning.

### Phase 1: Train Meta-Episodic RL ✓ (Implementation Complete)

```bash
# Circular maneuver (primary)
python train_meta_pid.py \
    --algorithm RecurrentPPO \
    --maneuver circular \
    --timesteps 50000 \
    --n_envs 8 \
    --episodes_per_meta 10

# Other maneuvers (for comparison)
for maneuver in straight zigzag evasive; do
    python train_meta_pid.py \
        --algorithm RecurrentPPO \
        --maneuver $maneuver \
        --timesteps 50000 \
        --n_envs 8
done
```

### Phase 2: Run Ziegler-Nichols Baseline ✓ (Already Implemented)

```bash
# Already implemented in scripts/tune_ziegler_nichols.py
python scripts/tune_ziegler_nichols.py --all
```

### Phase 3: Implement Genetic Algorithm (TODO)

Create `scripts/tune_genetic_algorithm.py`:
- Population-based optimization
- Fitness = hit rate + time bonus
- Mutation + crossover operators
- 100 generations, pop size 50

### Phase 4: Comparison

Use existing `scripts/compare_methods.py` to compare:
- Hit rate
- Average hit time
- Trajectory quality (smoothness, overshoot)
- Computational cost

## Advantages of Meta-Episodic Approach

### 1. True Sequential Decision Making
- Not contextual bandit anymore
- Agent learns meta-strategy: "If last PID worked, keep similar"

### 2. Sample Efficiency
- 10 episodes = 10 samples (vs 1 in old episodic)
- Better than step-level (500 samples but noisier)

### 3. Stationary Initial Observation
- Context is meaningful (scenario parameters)
- Not zeros like old episodic

### 4. Exploration-Exploitation Balance
- Agent can explore in early episodes
- Exploit best PID in later episodes within meta-episode

### 5. Better for LSTM
- Sequential structure perfect for LSTM
- LSTM can learn: "High reward → stick with similar PID"

## Comparison with Other Approaches

| Approach | Decision Frequency | Samples/Episode | Sequential | MDP? |
|----------|-------------------|-----------------|------------|------|
| **Step-level Adaptive** | Every 0.01s | 500 | ✓ | ✓ |
| **Old Episodic** | Once | 1 | ✗ | ✗ (Bandit) |
| **Meta-Episodic** ⭐ | 10 times | 10 | ✓ | ✓ |
| **Ziegler-Nichols** | Once | N/A | ✗ | ✗ (Classical) |
| **Genetic Algorithm** | Once | N/A | ✗ | ✗ (Evolutionary) |

## Expected Results

### Meta-Episodic RL
- **Hit Rate**: 85-95% (with RecurrentPPO)
- **Adaptability**: High (learns from history)
- **Computation**: Medium (parallel training)
- **Interpretability**: Medium (PID values are discrete)

### Ziegler-Nichols
- **Hit Rate**: 60-75% (depends on oscillation detection)
- **Adaptability**: None (classical tuning)
- **Computation**: Low (binary search)
- **Interpretability**: High (classical method)

### Genetic Algorithm
- **Hit Rate**: 75-85% (population-based optimization)
- **Adaptability**: Medium (population diversity)
- **Computation**: High (100 gens × 50 pop = 5000 evals)
- **Interpretability**: High (direct PID optimization)

## Next Steps

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test environment**:
   ```bash
   python test_meta_env.py
   ```

3. **Train Meta-Episodic RL**:
   ```bash
   python train_meta_pid.py --algorithm RecurrentPPO --maneuver circular --timesteps 50000
   ```

4. **Monitor training**:
   ```bash
   tensorboard --logdir logs/meta_pid/
   ```

5. **Implement GA baseline** (TODO):
   ```bash
   python scripts/tune_genetic_algorithm.py --maneuver circular
   ```

6. **Compare all methods**:
   ```bash
   python scripts/compare_methods.py --all
   ```

## Technical Notes

### Why RecurrentPPO?
- LSTM handles sequential dependencies naturally
- Perfect for history-based decision making
- Outperforms MLP on sequential tasks

### Why Window Size = 5?
- Balance between memory and computational cost
- 5 episodes ≈ 2500 simulation steps of history
- Sufficient for learning PID adaptation patterns

### Why Episodes Per Meta = 10?
- Enough for sequential learning (not too short)
- Not too long (training stability)
- 10 episodes ≈ 5000 simulation steps per meta-episode

### Why Logarithmic Kp?
- PID control is sensitive to orders of magnitude
- log(Kp) ∈ [2, 4] → Kp ∈ [100, 10000]
- Easier for RL to explore uniformly across scales

## Conclusion

This Meta-Episodic MDP implementation addresses all the theoretical RL issues with the previous episodic approach while maintaining the benefits of fixed (non-adaptive) PID tuning per episode. It creates a true MDP with sequential decision making, making it suitable for comparison with classical methods like Ziegler-Nichols and Genetic Algorithms in an academic study.

---

**Implementation Status**: ✓ Complete and Ready for Training

**Date**: 2025-11-20

**Author**: Meta-Episodic PID Tuning System
