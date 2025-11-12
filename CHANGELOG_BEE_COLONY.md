# Changelog - Bee Colony Honeycomb Construction

## 🎯 Major Changes

This repository has been transformed from a fighter jet air combat simulation to a **bee colony honeycomb construction simulation** using Hierarchical Cooperative Multi-Agent Reinforcement Learning (MARL).

---

## ✨ New Features

### 1. **Grid-Based Bee Simulator**
- **Location**: `warsim/simulator/bee.py`, `warsim/simulator/bee_simulator.py`
- 500x500 grid world
- 32-directional smooth movement
- Collaborative building mechanics
- Build queue system with exponential speedup (1 bee: 256 ticks → 8 bees: 1 tick)
- Memory system (decaying visited areas)

### 2. **Grid Utilities**
- **Location**: `warsim/utils/grid_utils.py`
- Flood fill algorithm for enclosed area calculation
- Neighbor detection (4-way and 8-way)
- Direction vector conversion
- Observation window extraction

### 3. **Multi-Agent Environments**
- **Low-Level Environment**: `envs/env_bee_lowlevel.py`
  - Each step: movement + building decisions
  - 8x8 local observation window (4 channels)
  - MultiDiscrete action space [32, 9]
  - CTDE-compatible (centralized critic)

- **High-Level Environment**: `envs/env_bee_highlevel.py`
  - Task selection every 10-20 steps
  - 16x16 global observation (3 channels)
  - Discrete(4) action space for task selection
  - Executes low-level policy automatically

### 4. **Neural Network Models**
- **Location**: `models/ac_models_bee.py`
- CNN-based architecture for spatial observations
- Separate models for low-level and high-level policies
- Centralized critic for CTDE training

### 5. **Training Scripts**
- **Low-Level**: `train_bee_lowlevel.py`
  - Trains movement and building actions
  - Curriculum learning support (5 levels)
  - Shared policy across all bees
  - PPO with centralized critic

- **High-Level**: `train_bee_highlevel.py`
  - Trains task selection
  - Uses frozen low-level policy
  - Temporal abstraction (10-20 substeps)

### 6. **Configuration System**
- **Location**: `config.py` (new class: `BeeColonyConfig`)
- Separate config for bee colony simulation
- Auto-restore from previous levels
- Comprehensive hyperparameter management

### 7. **Visualization**
- **Location**: `warsim/scenplotter/bee_plotter.py`
- Grid-based matplotlib rendering
- Real-time metrics display
- Color-coded visualization:
  - Blue bees: idle/moving
  - Red bees: building
  - Black: completed walls
  - Orange: walls in progress
  - Green: enclosed areas

### 8. **Testing Framework**
- **Location**: `test_bee_colony.py`
- Comprehensive system tests
- Simulator, environment, and visualization tests
- Enclosed area calculation validation

---

## 🔄 Key Differences from Original

| Aspect | Original (Air Combat) | New (Bee Colony) |
|--------|----------------------|------------------|
| **World** | Continuous 2D space with geodesics | 500x500 discrete grid |
| **Agents** | Fighter jets (heterogeneous) | Bees (homogeneous) |
| **Movement** | Complex flight physics | Simple grid movement (32 directions) |
| **Actions** | Cannon/missile firing | Wall building (8 neighbors) |
| **Cooperation** | Combat tactics | Collaborative building |
| **Reward** | Kill-based | Enclosed area + coordination |
| **Observation** | Tactical angles, distances | Grid-based (8x8 window) |
| **Hierarchy** | Fight/Escape modes | Explore/Build tasks |

---

## 📊 Reward System

### Cooperative Rewards
1. **Enclosed Area Growth**: `+0.5 per new enclosed cell / num_bees`
2. **Adjacent Wall Bonus**: `+0.1` when building next to existing wall
3. **Collaboration Bonus**: `+0.05` per bee when multiple bees build together

### Penalties
1. **Redundant Building**: `-0.5` for building on completed walls

### Collaborative Mechanics
- Build time formula: `256 / (2^(num_builders - 1))`
- 1 bee: 256 ticks
- 2 bees: 128 ticks
- 3 bees: 64 ticks
- 4 bees: 32 ticks
- 5 bees: 16 ticks
- 6 bees: 8 ticks
- 7 bees: 4 ticks
- 8+ bees: 2 ticks

---

## 🏗️ Architecture Highlights

### Hierarchical Policy Structure
```
High-Level (every 10-20 steps):
  ├─ Task 0: Explore freely
  ├─ Task 1: Build zone Northwest
  ├─ Task 2: Build zone Northeast
  └─ Task 3: Build zone South

Low-Level (every step):
  ├─ Direction selection (32 options)
  └─ Build action (9 options: 0=none, 1-8=neighbors)
```

### CTDE (Centralized Training, Decentralized Execution)
- **Actor**: Each bee uses only local observation (8x8 window)
- **Critic**: Sees all bees' observations (centralized)
- **Benefit**: Learns coordination while maintaining scalability

---

## 📦 New Files

### Core Implementation
- `warsim/simulator/bee.py`
- `warsim/simulator/bee_simulator.py`
- `warsim/utils/grid_utils.py`
- `warsim/scenplotter/bee_plotter.py`

### Environments
- `envs/env_bee_lowlevel.py`
- `envs/env_bee_highlevel.py`

### Models
- `models/ac_models_bee.py`

### Training & Testing
- `train_bee_lowlevel.py`
- `train_bee_highlevel.py`
- `test_bee_colony.py`

### Documentation
- `BEE_COLONY_README.md`
- `CHANGELOG_BEE_COLONY.md` (this file)
- `requirements_bee_colony.txt`

---

## 🔧 Modified Files

### Configuration
- `config.py`: Added `BeeColonyConfig` class

### No modifications to original files
- All original air combat code remains unchanged
- Bee colony implementation is completely separate

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements_bee_colony.txt
```

### 2. Run Tests
```bash
python test_bee_colony.py
```

### 3. Train Low-Level Policy
```bash
python train_bee_lowlevel.py --level 1 --epochs 2000
```

### 4. Train High-Level Policy
```bash
python train_bee_highlevel.py --epochs 1000
```

---

## 🎓 Learning Objectives

The bee colony agents should learn to:
1. **Explore** the environment efficiently
2. **Coordinate** building locations
3. **Maximize enclosed areas** (hexagonal patterns emerge naturally)
4. **Collaborate** on building tasks for speedup
5. **Avoid redundancy** (don't rebuild completed walls)

---

## 📈 Expected Behavior

### Early Training
- Random movement
- Scattered wall building
- Low coordination
- Small enclosed areas

### Mid Training
- More purposeful movement
- Some clustering of walls
- Beginning of coordination
- Medium enclosed areas

### Late Training
- Strategic movement toward construction sites
- High collaboration (multiple bees on same wall)
- Connected wall structures
- Large enclosed areas
- **Emergent hexagonal patterns** (optimal for area/perimeter ratio)

---

## 🐛 Known Issues & Future Work

### Current Limitations
1. No curriculum for high-level policy yet
2. Fixed task zones (could be learned)
3. No communication channel between bees
4. Build queue visualization could be improved

### Future Enhancements
1. Add communication mechanism
2. Dynamic task discovery (learn optimal zones)
3. Multi-objective optimization (area + perimeter)
4. Adaptive build time based on resource availability
5. Integrate with real-world swarm robotics

---

## 📚 Technical Details

### PPO Hyperparameters
```python
lr = 5e-5
gamma = 0.99
lambda_ = 0.95
clip_param = 0.2
entropy_coeff = 0.01
num_sgd_iter = 10
```

### CNN Architecture
```
Low-Level Actor:
  Conv2D(4 → 16, k=3)
  Conv2D(16 → 32, k=3)
  Conv2D(32 → 64, k=3, s=2)
  Flatten → FC(1028 → 256 → 256)
  Output: 32 + 9 logits

Centralized Critic:
  FC(7168 → 512 → 256 → 1)
```

---

## 🙏 Acknowledgments

This implementation is built upon the excellent HHMARL 2D framework originally designed for air combat simulation. The hierarchical multi-agent structure and training pipeline have been adapted for cooperative bee colony behavior.

---

**Version**: 1.0.0
**Date**: 2025-01-12
**Status**: ✅ Fully Implemented and Tested
