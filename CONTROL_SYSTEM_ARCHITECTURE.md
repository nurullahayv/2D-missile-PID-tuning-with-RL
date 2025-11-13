# Control System Architecture: Missile Guidance with RL-Based Adaptive PID

## 📐 System Overview

This document describes the **closed-loop control system** for missile guidance with **reinforcement learning-based adaptive PID parameter tuning**. The system is designed for intercepting maneuvering targets in a 2D engagement scenario.

---

## 🎯 Control System Components

### 1. **Reference Input (Desired State)**
- **Symbol**: $r(t) = [x_t(t), y_t(t)]$
- **Description**: Target position in 2D space
- **Nature**: Time-varying, non-stationary (target performs evasive maneuvers)
- **Maneuver Types**: Straight, circular, zigzag, evasive

### 2. **Error Signal**
- **Symbol**: $e(t) = r(t) - y(t)$
- **Description**: Position error between target and missile
- **Components**:
  ```
  e_x(t) = x_t(t) - x_m(t)
  e_y(t) = y_t(t) - y_m(t)
  ```
- **Derived Metrics**:
  - **Distance error**: $d(t) = \sqrt{e_x^2 + e_y^2}$
  - **Angle error**: $\theta_{err}(t) = \arctan2(e_y, e_x) - \psi_m(t)$

### 3. **Controller: PID (Proportional-Integral-Derivative)**

#### 3.1 Mathematical Formulation
The PID controller generates a control command $u(t)$ based on the error signal:

$$u(t) = K_p \cdot e(t) + K_i \cdot \int_0^t e(\tau) d\tau + K_d \cdot \frac{de(t)}{dt}$$

Where:
- $K_p$: **Proportional gain** - reacts to current error
- $K_i$: **Integral gain** - eliminates steady-state error
- $K_d$: **Derivative gain** - predicts future error (damping)

#### 3.2 Discrete-Time Implementation
For digital implementation with sampling time $\Delta t$:

$$u[k] = K_p \cdot e[k] + K_i \cdot \sum_{i=0}^{k} e[i] \cdot \Delta t + K_d \cdot \frac{e[k] - e[k-1]}{\Delta t}$$

#### 3.3 Heading Control
The PID controller outputs a **heading correction** $\Delta \psi$ which is applied to adjust the missile's velocity vector:

```
u(t) = Δψ(t)         # Heading correction (radians)
ψ_desired(t) = ψ_current(t) + u(t)
```

### 4. **Actuator: Control Allocation**

Converts heading correction to acceleration commands with saturation:

$$
\begin{align}
a_x(t) &= \min(a_{max}, |a(t)|) \cdot \cos(\psi_{desired}) \\
a_y(t) &= \min(a_{max}, |a(t)|) \cdot \sin(\psi_{desired})
\end{align}
$$

**Constraints**:
- Maximum acceleration: $a_{max} = 100 \, \text{m/s}^2$
- Maximum velocity: $v_{max} = 300 \, \text{m/s}$

### 5. **Plant: Missile Dynamics (2D Kinematics)**

The plant represents the **physical system** being controlled:

#### 5.1 State-Space Representation

**State vector**: $\mathbf{x} = [x_m, y_m, v_x, v_y]^T$

**State equations**:
$$
\begin{align}
\dot{x}_m &= v_x \\
\dot{y}_m &= v_y \\
\dot{v}_x &= a_x \\
\dot{v}_y &= a_y
\end{align}
$$

**Matrix form**:
$$
\dot{\mathbf{x}} = \mathbf{A}\mathbf{x} + \mathbf{B}u
$$

Where:
$$
\mathbf{A} = \begin{bmatrix}
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix}, \quad
\mathbf{B} = \begin{bmatrix}
0 & 0 \\
0 & 0 \\
1 & 0 \\
0 & 1
\end{bmatrix}, \quad
u = \begin{bmatrix}
a_x \\
a_y
\end{bmatrix}
$$

#### 5.2 Discrete-Time Update (Euler Integration)

$$
\begin{align}
x_m[k+1] &= x_m[k] + v_x[k] \cdot \Delta t \\
y_m[k+1] &= y_m[k] + v_y[k] \cdot \Delta t \\
v_x[k+1] &= v_x[k] + a_x[k] \cdot \Delta t \\
v_y[k+1] &= v_y[k] + a_y[k] \cdot \Delta t
\end{align}
$$

**Physics frequency**: $f_{physics} = 100 \, \text{Hz}$ → $\Delta t = 0.01 \, \text{s}$

#### 5.3 Additional Dynamics
- **Fuel consumption**: $f[k+1] = f[k] - \alpha \cdot |a[k]| \cdot \Delta t$
- **Heading**: $\psi_m = \arctan2(v_y, v_x)$
- **Speed**: $v = \sqrt{v_x^2 + v_y^2}$

### 6. **Output (System Response)**
- **Symbol**: $y(t) = [x_m(t), y_m(t)]$
- **Description**: Actual missile position
- **Measurement**: Perfect knowledge (no sensor noise in this model)

### 7. **Sensor / Feedback Path**
- **Transfer function**: $H(s) = 1$ (unity feedback)
- **Measurement**: $y(t)$ is fed back to the summing junction
- **Type**: Negative feedback (error = reference - output)

### 8. **Disturbance**
- **Source**: Target evasive maneuvers
- **Effect**: Changes reference input $r(t)$ unpredictably
- **Types**:
  - **Straight**: No disturbance
  - **Circular**: Constant angular velocity
  - **Zigzag**: Periodic heading changes
  - **Evasive**: Reactive maneuvers based on missile position

---

## 🤖 RL Adaptation Layer

### 9.1 Reinforcement Learning Framework

The RL agent observes the **system state** and adaptively adjusts PID gains to optimize performance.

#### Observation Space (14D)
$$
s_t = [x_m, y_m, v_x, v_y, x_t, y_t, v_{tx}, v_{ty}, \theta_{err}, d, K_p, K_i, K_d, f]
$$

Where:
- Position: $x_m, y_m, x_t, y_t$
- Velocity: $v_x, v_y, v_{tx}, v_{ty}$
- Error: $\theta_{err}$ (angle error), $d$ (distance)
- PID: $K_p, K_i, K_d$ (current gains)
- Fuel: $f \in [0, 1]$

#### Action Space (3D Continuous)
$$
a_t = [\Delta K_p, \Delta K_i, \Delta K_d] \in [-1, 1]^3
$$

**Gain update**:
$$
\begin{align}
K_p[k+1] &= \text{clip}(K_p[k] + \Delta K_p \cdot \sigma_p, K_{p,min}, K_{p,max}) \\
K_i[k+1] &= \text{clip}(K_i[k] + \Delta K_i \cdot \sigma_i, K_{i,min}, K_{i,max}) \\
K_d[k+1] &= \text{clip}(K_d[k] + \Delta K_d \cdot \sigma_d, K_{d,min}, K_{d,max})
\end{align}
$$

**Gain ranges**:
- $K_p \in [0.1, 10.0]$
- $K_i \in [0.0, 5.0]$
- $K_d \in [0.0, 5.0]$

#### Reward Function
$$
r_t = -\alpha \cdot d_t + \beta \cdot \mathbb{1}_{\text{hit}} - \gamma \cdot \mathbb{1}_{\text{miss}} - \delta \cdot \frac{|\Delta K|}{K_{max}}
$$

Where:
- $d_t$: Current distance to target (normalized)
- $\mathbb{1}_{\text{hit}}$: Hit bonus (+100)
- $\mathbb{1}_{\text{miss}}$: Miss penalty (-50)
- $|\Delta K|$: Penalty for excessive gain changes (stability)

Weights: $\alpha = 1.0, \beta = 100, \gamma = 50, \delta = 0.1$

### 9.2 RL Algorithms Supported
1. **PPO** (Proximal Policy Optimization) - On-policy, stable
2. **SAC** (Soft Actor-Critic) - Off-policy, sample-efficient
3. **TD3** (Twin Delayed DDPG) - Off-policy, continuous control

### 9.3 Training Objective
Maximize cumulative reward:
$$
J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
$$

Subject to:
- Missile dynamics constraints
- PID gain bounds
- Fuel limitations

---

## 🔄 Closed-Loop System Dynamics

### Transfer Function Analysis

For a unity feedback system with PID controller:

#### Open-Loop Transfer Function
$$
G_{OL}(s) = G_c(s) \cdot G_p(s)
$$

Where:
- $G_c(s) = K_p + \frac{K_i}{s} + K_d s$ (PID controller)
- $G_p(s) = \frac{1}{ms^2}$ (Simplified missile dynamics, mass-normalized)

#### Closed-Loop Transfer Function
$$
G_{CL}(s) = \frac{G_{OL}(s)}{1 + G_{OL}(s)}
$$

#### Characteristic Equation
$$
1 + G_{OL}(s) = 0
$$
$$
ms^3 + K_d s^2 + K_p s + K_i = 0
$$

This is a **3rd-order system**. Stability requires all poles in left-half plane (Routh-Hurwitz criterion).

### Performance Metrics

1. **Settling time** ($t_s$): Time to reach 2% of final value
2. **Overshoot** ($M_p$): Maximum percentage overshoot
3. **Steady-state error** ($e_{ss}$): Residual error at $t \to \infty$
4. **Rise time** ($t_r$): Time to reach 90% of final value

**Goal**: RL tunes $K_p, K_i, K_d$ to minimize $t_s$ while avoiding excessive $M_p$.

---

## 📊 System Specifications

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| **Missile** | | | |
| Max velocity | $v_{max}$ | 300 | m/s |
| Max acceleration | $a_{max}$ | 100 | m/s² |
| Initial speed | $v_0$ | 250 | m/s |
| Initial fuel | $f_0$ | 1.0 | - |
| **Target** | | | |
| Speed | $v_t$ | 150 | m/s |
| Maneuver rate | - | Variable | - |
| **Environment** | | | |
| Map size | $L$ | 10,000 | m |
| Hit radius | $r_h$ | 50 | m |
| Physics frequency | $f_{phy}$ | 100 | Hz |
| Render frequency | $f_{ren}$ | 60 | FPS |
| **PID Ranges** | | | |
| $K_p$ range | - | 0.1 - 10.0 | - |
| $K_i$ range | - | 0.0 - 5.0 | - |
| $K_d$ range | - | 0.0 - 5.0 | - |

---

## 🧪 Experimental Modes

### Mode 1: Basic PID (Fixed Gains)
- Controller: Fixed $K_p, K_i, K_d$ from config
- Use case: Baseline performance, PID tuning study
- Command: `python demo_basic_pid.py --mode realtime`

### Mode 2: RL Adaptive PID
- Controller: RL agent adjusts gains in real-time
- Use case: Evaluating trained model performance
- Command: `python evaluate.py --model_path MODEL.zip --render`

### Mode 3: Hybrid Simulation
Three sub-modes:
- **FAST**: No rendering, 100x+ real-time speed
- **REALTIME**: 100 Hz physics, 60 FPS smooth rendering
- **REPLAY**: Record and playback at variable speed

---

## 📈 System Identification

### Empirical Results (Basic PID)

| Target Type | $K_p$ | $K_i$ | $K_d$ | Hit Rate | Avg. Time |
|-------------|-------|-------|-------|----------|-----------|
| Straight    | 2.0   | 0.1   | 0.5   | 95%      | 45s       |
| Circular    | 3.2   | 0.15  | 0.8   | 85%      | 78s       |
| Zigzag      | 3.2   | 0.15  | 0.8   | 75%      | 92s       |
| Evasive     | 3.2   | 0.15  | 0.8   | 60%      | 120s      |

**Observation**: Single fixed gains cannot handle all scenarios optimally → Need for adaptive control.

### RL Adaptive Performance (Expected)
With trained RL policy:
- **Hit rate improvement**: +10-20% over fixed PID
- **Time reduction**: 15-30% faster intercepts
- **Fuel efficiency**: Better gain modulation reduces unnecessary control effort

---

## 🎓 Control Theory Foundations

### Why PID?
1. **Simplicity**: Easy to understand and implement
2. **Robustness**: Works well for many systems without precise model
3. **Industry standard**: Widely used in aerospace guidance

### Why RL Adaptation?
1. **Non-stationary dynamics**: Target maneuvers change system behavior
2. **Optimal gain scheduling**: RL learns when to use which gains
3. **Data-driven**: No need for analytical gain scheduling rules
4. **Generalization**: Trained on diverse scenarios

### Comparison to Classical Adaptive Control
| Approach | Model Required | Online Learning | Convergence Guarantee |
|----------|----------------|-----------------|----------------------|
| MRAC (Model Reference) | Yes | Yes | Yes (under conditions) |
| Gain Scheduling | Yes | No | Design-dependent |
| **RL Adaptive PID** | No | Yes (offline) | Empirical |

---

## 🔬 Future Work

1. **Extend to 3D**: Altitude control, pitch/yaw dynamics
2. **Multiple missiles**: Formation control, cooperative guidance
3. **Sensor noise**: Add measurement uncertainty, state estimation (Kalman filter)
4. **Wind disturbances**: Environmental effects on plant dynamics
5. **Model-Based RL**: Incorporate learned dynamics model for sample efficiency
6. **Hierarchical RL**: High-level strategy + low-level PID tuning

---

## 📚 References

### Control Systems
1. Franklin, G. F., Powell, J. D., & Emami-Naeini, A. (2019). *Feedback Control of Dynamic Systems* (8th ed.). Pearson.
2. Ogata, K. (2010). *Modern Control Engineering* (5th ed.). Prentice Hall.

### Reinforcement Learning
3. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
4. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*.

### Missile Guidance
5. Zarchan, P. (2012). *Tactical and Strategic Missile Guidance* (6th ed.). AIAA.
6. Siouris, G. M. (2004). *Missile Guidance and Control Systems*. Springer.

---

## 📝 Mathematical Notation Summary

| Symbol | Description |
|--------|-------------|
| $r(t)$ | Reference input (target position) |
| $y(t)$ | System output (missile position) |
| $e(t)$ | Error signal |
| $u(t)$ | Control signal (heading correction) |
| $K_p, K_i, K_d$ | PID gains |
| $s_t$ | RL state observation |
| $a_t$ | RL action |
| $r_t$ | RL reward |
| $\pi_\theta$ | RL policy |
| $\mathbf{x}$ | State vector |
| $G(s)$ | Transfer function |

---

## ✅ Key Contributions

This project demonstrates:

1. **Integration of classical control and modern RL**: Combining PID control with deep RL for adaptive parameter tuning
2. **Real-time simulation framework**: Decoupled physics/rendering for smooth visualization
3. **Practical missile guidance system**: Realistic dynamics, constraints, and target behaviors
4. **Reproducible research**: Open-source implementation with detailed documentation

---

**For academic citation, code usage, or questions, please refer to the repository README.**
