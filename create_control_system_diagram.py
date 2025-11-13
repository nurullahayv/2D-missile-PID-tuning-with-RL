"""
Control System Block Diagram Generator
Creates professional block diagram for missile guidance system with RL-based PID adaptation
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def create_control_system_diagram():
    """
    Generate control system block diagram for missile guidance

    System Architecture:
    - Reference Input: Target position (x_t, y_t)
    - Controller: PID with adaptive gains (Kp, Ki, Kd)
    - Plant: Missile dynamics (2D kinematics)
    - Sensor: Position feedback
    - RL Adaptation Layer: Adjusts PID gains based on system state
    - Disturbance: Target maneuvers
    """

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define colors - Professional academic style
    color_reference = '#2E86AB'  # Blue
    color_controller = '#A23B72'  # Purple
    color_plant = '#F18F01'      # Orange
    color_feedback = '#C73E1D'   # Red
    color_rl = '#06A77D'         # Green
    color_disturbance = '#8B4513' # Brown

    # ========================
    # 1. REFERENCE INPUT (Target)
    # ========================
    ref_box = FancyBboxPatch((0.5, 4.5), 1.5, 1.0,
                              boxstyle="round,pad=0.1",
                              edgecolor=color_reference, facecolor='white',
                              linewidth=2.5)
    ax.add_patch(ref_box)
    ax.text(1.25, 5.0, 'Reference\n$r(t)$\n$(x_t, y_t)$',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # ========================
    # 2. SUMMING JUNCTION (Error calculation)
    # ========================
    sum_circle = Circle((3.0, 5.0), 0.3, edgecolor='black',
                        facecolor='white', linewidth=2.5)
    ax.add_patch(sum_circle)
    ax.text(3.0, 5.0, '+', ha='center', va='center',
            fontsize=16, fontweight='bold')
    ax.text(2.7, 4.5, '−', ha='center', va='center',
            fontsize=16, fontweight='bold')
    ax.text(3.0, 5.6, 'Error\n$e(t)$', ha='center', va='center',
            fontsize=9, style='italic')

    # ========================
    # 3. PID CONTROLLER
    # ========================
    pid_box = FancyBboxPatch((4.0, 4.2), 2.2, 1.6,
                              boxstyle="round,pad=0.1",
                              edgecolor=color_controller, facecolor='#F5E6F0',
                              linewidth=2.5)
    ax.add_patch(pid_box)
    ax.text(5.1, 5.4, 'PID Controller', ha='center', va='center',
            fontsize=12, fontweight='bold', color=color_controller)
    ax.text(5.1, 4.9, r'$u(t) = K_p e(t) + K_i \int e(t)dt + K_d \frac{de(t)}{dt}$',
            ha='center', va='center', fontsize=10, style='italic')
    ax.text(5.1, 4.5, r'Gains: $K_p, K_i, K_d$',
            ha='center', va='center', fontsize=9, color=color_rl)

    # ========================
    # 4. ACTUATOR / CONTROL ALLOCATION
    # ========================
    actuator_box = FancyBboxPatch((6.8, 4.5), 1.4, 1.0,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='black', facecolor='white',
                                   linewidth=2.0)
    ax.add_patch(actuator_box)
    ax.text(7.5, 5.0, 'Actuator\n$a(t)$',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # ========================
    # 5. PLANT (Missile Dynamics)
    # ========================
    plant_box = FancyBboxPatch((8.8, 4.0), 2.5, 2.0,
                                boxstyle="round,pad=0.1",
                                edgecolor=color_plant, facecolor='#FFF4E6',
                                linewidth=2.5)
    ax.add_patch(plant_box)
    ax.text(10.05, 5.6, 'Plant: Missile Dynamics', ha='center', va='center',
            fontsize=12, fontweight='bold', color=color_plant)
    ax.text(10.05, 5.15, r'$\dot{x} = v_x, \quad \dot{y} = v_y$',
            ha='center', va='center', fontsize=10, style='italic')
    ax.text(10.05, 4.75, r'$\dot{v}_x = a_x, \quad \dot{v}_y = a_y$',
            ha='center', va='center', fontsize=10, style='italic')
    ax.text(10.05, 4.3, 'Constraints: $v_{max}, a_{max}$',
            ha='center', va='center', fontsize=9, color='gray')

    # ========================
    # 6. OUTPUT (Missile Position)
    # ========================
    output_box = FancyBboxPatch((12.0, 4.5), 1.5, 1.0,
                                 boxstyle="round,pad=0.1",
                                 edgecolor=color_plant, facecolor='white',
                                 linewidth=2.5)
    ax.add_patch(output_box)
    ax.text(12.75, 5.0, 'Output\n$y(t)$\n$(x_m, y_m)$',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # ========================
    # 7. SENSOR (Feedback)
    # ========================
    sensor_box = FancyBboxPatch((11.5, 2.5), 2.5, 0.8,
                                 boxstyle="round,pad=0.1",
                                 edgecolor=color_feedback, facecolor='white',
                                 linewidth=2.0)
    ax.add_patch(sensor_box)
    ax.text(12.75, 2.9, 'Sensor / Feedback\n$H(s) = 1$',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # ========================
    # 8. RL ADAPTATION LAYER
    # ========================
    rl_box = FancyBboxPatch((4.0, 7.0), 4.3, 1.8,
                             boxstyle="round,pad=0.1",
                             edgecolor=color_rl, facecolor='#E6F5F0',
                             linewidth=3.0, linestyle='--')
    ax.add_patch(rl_box)
    ax.text(6.15, 8.4, 'RL Adaptation Layer (PPO/SAC/TD3)',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color=color_rl)
    ax.text(6.15, 7.9, r'Observation: $s_t = [x_m, y_m, v_x, v_y, x_t, y_t, \theta_{err}, d, K_p, K_i, K_d, ...]$',
            ha='center', va='center', fontsize=9, style='italic')
    ax.text(6.15, 7.5, r'Action: $a_t = [\Delta K_p, \Delta K_i, \Delta K_d]$',
            ha='center', va='center', fontsize=9, style='italic')
    ax.text(6.15, 7.1, 'Reward: $r_t = -d_t + \text{bonus}_{\text{hit}} - \text{penalty}_{\text{fuel}}$',
            ha='center', va='center', fontsize=9, style='italic')

    # ========================
    # 9. DISTURBANCE (Target Maneuvers)
    # ========================
    dist_box = FancyBboxPatch((9.5, 6.5), 1.5, 0.8,
                               boxstyle="round,pad=0.1",
                               edgecolor=color_disturbance, facecolor='white',
                               linewidth=2.0, linestyle=':')
    ax.add_patch(dist_box)
    ax.text(10.25, 6.9, 'Disturbance\n$d(t)$',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=color_disturbance)

    # ========================
    # ARROWS / SIGNAL FLOW
    # ========================

    # Reference to Summing Junction
    arrow1 = FancyArrowPatch((2.0, 5.0), (2.7, 5.0),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=2, color='black')
    ax.add_patch(arrow1)

    # Summing Junction to PID
    arrow2 = FancyArrowPatch((3.3, 5.0), (4.0, 5.0),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=2, color='black')
    ax.add_patch(arrow2)
    ax.text(3.65, 5.3, '$e(t)$', ha='center', va='bottom',
            fontsize=9, style='italic')

    # PID to Actuator
    arrow3 = FancyArrowPatch((6.2, 5.0), (6.8, 5.0),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=2, color='black')
    ax.add_patch(arrow3)
    ax.text(6.5, 5.3, '$u(t)$', ha='center', va='bottom',
            fontsize=9, style='italic')

    # Actuator to Plant
    arrow4 = FancyArrowPatch((8.2, 5.0), (8.8, 5.0),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=2, color='black')
    ax.add_patch(arrow4)
    ax.text(8.5, 5.3, '$a(t)$', ha='center', va='bottom',
            fontsize=9, style='italic')

    # Plant to Output
    arrow5 = FancyArrowPatch((11.3, 5.0), (12.0, 5.0),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=2, color='black')
    ax.add_patch(arrow5)

    # Output to Sensor
    arrow6 = FancyArrowPatch((12.75, 4.5), (12.75, 3.3),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=2, color=color_feedback)
    ax.add_patch(arrow6)

    # Sensor to Summing Junction (Feedback)
    arrow7 = FancyArrowPatch((11.5, 2.9), (3.0, 2.9),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=2, color=color_feedback)
    ax.add_patch(arrow7)
    arrow7b = FancyArrowPatch((3.0, 2.9), (3.0, 4.7),
                              arrowstyle='->', mutation_scale=20,
                              linewidth=2, color=color_feedback)
    ax.add_patch(arrow7b)
    ax.text(7.0, 2.6, 'Feedback: $y(t)$', ha='center', va='top',
            fontsize=9, style='italic', color=color_feedback)

    # RL Layer to PID (Adaptation)
    arrow8 = FancyArrowPatch((5.1, 7.0), (5.1, 5.8),
                             arrowstyle='->', mutation_scale=25,
                             linewidth=3, color=color_rl, linestyle='--')
    ax.add_patch(arrow8)
    ax.text(5.5, 6.4, r'Adapt $K_p, K_i, K_d$', ha='left', va='center',
            fontsize=10, fontweight='bold', color=color_rl)

    # State to RL Layer (Observation)
    arrow9 = FancyArrowPatch((10.05, 6.0), (7.5, 7.0),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=2, color=color_rl, linestyle=':')
    ax.add_patch(arrow9)
    ax.text(9.0, 6.7, 'State $s_t$', ha='center', va='bottom',
            fontsize=9, style='italic', color=color_rl)

    # Disturbance to Plant
    arrow10 = FancyArrowPatch((10.25, 6.5), (10.05, 6.0),
                              arrowstyle='->', mutation_scale=20,
                              linewidth=2, color=color_disturbance,
                              linestyle=':')
    ax.add_patch(arrow10)

    # ========================
    # TITLE AND LABELS
    # ========================
    ax.text(8.0, 9.5, 'Missile Guidance Control System with RL-Based Adaptive PID Tuning',
            ha='center', va='center', fontsize=16, fontweight='bold')

    # Legend
    legend_y = 1.2
    ax.text(0.5, legend_y, 'System Components:', fontsize=10, fontweight='bold')
    ax.text(0.5, legend_y - 0.3, '• Reference: Target position (desired state)',
            fontsize=9)
    ax.text(0.5, legend_y - 0.6, '• Controller: PID with adaptive gains',
            fontsize=9)
    ax.text(0.5, legend_y - 0.9, '• Plant: Missile dynamics (2D kinematics)',
            fontsize=9)

    ax.text(6.0, legend_y, 'Closed-Loop Architecture:', fontsize=10, fontweight='bold')
    ax.text(6.0, legend_y - 0.3, '• Negative feedback control',
            fontsize=9)
    ax.text(6.0, legend_y - 0.6, '• Unity feedback (H(s) = 1)',
            fontsize=9)
    ax.text(6.0, legend_y - 0.9, '• RL-based parameter adaptation',
            fontsize=9)

    ax.text(11.5, legend_y, 'RL Training Objective:', fontsize=10, fontweight='bold')
    ax.text(11.5, legend_y - 0.3, r'Minimize: $J = \int_{0}^{T} (d_t + \text{penalties}) dt$',
            fontsize=9, style='italic')
    ax.text(11.5, legend_y - 0.6, 'Subject to: $v \leq v_{max}, a \leq a_{max}$',
            fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig('control_system_block_diagram.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('control_system_block_diagram.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print("✓ Control system block diagram created:")
    print("  - control_system_block_diagram.png (high resolution)")
    print("  - control_system_block_diagram.pdf (vector format)")

    return fig, ax


if __name__ == "__main__":
    print("Generating control system block diagram...")
    fig, ax = create_control_system_diagram()
    print("\nDone! Diagram saved for academic publication.")
