# Neon Visualization & Demo Guide

Projenizde artık **dark digital blue background** ve **gradient neon effects** ile harika görselleştirmeler var! 🎨✨

## 🎯 Features

### Neon Renderer
- ✨ Dark digital blue background (#0a0e27)
- 📊 Grid system (major & minor)
- 🚀 Cyan neon missile trail (gradient fade)
- 🎯 Magenta neon target trail (gradient fade)
- 💫 Glowing entities with direction arrows
- 📈 Real-time info panel with PID gains
- 🎨 Smooth gradient effects on trajectories

### Demo Simulation
- 🎲 Random PID parameters
- 🎯 Multiple target maneuvers
- 📹 Frame capture for video creation
- 🖥️ Live visualization

## 🚀 Quick Start

### 1. Demo Simulation (No Training Needed!)

```bash
# Random PID parameters, circular target
python demo_simulation.py

# Specific target type
python demo_simulation.py --target circular

# Different PID modes
python demo_simulation.py --pid_mode random    # Random PID
python demo_simulation.py --pid_mode default   # Default PID (2.0, 0.1, 0.5)
python demo_simulation.py --pid_mode optimal   # Optimal PID (3.2, 0.15, 0.8)

# All target types
python demo_simulation.py --target straight
python demo_simulation.py --target circular
python demo_simulation.py --target zigzag
python demo_simulation.py --target evasive
```

### 2. Evaluation with Live Rendering

```bash
# Evaluate model with live neon visualization
python evaluate.py \
  --model_path ./models/your_exp/best_model.zip \
  --target_maneuver circular \
  --render
```

**Note:** `--render` flag activates the live neon visualization during evaluation!

## 📹 Creating Videos

### From Demo Simulation

```bash
# Run demo and save frames
python demo_simulation.py --save_video --target circular

# Create video with ffmpeg
ffmpeg -r 30 -i demo_frames/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p \
  -preset slow -crf 18 \
  missile_demo.mp4
```

### Custom Frame Rate

```bash
# Slow motion (15 fps)
ffmpeg -r 15 -i demo_frames/frame_%04d.png output.mp4

# Normal speed (30 fps)
ffmpeg -r 30 -i demo_frames/frame_%04d.png output.mp4

# Fast forward (60 fps)
ffmpeg -r 60 -i demo_frames/frame_%04d.png output.mp4
```

## 🎨 Visualization Details

### Color Scheme

```python
Background:     #0a0e27  (Dark blue)
Grid:           #1a2f5c  (Subtle blue)
Missile Trail:  #00ffff  (Cyan neon)
Target Trail:   #ff00ff  (Magenta neon)
Hit Radius:     #ff0066  (Hot pink)
Text/Labels:    #00ffff  (Cyan)
```

### Effects

1. **Gradient Trails**
   - Fades from 20% to 100% opacity
   - Multiple glow layers for depth
   - Smooth, anti-aliased lines

2. **Glowing Entities**
   - 3-layer glow effect
   - Direction arrows
   - Labeled with team colors

3. **Info Panel**
   - Semi-transparent background
   - Cyan border
   - Monospace font for data
   - Real-time PID gains
   - Fuel indicator

## 📊 Example Outputs

### Demo Simulation
```
==================================
Missile PID Control - Demo Simulation
==================================
Target Maneuver: circular
PID Mode: random
==================================
Random PID: Kp=3.127, Ki=0.845, Kd=1.234

Simulating... (Close window to stop)

✓ HIT! at step 245, distance: 38.50m

==================================
Simulation Complete
==================================
Total Steps: 245
Final Distance: 38.50m
Hit: YES
Fuel Remaining: 75.5%
Final PID Gains: Kp=3.127, Ki=0.845, Kd=1.234
==================================
```

### Evaluation with Rendering
```
==================================
Evaluating Missile PID RL Model
==================================
Model: ./models/exp/best_model.zip
Episodes: 10
Target Maneuver: circular
==================================
Live rendering enabled - displaying neon visualization

Episode 1/10: Reward=215.32, Length=195, Hit=Yes, Final Distance=42.10m
[Live neon visualization window opens]
```

## 🎯 Use Cases

### 1. Research Presentation
```bash
# Create beautiful visualizations for papers/presentations
python demo_simulation.py --target circular --pid_mode optimal --save_video
```

### 2. Debugging PID Behavior
```bash
# Visualize how different PID values affect trajectory
python demo_simulation.py --pid_mode random
python demo_simulation.py --pid_mode default
python demo_simulation.py --pid_mode optimal
```

### 3. Model Evaluation
```bash
# Watch trained model in action
python evaluate.py \
  --model_path ./models/best/best_model.zip \
  --target_maneuver evasive \
  --render \
  --n_episodes 5
```

### 4. Comparing Target Maneuvers
```bash
# See how missile performs against different targets
for maneuver in straight circular zigzag evasive; do
  python demo_simulation.py --target $maneuver --pid_mode optimal --save_video
done
```

## 🛠️ Customization

### Change Colors

Edit `warsim/visualization/neon_renderer.py`:

```python
# In __init__():
self.missile_color = '#00ff00'  # Green instead of cyan
self.target_color = '#ff6600'   # Orange instead of magenta
self.bg_color = '#000000'       # Pure black background
```

### Adjust Visualization Parameters

```python
# Line width for trajectories
linewidth=3.0  # Default, increase for thicker lines

# Entity marker size
size=150  # Default, increase for larger markers

# Glow intensity
alpha=0.8  # Default, increase for brighter glow
```

### Frame Rate for Rendering

Edit in `demo_simulation.py` or `evaluate.py`:

```python
if step % 2 == 0:  # Render every 2 steps (default)
if step % 1 == 0:  # Render every step (slower but smoother)
if step % 5 == 0:  # Render every 5 steps (faster but choppier)
```

## 📈 Performance Tips

### For Smooth Rendering
- Use `step % 2 == 0` (render every 2 steps)
- Lower DPI (dpi=100 instead of 150)
- Reduce window size

### For High-Quality Output
- Use `step % 1 == 0` (render every step)
- Higher DPI (dpi=200 or 300)
- Larger figure size

### For Video Creation
- Always use `--save_video`
- Render every step for smooth video
- Use high DPI (150-200)

## 🎬 Video Examples

### HD Quality
```bash
python demo_simulation.py \
  --target evasive \
  --pid_mode optimal \
  --max_steps 500 \
  --save_video

ffmpeg -r 30 -i demo_frames/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p \
  -preset slow -crf 18 \
  -vf "scale=1920:1080" \
  missile_hd.mp4
```

### 4K Quality
```bash
ffmpeg -r 30 -i demo_frames/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p \
  -preset slow -crf 18 \
  -vf "scale=3840:2160" \
  missile_4k.mp4
```

### GIF for Social Media
```bash
ffmpeg -r 15 -i demo_frames/frame_%04d.png \
  -vf "scale=800:-1:flags=lanczos,fps=15" \
  missile.gif
```

## 🐛 Troubleshooting

### Window doesn't appear
```bash
# Check matplotlib backend
export MPLBACKEND=TkAgg

# Or try Qt backend
export MPLBACKEND=Qt5Agg
```

### Slow rendering
- Reduce render frequency (`step % 5 == 0`)
- Lower DPI (`dpi=75`)
- Close other programs

### Video creation fails
```bash
# Install ffmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Windows:
# Download from https://ffmpeg.org/
```

## 📚 Technical Details

### Neon Renderer Architecture

```python
NeonRenderer
├── setup_figure()          # Dark theme setup
├── draw_gradient_trajectory()  # Gradient neon trails
├── draw_entity()           # Glowing markers with arrows
├── draw_hit_radius()       # Pulsing circle effect
├── draw_info_panel()       # Real-time stats panel
├── render_frame()          # Complete frame rendering
├── save_frame()            # Save to file
└── close()                 # Cleanup
```

### Gradient Effect Implementation

Uses matplotlib's `LineCollection` with varying alpha:

```python
alphas = np.linspace(0.2, 1.0, n_points)  # Fade effect
colors = [mcolors.to_rgba(color, alpha=alpha) for alpha in alphas]
lc = LineCollection(segments, colors=colors, linewidth=linewidth)
```

### Glow Effect

Multiple layers with decreasing width and alpha:

```python
for glow_width in [linewidth * 2, linewidth * 3]:
    glow_alphas = alphas * 0.3
    # Create glow layer...
```

## 🎉 Examples Gallery

After running simulations, you'll have beautiful visualizations like:

1. **Straight Target**
   - Clean, direct pursuit
   - Minimal PID adjustments
   - Quick interception

2. **Circular Target**
   - Curved pursuit path
   - Continuous PID tuning
   - Predictive interception

3. **Zigzag Target**
   - Adaptive tracking
   - Rapid PID changes
   - Challenge for control

4. **Evasive Target**
   - Complex pursuit
   - Aggressive PID tuning
   - Ultimate challenge

## 🚀 Next Steps

1. **Create your first demo:**
   ```bash
   python demo_simulation.py --target circular
   ```

2. **Evaluate with rendering:**
   ```bash
   python evaluate.py --model_path YOUR_MODEL --render
   ```

3. **Make a video:**
   ```bash
   python demo_simulation.py --save_video --target evasive
   ```

4. **Experiment with PID:**
   ```bash
   python demo_simulation.py --pid_mode random
   ```

Enjoy the beautiful neon visualizations! ✨🚀
