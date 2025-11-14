# 2D Missile PID Tuning with Reinforcement Learning

**Episode-level RL** - RL agent observes full simulation trajectory and learns optimal FIXED PID parameters.

## 🎯 Amaç

- **Sistem**: 2D füze (PID kontrollü) → hareketli hedefi takip et
- **RL Görevi**: Optimal sabit PID parametrelerini (Kp, Ki, Kd) bul
- **Yaklaşım**: Episode-level RL - Tüm simülasyon trajectory'si observation
- **Test**: RecurrentPPO (LSTM) ile trajectory'yi öğren

## 📦 Stack

- **Gymnasium**: RL environment
- **Pygame**: Görselleştirme
- **PyTorch**: Neural network backend
- **Stable-Baselines3**: RL algorithms (PPO, SAC)
- **SB3-Contrib**: RecurrentPPO (LSTM policy)

## 🏗️ Yapı

```
src/
  missile.py                    # PID kontrollü füze
  target.py                     # Hareketli hedef (4 manevra tipi)
  episodic_fixed_pid_env.py    # Episode-level RL environment ⭐
  renderer.py                   # Pygame görselleştirme
train_fixed_pid.py              # RL training (RecurrentPPO) ⭐
evaluate.py                     # Model evaluation
demo.py                         # Basit demo (RL yok)
kaggle_training_fixed_pid.ipynb # Kaggle GPU training notebook 🎮
config.yaml                     # Konfigürasyon
```

## 🚀 Kurulum

```bash
pip install -r requirements.txt
```

**Not:** `sb3-contrib` gerekli (RecurrentPPO için)

## 💻 Kullanım

### 1. Demo (RL olmadan, sabit PID)

```bash
# Dairesel manevra yapan hedef
python demo.py --maneuver circular --kp 2.0 --ki 0.1 --kd 0.5

# Kaçan hedef
python demo.py --maneuver evasive --kp 3.0 --ki 0.15 --kd 0.8

# Düz giden hedef (kolay)
python demo.py --maneuver straight --kp 1.5 --ki 0.05 --kd 0.3

# Zigzag yapan hedef
python demo.py --maneuver zigzag --kp 2.5 --ki 0.12 --kd 0.6
```

### 2. RL Training (Episode-level Fixed PID) ⭐

**En pratik yaklaşım**: RL ile optimal **sabit** PID parametrelerini bul

```bash
# RecurrentPPO ile circular hedef için optimal PID bul
python train_fixed_pid.py --algorithm RecurrentPPO --maneuver circular --timesteps 10000

# Evasive hedef için
python train_fixed_pid.py --algorithm RecurrentPPO --maneuver evasive --timesteps 20000

# Standard PPO (LSTM olmadan)
python train_fixed_pid.py --algorithm PPO --maneuver circular --timesteps 10000

# SAC (LSTM olmadan, off-policy)
python train_fixed_pid.py --algorithm SAC --maneuver circular --timesteps 50000
```

**Önemli:**
- 1 timestep = 1 episode (full 500-step simulation)
- RecurrentPPO: LSTM ile trajectory'yi öğrenir
- 10K timesteps = 10K episode = ~2-3 hours

**Çıktı**: Script otomatik olarak optimal PID parametrelerini bulur ve terminale yazdırır:
```
Optimal PID Parameters for 'circular' target:
  Kp = 3.245 ± 0.123
  Ki = 0.187 ± 0.042
  Kd = 0.712 ± 0.089

💡 Use these values in demo.py:
   python demo.py --maneuver circular --kp 3.245 --ki 0.187 --kd 0.712
```

**Avantajlar**:
- ✅ Gerçek füze sistemlerine benzer (sabit PID)
- ✅ Yorumlanabilir sonuçlar (somut PID değerleri)
- ✅ Trajectory observation (tüm simülasyon görülür)
- ✅ Demo'da test edilebilir

### 3. Kaggle GPU Training (Önerilen!)

**En hızlı yol: Kaggle'da ücretsiz GPU ile eğit!**

1. Kaggle'a git: https://www.kaggle.com
2. `kaggle_training_fixed_pid.ipynb` dosyasını upload et
3. Settings → Accelerator → **GPU T4** seç
4. "Run All" - 1-2 saatte model hazır!
5. Optimal PID değerleri notebook'ta gösterilir

**Avantajlar:**
- ✅ Ücretsiz GPU (T4/P100)
- ✅ Kurulum yok, direkt çalışır
- ✅ 1-2 saatte eğitim tamamlanır
- ✅ Optimal PID değerleri otomatik çıkar

### 4. Trained Model Evaluation

```bash
# Görselleştirme ile
python evaluate.py models/recurrentppo_circular_*/best_model.zip --render --n_episodes 10

# Sadece metrikler
python evaluate.py models/sac_evasive_*/final_model.zip --n_episodes 20
```

## 📊 Sistem Detayları

### Füze
- **State**: Pozisyon (x, y), Hız (vx, vy)
- **Kontrolör**: PID (heading kontrolü)
- **Kısıtlar**: max_speed=1000m/s, max_accel=1000m/s²
- **Fizik**: 100 Hz güncelleme (dt=0.01s)

### Hedef
- **Hız**: 1000 m/s
- **Manevralar**:
  - `straight`: Manevra yok
  - `circular`: Sabit dönüş hızı
  - `zigzag`: Periyodik yön değişimleri
  - `evasive`: Füzeye tepkisel kaçış

### Episode-level RL Environment ⭐

**Workflow:**
1. RL agent selects [Kp, Ki, Kd] once
2. Environment runs FULL simulation (500 steps)
3. Trajectory is downsampled (every 10 steps → 50 samples)
4. Observation = trajectory features (600D)
5. Reward = episodic (hit, time, trajectory quality)

**Observation (600D)**:
- Downsampled trajectory: 50 samples × 12 features
- Features per sample: [m_x, m_y, m_vx, m_vy, t_x, t_y, t_vx, t_vy, distance, angle_error, closing_velocity, heading_error]

**Action (3D continuous)**:
- `[Kp, Ki, Kd]` - Direkt PID değerleri
- Kp ∈ [0.1, 10000], Ki ∈ [0.0, 50], Kd ∈ [0.0, 50]
- Episode başında bir kere seçilir, sonra sabit kalır

**Reward (Episodic)**:
```python
reward = 0
if hit:
    reward += 100 + time_bonus
else:
    reward -= 50 + distance_penalty

reward -= avg_distance_penalty
reward -= trajectory_smoothness_penalty
reward += closing_velocity_bonus
```

### Desteklenen Algoritmalar
- **RecurrentPPO**: LSTM policy, trajectory sequence öğrenir ⭐
- **PPO**: On-policy, stabil, iyi baseline
- **SAC**: Off-policy, sample-efficient (ama LSTM yok)

## 📈 Beklenen Sonuçlar

| Algorithm      | Maneuver  | Hit Rate | Avg Time | Training Time |
|----------------|-----------|----------|----------|---------------|
| RecurrentPPO   | Circular  | ~80%     | 200      | 2-3 hours     |
| RecurrentPPO   | Evasive   | ~60%     | 280      | 3-4 hours     |
| PPO (no LSTM)  | Circular  | ~70%     | 220      | 2 hours       |
| SAC (no LSTM)  | Circular  | ~75%     | 210      | 4-5 hours     |

**RecurrentPPO önerilir:** Trajectory sequence'i LSTM ile öğrenir.

## ⚙️ Konfigürasyon

`config.yaml` dosyasını düzenle:
- Harita boyutu, vuruş yarıçapı
- Füze/hedef hızları
- PID aralıkları (wide range: Kp up to 10000!)
- Training hyperparameters

## 🎨 Görselleştirme

Pygame renderer gösterir:
- Füze (cyan) ve hedef (red)
- Trajectory'ler (son 100 nokta)
- Vuruş yarıçapı çemberi
- Real-time info: mesafe, PID gains, hız

Kontroller:
- **ESC** veya **Q**: Çıkış

## 🔧 İleri Kullanım

### Paralel Training

```bash
# Daha fazla paralel environment
python train_fixed_pid.py --algorithm RecurrentPPO --n_envs 8 --timesteps 20000
```

### Hyperparameter Tuning

`train_fixed_pid.py` içinde değiştir:
- Learning rate
- Batch size
- Network architecture
- LSTM hidden size

### Custom Maneuvers

`src/target.py` içinde yeni manevra ekle:

```python
elif self.maneuver == 'custom':
    # Kendi manevralarınız
    pass
```

## 🐛 Sorun Giderme

**Yavaş training**: Normal! 1 episode = 500 simulation step. RecurrentPPO LSTM overhead ekler.

**LSTM memory error**: `lstm_hidden_size` küçült (256 → 128)

**Training converge olmuyor**:
- Timesteps artır (10K → 20K)
- Reward function ayarla (`episodic_fixed_pid_env.py`)
- Farklı algoritma dene (RecurrentPPO → PPO)

**Import errors**:
```bash
pip install sb3-contrib>=2.0.0
```

## 📚 Kaynaklar

1. **Control Systems**: Franklin et al., "Feedback Control of Dynamic Systems"
2. **RL**: Sutton & Barto, "Reinforcement Learning: An Introduction"
3. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms"
4. **SAC**: Haarnoja et al., "Soft Actor-Critic"
5. **LSTM**: Hochreiter & Schmidhuber, "Long Short-Term Memory"

---

## 🆚 Episode-level vs Step-level RL

| Özellik | Episode-level (Bu Repo) | Step-level |
|---------|-------------------------|------------|
| **Observation** | Full trajectory (600D) | Current state (11D) |
| **Action frequency** | Once per episode | Every step |
| **Training samples** | 1 per episode | 500 per episode |
| **Trajectory** | Explicit | Implicit (LSTM hidden) |
| **Training speed** | Slower (1 sample) | Faster (500 samples) |
| **Information** | Full trajectory | Current state only |
| **Best for PID tuning** | ✅ Yes | Maybe |

**Episode-level daha mantıklı çünkü:**
- Tüm trajectory görülür (like real PID tuning!)
- Reward episodic (hit, time, quality)
- More interpretable

---

**Akademik kontrol sistemleri dökümantasyonu için**: `CONTROL_SYSTEM_ARCHITECTURE.md`
