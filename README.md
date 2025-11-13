# 2D Missile PID Tuning with Reinforcement Learning

**Temiz, minimal implementation** - 2D füze güdüm sistemi, RL ile adaptif PID parametre ayarlama.

## 🎯 Amaç

- **Görev**: 2D füze (PID kontrollü) → hareketli hedefi takip et
- **RL Hedefi**: PID parametrelerini (Kp, Ki, Kd) adaptif olarak ayarla
- **Test**: Farklı RL algoritmalarını (PPO, SAC, TD3) karşılaştır

## 📦 Stack

- **Gymnasium**: RL environment
- **Pygame**: Görselleştirme
- **PyTorch**: Neural network backend
- **Stable-Baselines3**: RL algorithms (PPO, SAC, TD3)

## 🏗️ Yapı

```
src/
  missile.py      # PID kontrollü füze
  target.py       # Hareketli hedef (4 manevra tipi)
  environment.py  # Gym environment
  renderer.py     # Pygame görselleştirme
train.py          # RL training
evaluate.py       # Model evaluation
demo.py           # Basit demo (RL yok)
config.yaml       # Konfigürasyon
```

## 🚀 Kurulum

```bash
pip install -r requirements.txt
```

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

### 2. RL Training

```bash
# PPO ile eğit (dairesel hedef)
python train.py --algorithm PPO --maneuver circular --timesteps 1000000

# SAC ile eğit (kaçan hedef)
python train.py --algorithm SAC --maneuver evasive --timesteps 1000000 --n_envs 8

# TD3 ile eğit (zigzag hedef)
python train.py --algorithm TD3 --maneuver zigzag --timesteps 500000
```

**Output**: `models/` klasörüne kaydedilir

### 3. Trained Model Evaluation

```bash
# Görselleştirme ile
python evaluate.py models/PPO_circular_*/best_model/best_model.zip --render --n_episodes 10

# Sadece metrikler
python evaluate.py models/SAC_evasive_*/final_model.zip --n_episodes 20
```

## 📊 Sistem Detayları

### Füze
- **State**: Pozisyon (x, y), Hız (vx, vy)
- **Kontrolör**: PID (heading kontrolü)
- **Kısıtlar**: max_speed=300m/s, max_accel=100m/s²
- **Fizik**: 100 Hz güncelleme (dt=0.01s)

### Hedef
- **Hız**: 150 m/s (füzeden yavaş)
- **Manevralar**:
  - `straight`: Manevra yok
  - `circular`: Sabit dönüş hızı
  - `zigzag`: Periyodik yön değişimleri
  - `evasive`: Füzeye tepkisel kaçış

### RL Environment

**Observation (14D)**:
- Füze: pozisyon, hız, PID gains, fuel
- Hedef: pozisyon, hız
- Relative: mesafe, açı hatası

**Action (3D continuous)**:
- `[Δkp, Δki, Δkd]` ∈ [-1, 1]³

**Reward**:
- -distance (normalize edilmiş)
- +hedefe yaklaşma bonusu
- +100 (vurdu)
- -50 (ıskaladı)

### Desteklenen Algoritmalar
- **PPO**: On-policy, stabil, iyi baseline
- **SAC**: Off-policy, sample-efficient
- **TD3**: Off-policy, deterministic, robust

## 📈 Beklenen Sonuçlar

| Method    | Maneuver  | Hit Rate | Avg Steps |
|-----------|-----------|----------|-----------|
| Sabit PID | Straight  | ~90%     | 120       |
| Sabit PID | Circular  | ~70%     | 180       |
| Sabit PID | Evasive   | ~40%     | 250       |
| RL (PPO)  | Circular  | ~85%     | 150       |
| RL (SAC)  | Evasive   | ~65%     | 200       |

RL ajanları zor manevralarda **+10-20% iyileştirme** göstermeli.

## ⚙️ Konfigürasyon

`config.yaml` dosyasını düzenle:
- Harita boyutu, vuruş yarıçapı
- Füze/hedef hızları
- PID default değerleri ve aralıkları
- Training hyperparameters

## 🎨 Görselleştirme

Pygame renderer gösterir:
- Füze (cyan) ve hedef (red)
- Trajectory'ler (son 100 nokta)
- Vuruş yarıçapı çemberi
- Real-time info: mesafe, PID gains, fuel, hız

Kontroller:
- **ESC** veya **Q**: Çıkış

## 🔧 İleri Kullanım

### Paralel Training

```bash
# Daha fazla paralel environment
python train.py --algorithm PPO --n_envs 8 --timesteps 2000000
```

### Hyperparameter Tuning

`train.py` içinde değiştir:
- Learning rate
- Batch size
- Network architecture

### Custom Maneuvers

`src/target.py` içinde yeni manevra ekle:

```python
elif self.maneuver == 'custom':
    # Kendi manevralarınız
    pass
```

## 🐛 Sorun Giderme

**Yavaş rendering**: `demo.py` veya `evaluate.py` çalıştırırken `--render` kullanma

**Training converge olmuyor**:
- Timesteps artır
- `src/environment.py` içinde reward function ayarla
- Farklı algoritma dene (SAC genelde daha sample-efficient)

**Import errors**: Proje root'undan çalıştırdığınızdan ve `requirements.txt` yüklendiğinden emin olun

## 📚 Kaynaklar

1. **Control Systems**: Franklin et al., "Feedback Control of Dynamic Systems"
2. **RL**: Sutton & Barto, "Reinforcement Learning: An Introduction"
3. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms"
4. **SAC**: Haarnoja et al., "Soft Actor-Critic"
5. **TD3**: Fujimoto et al., "Addressing Function Approximation Error"

---

**Akademik kontrol sistemleri dökümantasyonu için**: `CONTROL_SYSTEM_ARCHITECTURE.md`
