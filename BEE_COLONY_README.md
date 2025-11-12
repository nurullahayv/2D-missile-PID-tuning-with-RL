# 🐝 Bee Colony Honeycomb Construction Simulation

Bu repository, **Hierarchical Cooperative Multi-Agent Reinforcement Learning (MARL)** kullanarak arı kolonisinin petek inşa etme davranışını simüle etmek için güncellenmiştir.

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Sistem Mimarisi](#sistem-mimarisi)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Konfigürasyon](#konfigürasyon)
- [Dosya Yapısı](#dosya-yapısı)

---

## 🎯 Genel Bakış

### Temel Özellikler

- **Hierarchical MARL**: İki seviyeli politika yapısı
  - **High-Level Policy**: Her arı hangi görev/bölgeye odaklanacağına karar verir
  - **Low-Level Policy**: Hareket ve inşa eylemlerini gerçekleştirir

- **Cooperative Learning**: Arılar işbirliği yaparak petek inşa eder
  - Birlikte inşa edildiğinde süre kısalır (256 tick → 128 → 64 → ... → 1 tick)
  - Bitişik duvarlar için koordinasyon bonusu
  - Kapalı alan (enclosed area) oluşturma için ortak ödül

- **Grid-Based World**: 500x500 grid dünyası
  - 32 yönlü smooth hareket
  - 8x8 local observation window
  - Decaying memory (ziyaret edilen alanlar)

- **Building Mechanics**:
  - Arılar bulundukları pozisyonun 8 komşusuna duvar inşa edebilir
  - Collaborative building: Birden fazla arı aynı yere inşa ederse hızlanır
  - Redundant building penalty: Tamamlanmış duvara tekrar inşa etmek ceza

- **Reward System**:
  - Enclosed area increase: Kapalı alanlar büyüdükçe ödül
  - Adjacent wall bonus: Bitişik duvar inşa etme bonusu
  - Coordination bonus: İşbirlikli inşa bonusu
  - Penalties: Gereksiz/tekrarlı inşa cezası

---

## 🏗️ Sistem Mimarisi

### Hierarchical Policy Structure

```
┌─────────────────────────────────────────┐
│         High-Level Policy (Her arı için)│
│  Görev Seçimi:                          │
│  - Explore freely                       │
│  - Build zone NW                        │
│  - Build zone NE                        │
│  - Build zone S                         │
│  (10-20 step'te bir çalışır)           │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│         Low-Level Policy (Her step)     │
│  Eylemler:                              │
│  - Direction: 32 yön (0-31)             │
│  - Build: 9 seçenek (0=yok, 1-8=komşu) │
└─────────────────────────────────────────┘
```

### Observation Space

#### Low-Level Observation
```python
{
    'grid_obs': (8, 8, 4),  # Local 8x8 window
        # Channel 0: Other bees (1.0=idle, 0.5=building)
        # Channel 1: Walls (1.0=wall exists)
        # Channel 2: Visited areas (decaying memory)
        # Channel 3: Build progress (0 to 1)

    'scalar_obs': [x, y, direction, current_task]  # Normalized
}
```

#### High-Level Observation
```python
{
    'global_obs': (16, 16, 3),  # Global downsampled view
        # Channel 0: Bee density
        # Channel 1: Wall density
        # Channel 2: Build activity

    'scalar_obs': [x, y, direction, prev_task]  # Normalized
}
```

### Neural Network Architecture

#### Low-Level Model
```
Input: 8x8x4 grid + 4 scalar features
  ↓
CNN: Conv2D(4→16) → Conv2D(16→32) → Conv2D(32→64)
  ↓
Flatten + Concat with scalar
  ↓
FC: 1024+4 → 256 → 256
  ↓
Output: 32 (direction) + 9 (build) logits

Centralized Critic:
  Input: All bees' observations (7 × (1024+4))
  ↓
  FC: 7168 → 512 → 256 → 1 (value)
```

#### High-Level Model
```
Input: 16x16x3 global + 4 scalar features
  ↓
CNN: Conv2D(3→16) → Conv2D(16→32) → Conv2D(32→64)
  ↓
Flatten + Concat with scalar
  ↓
FC: 1024+4 → 256 → 128
  ↓
Output: 4 task logits

Centralized Critic: Similar structure
```

---

## 🚀 Kurulum

### 1. Gerekli Paketleri Yükleyin

```bash
pip install -r requirements_bee_colony.txt
```

Veya manuel olarak:

```bash
pip install numpy torch gymnasium ray[rllib] matplotlib tqdm tensorboard
```

### 2. Test Edin

```bash
python test_bee_colony.py
```

Bu test şunları kontrol eder:
- ✓ Simulator çalışıyor mu
- ✓ Environment doğru çalışıyor mu
- ✓ Enclosed area hesaplaması doğru mu
- ✓ Visualization çalışıyor mu

---

## 🎮 Kullanım

### 1. Low-Level Policy Training

İlk olarak low-level policy'yi eğitin (hareket + inşa):

```bash
python train_bee_lowlevel.py --level 1 --epochs 2000
```

**Parametreler:**
- `--level`: Training level (1-5, curriculum learning)
- `--epochs`: Training epoch sayısı
- `--num_bees`: Arı sayısı (default: 7)
- `--grid_size`: Grid boyutu (default: 500)
- `--horizon`: Episode uzunluğu (default: 5000)
- `--batch_size`: PPO batch size (default: 4000)
- `--lr`: Learning rate (default: 5e-5)
- `--gpu`: GPU kullanımı (default: 0)
- `--num_workers`: Parallel worker sayısı (default: 4)

**Checkpoint:**
Eğitim sonucu `results/BeeColony_LowLevel_L1_7bees/checkpoint/` altına kaydedilir.

### 2. High-Level Policy Training

Low-level policy eğitildikten sonra, high-level policy'yi eğitin:

```bash
python train_bee_highlevel.py --epochs 1000
```

High-level policy otomatik olarak en son low-level checkpoint'i yükler.

### 3. Evaluation

Eğitilmiş policy'leri değerlendirin:

```bash
python evaluate_bee_colony.py --checkpoint results/BeeColony_HighLevel_7bees/checkpoint
```

---

## ⚙️ Konfigürasyon

### BeeColonyConfig (config.py)

```python
from config import BeeColonyConfig

# Mode 0: Low-level training
config = BeeColonyConfig(mode=0)

# Mode 1: High-level training
config = BeeColonyConfig(mode=1)

# Mode 2: Evaluation
config = BeeColonyConfig(mode=2)
```

### Önemli Parametreler

#### Environment Parameters
```python
--grid_size 500           # 500x500 grid dünyası
--num_bees 7              # 7 arı
--num_directions 32       # 32 yönlü hareket
--window_size 8           # 8x8 local observation
--movement_speed 1.0      # Hareket hızı
--base_build_ticks 256    # Tek arı için inşa süresi
```

#### Training Parameters
```python
--lr 5e-5                 # Learning rate
--gamma 0.99              # Discount factor
--lambda_ 0.95            # GAE lambda
--batch_size 4000         # PPO batch size
--mini_batch_size 512     # Mini-batch size
--epochs 5000             # Training epochs
```

#### High-Level Parameters
```python
--substeps_min 10         # Min substeps per high-level action
--substeps_max 20         # Max substeps per high-level action
--global_resolution 16    # Global observation resolution
```

---

## 📁 Dosya Yapısı

```
hhmarl_2D-for-bee-colony/
├── warsim/
│   ├── simulator/
│   │   ├── bee.py                    # Arı entity class'ı
│   │   ├── bee_simulator.py          # Ana simulator (build queue, rewards)
│   │   └── ...
│   ├── utils/
│   │   ├── grid_utils.py             # Grid utilities, flood fill
│   │   └── ...
│   └── scenplotter/
│       ├── bee_plotter.py            # Visualization
│       └── ...
├── envs/
│   ├── env_bee_lowlevel.py           # Low-level environment
│   ├── env_bee_highlevel.py          # High-level environment
│   └── ...
├── models/
│   ├── ac_models_bee.py              # Neural network models
│   └── ...
├── config.py                          # BeeColonyConfig class
├── train_bee_lowlevel.py             # Low-level training script
├── train_bee_highlevel.py            # High-level training script
├── test_bee_colony.py                # Test script
├── requirements_bee_colony.txt       # Dependencies
└── BEE_COLONY_README.md              # Bu dosya
```

---

## 🎯 Reward Yapısı Detayları

### 1. Enclosed Area Reward
```python
area_increase = current_area - previous_area
reward = area_increase * 0.5 / num_bees  # Her arıya dağıtılır
```

### 2. Adjacent Wall Bonus
```python
if new_wall_is_adjacent_to_existing_wall:
    reward += 0.1
```

### 3. Collaborative Building Bonus
```python
if multiple_bees_building_same_location:
    for each_bee:
        reward += 0.05
    build_time /= 2  # Her ek arı süreyi yarıya indirir
```

### 4. Penalties
```python
if building_on_completed_wall:
    reward -= 0.5
```

---

## 🎨 Visualization

### Plotter Kullanımı

```python
from warsim.simulator.bee_simulator import BeeSimulator
from warsim.scenplotter.bee_plotter import BeePlotter

sim = BeeSimulator(num_bees=7, grid_size=500)
plotter = BeePlotter(grid_size=500, downsample=5)

# Her step'te
sim.do_tick(actions)
plotter.plot(sim, save_path="frame.png", show=True)
```

### Görsel Öğeler
- 🔵 **Mavi arılar**: Idle/hareket ediyor
- 🔴 **Kırmızı arılar**: İnşa yapıyor
- ⬛ **Siyah kareler**: Tamamlanmış duvarlar
- 🟧 **Turuncu kareler**: İnşa devam ediyor
- 🟩 **Yeşil bölgeler**: Kapalı alanlar (enclosed areas)

---

## 📊 Training Tips

### 1. Curriculum Learning
Low-level training için level'ları sırayla eğitin:
```bash
python train_bee_lowlevel.py --level 1 --epochs 1000
python train_bee_lowlevel.py --level 2 --epochs 1000
python train_bee_lowlevel.py --level 3 --epochs 1000
# ...
```

### 2. Hyperparameter Tuning
- Learning rate çok yüksekse training unstable olur
- Batch size büyütmek genellikle daha stabil training sağlar
- Horizon'u artırmak daha uzun vadeli stratejiler öğretir

### 3. Monitoring
TensorBoard ile training'i izleyin:
```bash
tensorboard --logdir results/
```

### 4. Checkpoint Management
Her 50 epoch'ta checkpoint kaydedilir. İstediğiniz checkpoint'ten devam edebilirsiniz:
```bash
python train_bee_lowlevel.py --restore --restore_path results/.../checkpoint
```

---

## 🔬 Algoritma Detayları

### PPO (Proximal Policy Optimization)
- **Algorithm**: PPO-Clip
- **Policy Network**: Shared weights (all bees use same policy)
- **Critic Network**: Centralized (CTDE - Centralized Training, Decentralized Execution)
- **Advantage Estimation**: GAE (Generalized Advantage Estimation)

### CTDE (Centralized Training, Decentralized Execution)
- **Training**: Critic sees all bees' observations (centralized)
- **Execution**: Each bee only uses its own observation (decentralized)
- **Benefit**: Learns coordination while maintaining decentralized execution

---

## 🐛 Troubleshooting

### Problem: OOM (Out of Memory)
**Çözüm**:
- `--batch_size` ve `--mini_batch_size` azaltın
- `--num_workers` azaltın
- Grid size'ı küçültün

### Problem: Training çok yavaş
**Çözüm**:
- `--grid_size` küçültün (500 → 250)
- `--horizon` kısaltın
- `--num_workers` artırın (eğer CPU yeterliyse)

### Problem: Policy öğrenmiyor
**Çözüm**:
- Learning rate'i ayarlayın
- Reward scaling'i kontrol edin
- Daha uzun süre eğitin
- Simpler görevlerle başlayın

---

## 📚 Referanslar

Bu proje şu çalışmalardan esinlenmiştir:
- QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL
- CommNet: Learning Multiagent Communication
- Feudal Networks for Hierarchical Reinforcement Learning

---

## 📝 Citation

Eğer bu kodu kullanırsanız, lütfen orijinal repository'yi cite edin:
```
@misc{bee_colony_marl,
  title={Bee Colony Honeycomb Construction with Hierarchical MARL},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[username]/hhmarl_2D-for-bee-colony}
}
```

---

## 🤝 Contributing

Pull request'ler memnuniyetle karşılanır! Önemli değişiklikler için lütfen önce bir issue açın.

---

## 📧 İletişim

Sorularınız için issue açabilirsiniz.

---

**İyi eğitimler! 🐝🍯**
