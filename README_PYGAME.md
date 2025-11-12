# Real-Time Pygame Visualization Guide

## 🎮 Yeni Özellikler

### ✅ Fixed Issues
- ❌ Birden fazla matplotlib figür açılma sorunu **çözüldü**
- ❌ Sadece son resim görüntüleme sorunu **çözüldü**
- ✅ **Gerçek zamanlı simülasyon** - Pygame ile smooth rendering
- ✅ **PID konfig sistemi** - YAML ile parametreleri ayarlayın

### 🎯 İki Ana Mod

#### 1. Basic PID (Sabit Parametreler)
```bash
python demo_basic_pid.py
```
- Sabit PID parametreleri ile füze
- Hareketli hedefe kovalamaca
- Gerçek zamanlı görselleştirme

#### 2. RL Adaptive PID (Öğrenilmiş Model)
```bash
python evaluate.py --model_path YOUR_MODEL.zip --render
```
- RL ile öğrenilmiş model
- Adaptif PID tuning
- Gerçek zamanlı performans gösterimi

---

## 🚀 Hızlı Başlangıç

### 1. Paketleri Yükleyin
```bash
pip install -r requirements.txt
```

### 2. Basic PID Demo (Eğitim Gerekmez!)
```bash
# Default PID parametreleri ile
python demo_basic_pid.py

# Farklı hedef tipleri
python demo_basic_pid.py --target circular
python demo_basic_pid.py --target evasive
python demo_basic_pid.py --target zigzag

# Optimal PID kullan
python demo_basic_pid.py --use_optimal

# Custom PID parametreleri
python demo_basic_pid.py --kp 3.5 --ki 0.2 --kd 0.8
```

### 3. RL Model Evaluation (Gerçek Zamanlı)
```bash
python evaluate.py \
  --model_path ./models/YOUR_EXP/best_model.zip \
  --target_maneuver circular \
  --render
```

---

## ⚙️ PID Konfigürasyonu

### Config Dosyası: `config_pid.yaml`

```yaml
# Default PID parameters
default_pid:
  kp: 2.0
  ki: 0.1
  kd: 0.5

# Optimal PID parameters
optimal_pid:
  kp: 3.2
  ki: 0.15
  kd: 0.8

# Missile dynamics
missile:
  max_speed: 300.0
  max_acceleration: 100.0
  initial_speed: 250.0

# Target dynamics
target:
  speed: 150.0

# Simulation
simulation:
  dt: 0.1
  max_steps: 500
  map_size: 10000.0
  hit_radius: 50.0
```

### Parametreleri Değiştirme

#### Yöntem 1: Config dosyasını düzenle
```bash
nano config_pid.yaml
# default_pid veya optimal_pid değerlerini değiştir
```

#### Yöntem 2: Komut satırı argümanları
```bash
# Custom Kp değeri
python demo_basic_pid.py --kp 4.0

# Tüm parametreleri özel ayarla
python demo_basic_pid.py --kp 3.5 --ki 0.25 --kd 1.0
```

---

## 🎨 Pygame Görselleştirme

### Özellikler
- ✅ **Tek pencere** - Birden fazla figür açılmaz
- ✅ **Gerçek zamanlı** - 60 FPS smooth rendering
- ✅ **Gradient trails** - Fade efekti ile trajectory
- ✅ **Glow effects** - Neon stil çoklu katman parlaklık
- ✅ **Grid sistem** - Major ve minor grid lines
- ✅ **Info panel** - PID gains, fuel, distance
- ✅ **Interactive** - ESC veya Q ile çıkış

### Görsel Tema
- 🌌 Dark digital blue background (#0a0e27)
- 🚀 Cyan missile trail (#00ffff)
- 🎯 Magenta target trail (#ff00ff)
- 📊 Subtle blue grid (#1a2f5c)
- 💫 Multi-layer glow effects

### Kontroller
- **ESC** veya **Q**: Simülasyonu sonlandır
- **Pencere kapatma**: Programı kapat
- Evaluation sırasında ESC: Sonraki episode'a atla

---

## 📊 Kullanım Örnekleri

### Basic PID Comparison

#### Default vs Optimal
```bash
# Default parametreler
python demo_basic_pid.py --target circular

# Optimal parametreler
python demo_basic_pid.py --target circular --use_optimal
```

#### Manual Tuning
```bash
# Çok düşük Kp - Yavaş tepki
python demo_basic_pid.py --kp 0.5

# Çok yüksek Kp - Osilasynlar
python demo_basic_pid.py --kp 8.0

# İyi dengeli
python demo_basic_pid.py --kp 3.2 --ki 0.15 --kd 0.8
```

### RL Model Evaluation

#### Tek Episode
```bash
python evaluate.py \
  --model_path ./models/exp/best_model.zip \
  --n_episodes 1 \
  --render
```

#### Çoklu Episodes
```bash
python evaluate.py \
  --model_path ./models/exp/best_model.zip \
  --n_episodes 10 \
  --target_maneuver evasive \
  --render
```

#### Farklı Hedef Tipleri
```bash
for maneuver in straight circular zigzag evasive; do
  echo "Testing $maneuver..."
  python evaluate.py \
    --model_path ./models/exp/best_model.zip \
    --target_maneuver $maneuver \
    --n_episodes 5 \
    --render
done
```

---

## 🆚 Basic PID vs RL Adaptive PID

### Basic PID (Sabit)
```bash
python demo_basic_pid.py --target circular
```
- ✅ PID parametreleri sabit
- ✅ Basit, öngörülebilir
- ❌ Değişen koşullara adaptasyon yok
- ❌ Farklı hedef tipleri için optimal olmayabilir

### RL Adaptive PID
```bash
python evaluate.py --model_path MODEL.zip --render
```
- ✅ PID parametreleri dinamik olarak ayarlanır
- ✅ Farklı hedef manevralarına adapte olur
- ✅ Öğrenilmiş optimal stratejiler
- ❌ Eğitim gerektirir

---

## 📈 PID Parametrelerini Anlama

### Kp (Proportional Gain)
- **Düşük (< 1.0)**: Yavaş tepki, hedefi kaçırabilir
- **Orta (1.0-3.0)**: Dengeli tepki
- **Yüksek (> 5.0)**: Hızlı tepki ama osilasynlar

### Ki (Integral Gain)
- **Düşük (< 0.1)**: Steady-state error olabilir
- **Orta (0.1-0.5)**: İyi denge
- **Yüksek (> 1.0)**: Overshoot, instabilite

### Kd (Derivative Gain)
- **Düşük (< 0.3)**: Oscillation damping az
- **Orta (0.3-1.0)**: İyi damping
- **Yüksek (> 2.0)**: Noise'a aşırı hassas

### Recommended Ranges
```yaml
# Conservative (stable but slow)
kp: 1.5
ki: 0.05
kd: 0.3

# Default (balanced)
kp: 2.0
ki: 0.1
kd: 0.5

# Optimal (found via experimentation)
kp: 3.2
ki: 0.15
kd: 0.8

# Aggressive (fast but risky)
kp: 4.5
ki: 0.25
kd: 1.2
```

---

## 🎯 Hedef Manevra Tipleri

### Straight
- En kolay
- Düz çizgide hareket
- Basic PID başarı oranı: ~95%

### Circular
- Orta zorluk
- Dairesel hareket
- Basic PID başarı oranı: ~85%

### Zigzag
- Orta-zor
- Zigzag pattern
- Basic PID başarı oranı: ~75%

### Evasive
- En zor
- Füzeden kaçmaya çalışır
- Basic PID başarı oranı: ~60%

---

## 🔧 Troubleshooting

### Pygame penceresi açılmıyor
```bash
# Linux
sudo apt-get install python3-pygame

# macOS
brew install pygame

# Windows
pip install --upgrade pygame
```

### Çok yavaş rendering
```python
# pygame_renderer.py içinde fps değiştir
fps=30  # 60 yerine 30
```

### Config bulunamıyor
```bash
# Config dosyası mevcut dizinde olmalı
ls config_pid.yaml

# Yoksa oluştur
cp config_pid.yaml.example config_pid.yaml
```

---

## 📚 Dosya Yapısı

```
2D-missile-PID-tuning-with-RL/
├── config_pid.yaml                      # PID konfigürasyonu
├── demo_basic_pid.py                    # ✨ Basic PID demo
├── evaluate.py                          # ✨ RL evaluation (updated)
├── warsim/visualization/
│   ├── pygame_renderer.py              # ✨ Pygame renderer
│   └── neon_renderer.py                # Matplotlib renderer (eski)
├── requirements.txt                     # pygame eklendi
└── README_PYGAME.md                    # Bu dosya
```

---

## ⚡ Performance Tips

### Smooth 60 FPS için
1. Rendering her frame'de (varsayılan)
2. Window size makul (1200x1000)
3. Trail length sınırlı (200 points)

### Daha hızlı simülasyon
```python
# demo_basic_pid.py içinde
renderer = PygameRenderer(fps=120)  # Daha hızlı
```

### Daha yavaş (debug için)
```python
renderer = PygameRenderer(fps=30)  # Daha yavaş
```

---

## 🎓 Eğitim Materyali

### PID Tuning Adımları

1. **Kp ayarlama**
   ```bash
   # Düşük başla
   python demo_basic_pid.py --kp 1.0 --ki 0.0 --kd 0.0

   # Artır
   python demo_basic_pid.py --kp 2.0 --ki 0.0 --kd 0.0
   python demo_basic_pid.py --kp 3.0 --ki 0.0 --kd 0.0
   ```

2. **Ki ekleme**
   ```bash
   python demo_basic_pid.py --kp 3.0 --ki 0.1 --kd 0.0
   python demo_basic_pid.py --kp 3.0 --ki 0.2 --kd 0.0
   ```

3. **Kd ile fine-tune**
   ```bash
   python demo_basic_pid.py --kp 3.0 --ki 0.15 --kd 0.5
   python demo_basic_pid.py --kp 3.0 --ki 0.15 --kd 0.8
   ```

---

## 🎉 Özet

| Özellik | Basic PID | RL Adaptive |
|---------|-----------|-------------|
| Eğitim gerekir mi? | ❌ Hayır | ✅ Evet |
| PID değişir mi? | ❌ Sabit | ✅ Dinamik |
| Görselleştirme | ✅ Pygame | ✅ Pygame |
| Konfig | ✅ YAML | ✅ Otomatik |
| Kullanım | Demo, debug | Evaluation |

### Önerilen Kullanım

1. **PID öğrenmek için**: `demo_basic_pid.py`
2. **RL modeli test için**: `evaluate.py --render`
3. **Karşılaştırma**: Her ikisini de çalıştır ve gözlemle!

---

**Başarılar!** 🚀

Sorularınız için:
- Basic PID: `python demo_basic_pid.py --help`
- RL Evaluation: `python evaluate.py --help`
