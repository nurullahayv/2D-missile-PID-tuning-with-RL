# Model Evaluation Guide

Model eğittikten sonra performansını değerlendirmek için bu rehberi kullanın.

## 🚀 Hızlı Başlangıç

### Yöntem 1: Otomatik Script (Önerilen)

Model dosyanızı otomatik bulup evaluate eder:

```bash
# En son modeli otomatik bul ve evaluate et
./run_evaluation.sh

# Belirli bir model ile
./run_evaluation.sh ./models/experiment_name/best_model.zip

# Farklı hedef tipi ile
./run_evaluation.sh ./models/experiment_name/best_model.zip circular

# Daha fazla episode ile
./run_evaluation.sh ./models/experiment_name/best_model.zip circular 20
```

### Yöntem 2: Manuel Python Komutu

```bash
# Temel kullanım
python evaluate.py --model_path ./models/EXPERIMENT_NAME/final_model.zip

# Tüm parametrelerle
python evaluate.py \
  --model_path ./models/EXPERIMENT_NAME/best_model.zip \
  --n_episodes 10 \
  --target_maneuver circular \
  --output_dir ./evaluation_results/circular
```

## 📁 Model Dosyasını Bulma

### 1. Modeliniz nerede?

Eğitim sırasında modeller şu konuma kaydedilir:

```
./models/<experiment_name>/
├── best_model.zip          # En iyi performans (önerilen)
├── final_model.zip         # Son model
├── model_50000_steps.zip   # Checkpoint 1
├── model_100000_steps.zip  # Checkpoint 2
└── ...
```

### 2. Model dosyalarını listeleyin:

```bash
# Tüm model dosyalarını göster
find ./models -name '*.zip' -type f

# En son kaydedilen modeli bul
find ./models -name '*.zip' -type f -printf '%T+ %p\n' | sort -r | head -5
```

### 3. Hangi modeli kullanmalı?

- **best_model.zip** → En iyi validation performansı (önerilen)
- **final_model.zip** → Eğitimin sonundaki model
- **model_XXXXX_steps.zip** → Belirli bir checkpoint

## 🎯 Evaluation Parametreleri

### Temel Parametreler

```bash
--model_path         # Model dosyası yolu (zorunlu)
--n_episodes 10      # Test episode sayısı (varsayılan: 10)
--target_maneuver    # Hedef tipi (varsayılan: straight)
--output_dir         # Sonuç klasörü (varsayılan: ./evaluation_results)
```

### Hedef Tipleri

| Parametre | Açıklama | Zorluk |
|-----------|----------|--------|
| `straight` | Düz çizgide hareket | Kolay |
| `circular` | Dairesel hareket | Orta |
| `zigzag` | Zigzag manevra | Orta-Zor |
| `evasive` | Füzeden kaçış | Zor |

## 📊 Örnek Komutlar

### Tek Hedef Tipi

```bash
# Düz hareket eden hedefe karşı
python evaluate.py \
  --model_path ./models/exp/best_model.zip \
  --target_maneuver straight \
  --n_episodes 10

# Dairesel hareket eden hedefe karşı
python evaluate.py \
  --model_path ./models/exp/best_model.zip \
  --target_maneuver circular \
  --n_episodes 10

# Kaçış manevrası yapan hedefe karşı
python evaluate.py \
  --model_path ./models/exp/best_model.zip \
  --target_maneuver evasive \
  --n_episodes 20
```

### Tüm Hedef Tiplerini Test Et

```bash
# Otomatik script ile (önerilen)
./evaluate_all_targets.sh ./models/exp/best_model.zip

# Manuel olarak
for maneuver in straight circular zigzag evasive; do
  echo "Testing $maneuver..."
  python evaluate.py \
    --model_path ./models/exp/best_model.zip \
    --target_maneuver $maneuver \
    --n_episodes 10 \
    --output_dir ./evaluation_results/$maneuver
done
```

### Farklı Modelleri Karşılaştır

```bash
# Checkpoint modelleri karşılaştır
for model in ./models/exp/model_*_steps.zip; do
  model_name=$(basename $model .zip)
  echo "Evaluating $model_name..."
  python evaluate.py \
    --model_path $model \
    --target_maneuver circular \
    --n_episodes 10 \
    --output_dir ./evaluation_results/$model_name
done
```

## 📈 Çıktılar

Evaluation sonuçları `./evaluation_results/` klasörüne kaydedilir:

```
./evaluation_results/
├── circular/
│   ├── evaluation_summary.png       # Özet grafikler
│   ├── trajectory_episode_1.png     # Trajectory 1
│   ├── trajectory_episode_2.png     # Trajectory 2
│   └── trajectory_episode_3.png     # Trajectory 3
├── straight/
│   └── ...
└── evasive/
    └── ...
```

### Grafikler

1. **evaluation_summary.png** - 6 panel:
   - Episode rewards
   - Episode lengths
   - Hit success rate
   - Final distances
   - PID gains evolution
   - PID gains distribution

2. **trajectory_episode_X.png**:
   - Füze trajectory (mavi)
   - Hedef trajectory (kırmızı)
   - Hit radius (kırmızı çember)

### Terminal Çıktısı

```
Episode 1/10: Reward=250.45, Length=180, Hit=Yes, Final Distance=35.20m
Episode 2/10: Reward=180.32, Length=220, Hit=Yes, Final Distance=42.10m
...
================================
Evaluation Summary
================================
Average Reward: 215.32 ± 45.12
Average Episode Length: 195.40 ± 30.25
Hit Success Rate: 85.0%
Average Final Distance: 38.45m
```

## 🔍 Sonuçları Analiz Etme

### İyi Performans Göstergeleri

✅ **Yüksek Hit Rate**: >80%
✅ **Kısa Episode Length**: <250 steps
✅ **Düşük Final Distance**: <100m
✅ **Stabil PID Gains**: Smooth değişim

### Sorun Göstergeleri

❌ **Düşük Hit Rate**: <50%
❌ **Uzun Episode Length**: >400 steps
❌ **Yüksek Final Distance**: >500m
❌ **Unstable PID Gains**: Hızlı osilasynlar

### İyileştirme Önerileri

**Hit rate düşükse:**
- Daha uzun eğitim
- Learning rate azalt
- Reward weights ayarla

**PID gains unstable ise:**
- Action penalty weight artır
- Batch size artır
- Training daha smooth yap

## 📊 Benchmark Sonuçlar

Referans değerler (1M timesteps, PPO):

| Target Type | Hit Rate | Avg Steps | Avg Distance |
|------------|----------|-----------|--------------|
| Straight   | ~95%     | ~200      | ~25m        |
| Circular   | ~85%     | ~250      | ~40m        |
| Zigzag     | ~80%     | ~280      | ~55m        |
| Evasive    | ~70%     | ~300      | ~80m        |

## 🛠️ Troubleshooting

### Model dosyası bulunamıyor

```bash
# Model klasörlerini kontrol et
ls -R ./models/

# Training log'larını kontrol et
ls -R ./logs/
```

### Import hatası

```bash
# Python path ayarla
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Paketleri kontrol et
pip install -r requirements.txt
```

### CUDA/GPU hatası

```bash
# CPU'da çalıştır (evaluation için yeterli)
python evaluate.py --model_path ... --device cpu
```

### Grafik görüntülenmiyor

```bash
# matplotlib backend ayarla
export MPLBACKEND=Agg

# Sadece dosyaya kaydet
python evaluate.py --model_path ... --save_plots
```

## 📝 İleri Seviye

### Custom Evaluation Script

```python
from evaluate import evaluate_model

results = evaluate_model(
    model_path='./models/exp/best_model.zip',
    n_episodes=20,
    target_maneuver='circular',
    render=False,
    save_plots=True,
    output_dir='./my_results'
)

# Sonuçları analiz et
print(f"Mean reward: {np.mean(results['rewards']):.2f}")
print(f"Hit rate: {np.mean(results['hit_success'])*100:.1f}%")
```

### Batch Evaluation

```python
# evaluate_batch.py
import subprocess
import os

models = [
    './models/exp1/best_model.zip',
    './models/exp2/best_model.zip',
    './models/exp3/best_model.zip',
]

for model_path in models:
    exp_name = os.path.basename(os.path.dirname(model_path))
    for maneuver in ['straight', 'circular', 'zigzag', 'evasive']:
        print(f"Evaluating {exp_name} on {maneuver}...")
        subprocess.run([
            'python', 'evaluate.py',
            '--model_path', model_path,
            '--target_maneuver', maneuver,
            '--n_episodes', '10',
            '--output_dir', f'./results/{exp_name}/{maneuver}'
        ])
```

## 💡 Tips

1. **Her zaman best_model.zip kullanın** - En iyi performansı verir
2. **Tüm hedef tiplerini test edin** - Generalization görmek için
3. **Yeterli episode sayısı** - En az 10, ideal 20-50
4. **Sonuçları karşılaştırın** - Farklı modelleri ve parametreleri
5. **Trajectory'leri inceleyin** - PID davranışını görsel olarak anlayın

## 🎓 Yorumlama

### Örnek Sonuç Analizi

```
Target: Circular
Hit Rate: 85%
Avg Steps: 245
Final Distance: 42m
```

**Yorum:**
- ✅ İyi hit rate (>80%)
- ✅ Makul step count
- ✅ Hedefe yakın (<50m)
- → Model circular hedeflere karşı başarılı!

```
PID Evolution:
Kp: 2.5 → 3.2 → 2.8 (smooth)
Ki: 0.1 → 0.2 → 0.15 (stable)
Kd: 0.5 → 0.8 → 0.6 (gradual)
```

**Yorum:**
- ✅ Smooth değişimler
- ✅ Makul range'lerde
- ✅ Episode boyunca adaptif
- → RL PID parametrelerini başarıyla ayarlıyor!

## 📧 Yardım

Sorularınız için:
1. `QUICK_START.md` dosyasına bakın
2. `README.md` ana dökümana bakın
3. Issue açın
