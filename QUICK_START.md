# Quick Start Guide

Bu rehber, projeyi hızlıca başlatmanız için temel adımları içerir.

## 1. Kurulum

```bash
# Paketleri yükle
pip install -r requirements.txt
```

## 2. Basit Test

Kurulumun doğru olduğunu test edin:

```bash
python test_setup.py
```

## 3. İlk Eğitim (Hızlı Test)

En basit hedef (düz hareket) ile kısa bir eğitim:

```bash
python train.py \
  --target_maneuver straight \
  --total_timesteps 50000 \
  --algorithm PPO
```

Bu yaklaşık 5-10 dakika sürer (CPU'da).

## 4. Tam Eğitim

Daha uzun ve etkili eğitim için:

```bash
python train.py \
  --target_maneuver circular \
  --total_timesteps 1000000 \
  --algorithm PPO \
  --device cuda  # GPU varsa
```

## 5. Eğitimi İzleme

TensorBoard ile eğitimi canlı izleyin:

```bash
tensorboard --logdir logs/
```

Tarayıcıda `http://localhost:6006` adresine gidin.

## 6. Model Değerlendirme

Eğitilmiş modeli test edin:

```bash
python evaluate.py \
  --model_path ./models/YOUR_EXP_NAME/final_model.zip \
  --n_episodes 10 \
  --target_maneuver circular
```

## 7. Kaggle'da Eğitim (GPU)

1. `kaggle_training.ipynb` dosyasını Kaggle'a yükleyin
2. GPU'yu etkinleştirin (Settings > Accelerator > GPU)
3. Internet'i açın (Add Data > Internet)
4. Tüm hücreleri çalıştırın

## Hedef Manevra Tipleri

- `straight`: Düz hareket (en kolay)
- `circular`: Dairesel hareket (orta)
- `zigzag`: Zigzag hareketi (orta-zor)
- `evasive`: Kaçış manevrası (zor)

## Algoritmalar

- `PPO`: En stabil ve güvenilir (önerilen)
- `SAC`: Daha hızlı öğrenir, ama daha az stabil
- `TD3`: Continuous action için optimize edilmiş

## Örnek Komutlar

### Dairesel hedef, PPO, 1M steps
```bash
python train.py --target_maneuver circular --total_timesteps 1000000
```

### Kaçış manevrası, SAC, GPU
```bash
python train.py --target_maneuver evasive --algorithm SAC --device cuda
```

### Özel parametrelerle eğitim
```bash
python train.py \
  --target_maneuver zigzag \
  --algorithm PPO \
  --total_timesteps 2000000 \
  --learning_rate 0.0003 \
  --batch_size 128 \
  --hidden_size 512
```

## Beklenen Sonuçlar

| Hedef Tipi | Hit Rate | Ortalama Steps |
|-----------|----------|----------------|
| Straight  | ~95%     | ~200           |
| Circular  | ~85%     | ~250           |
| Zigzag    | ~80%     | ~280           |
| Evasive   | ~70%     | ~300           |

## Sorun Giderme

### Import Hatası
```bash
# Python path'i kontrol edin
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### GPU Kullanımı
```bash
# CUDA kontrolü
python -c "import torch; print(torch.cuda.is_available())"
```

### Düşük Performans
1. Learning rate'i azaltın (`--learning_rate 0.0001`)
2. Batch size'ı artırın (`--batch_size 128`)
3. Network boyutunu artırın (`--hidden_size 512`)
4. Daha fazla timestep kullanın (`--total_timesteps 2000000`)

## İleri Seviye

### Curriculum Learning
Kolay hedeflerden zor hedeflere:

```bash
# 1. Düz hareket
python train.py --target_maneuver straight --total_timesteps 500000

# 2. Dairesel (önceki modelden devam)
python train.py --target_maneuver circular --total_timesteps 500000 \
  --model_path ./models/previous_exp/final_model.zip

# 3. Kaçış
python train.py --target_maneuver evasive --total_timesteps 1000000 \
  --model_path ./models/previous_exp/final_model.zip
```

### Hyperparameter Tuning
Farklı kombinasyonları deneyin:

```bash
for lr in 0.0001 0.0003 0.001; do
  for hs in 128 256 512; do
    python train.py \
      --learning_rate $lr \
      --hidden_size $hs \
      --exp_name "lr_${lr}_hs_${hs}"
  done
done
```

## Yardım

Tüm parametreleri görmek için:

```bash
python train.py --help
python evaluate.py --help
```
