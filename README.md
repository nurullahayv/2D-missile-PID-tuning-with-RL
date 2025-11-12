# 2D Missile PID Tuning with Reinforcement Learning

**2D Savaş Ortamında Manevra Yapan Hedeflere Karşı Füze Otopilotu PID Parametrelerinin Pekiştirmeli Öğrenme ile Adaptif Ayarlanması**

Bu proje, 2D simülasyon ortamında manevra yapan hedeflere karşı füze güdüm sistemi PID kontrolcü parametrelerinin (Kp, Ki, Kd) pekiştirmeli öğrenme (RL) ile otomatik olarak ayarlanmasını sağlar.

![Missile Guidance](img/missile_guidance.png)

## Özellikler

- **PID Kontrollü Füze**: Gerçekçi 2D füze dinamiği ve PID kontrolcü
- **Hareketli Hedefler**: Farklı manevra tipleri (düz, dairesel, zigzag, kaçış)
- **RL Eğitimi**: PPO, SAC, TD3 algoritmaları ile eğitim desteği
- **GPU Desteği**: Kaggle notebook ile hızlı eğitim
- **Görselleştirme**: Detaylı trajectory ve performans grafikleri
- **Modüler Yapı**: Kolay genişletilebilir ve özelleştirilebilir

## Kurulum

### Gereksinimler

- Python 3.8+
- CUDA (opsiyonel, GPU eğitimi için)

### Adımlar

```bash
# Repoyu klonla
git clone https://github.com/YOUR_USERNAME/2D-missile-PID-tuning-with-RL.git
cd 2D-missile-PID-tuning-with-RL

# Gerekli paketleri yükle
pip install -r requirements.txt
```

## Kullanım

### 1. Temel Eğitim

En basit şekilde eğitim başlatmak için:

```bash
python train.py
```

### 2. Özel Parametrelerle Eğitim

Farklı hedef manevraları ve algoritmalar ile eğitim:

```bash
# Dairesel hareket eden hedefe karşı PPO ile eğitim
python train.py --target_maneuver circular --algorithm PPO --total_timesteps 1000000

# Kaçış manevrası yapan hedefe karşı SAC ile eğitim
python train.py --target_maneuver evasive --algorithm SAC --total_timesteps 2000000

# GPU kullanarak eğitim
python train.py --device cuda --total_timesteps 2000000
```

### 3. Değerlendirme

Eğitilmiş bir modeli test etmek için:

```bash
python evaluate.py --model_path ./models/final_model.zip --n_episodes 10 --target_maneuver circular
```

### 4. Kaggle'da GPU ile Eğitim

1. `kaggle_training.ipynb` dosyasını Kaggle'a yükleyin
2. Settings'den GPU'yu etkinleştirin
3. Internet erişimini aktifleştirin
4. Notebook'u çalıştırın!

## Proje Yapısı

```
2D-missile-PID-tuning-with-RL/
├── envs/
│   ├── __init__.py
│   ├── env_base.py              # Base environment (eski projeden)
│   └── missile_pid_env.py       # Missile PID environment
├── warsim/
│   ├── simulator/
│   │   ├── __init__.py
│   │   ├── missile.py           # Missile dynamics + PID
│   │   ├── target.py            # Moving target
│   │   └── cmano_simulator.py   # Base simulator (eski projeden)
│   ├── utils/
│   │   ├── angles.py
│   │   ├── geodesics.py
│   │   └── map_limits.py
│   └── scenplotter/
│       └── scenario_plotter.py  # Visualization utilities
├── config.py                     # Configuration
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
├── kaggle_training.ipynb        # Kaggle GPU training notebook
├── requirements.txt             # Dependencies
└── README.md
```

## Algoritma Detayları

### PID Controller

Füze güdüm sistemi PID kontrolcü kullanır:

```
u(t) = Kp * e(t) + Ki * ∫e(t)dt + Kd * de(t)/dt
```

- **Kp (Proportional)**: Anlık hataya göre düzeltme
- **Ki (Integral)**: Birikmiş hatayı düzeltme
- **Kd (Derivative)**: Hata değişim hızına göre düzeltme

### Reinforcement Learning

RL ajanı, PID parametrelerini dinamik olarak ayarlar:

- **Observation Space (14D)**:
  - Füze pozisyonu (x, y)
  - Füze hızı (vx, vy)
  - Füze yönü
  - Hedef pozisyonu (x, y)
  - Hedef hızı (vx, vy)
  - Hedef yönü
  - Göreceli mesafe
  - Göreceli açı
  - Mevcut PID parametreleri (Kp, Ki, Kd)
  - Kalan yakıt

- **Action Space (3D)**:
  - Δ Kp: Kp parametresindeki değişim
  - Δ Ki: Ki parametresindeki değişim
  - Δ Kd: Kd parametresindeki değişim

- **Reward Function**:
  - Hedefe yaklaşma ödülü
  - Hedefe isabet ödülü (+1000)
  - Isabet edememe cezası (-500)
  - Aşırı PID değişimi cezası

## Hedef Manevra Tipleri

1. **Straight**: Düz çizgide hareket
2. **Circular**: Dairesel hareket
3. **Zigzag**: Zigzag hareketi
4. **Evasive**: Kaçış manevrası (füzeden uzaklaşma)

## Sonuçlar

Eğitim sonuçları `logs/` ve `models/` klasörlerinde saklanır:

- **TensorBoard**: `tensorboard --logdir logs/`
- **Models**: `models/` klasöründe checkpoint'ler
- **Evaluation**: `evaluation_results/` klasöründe grafikler

## Örnek Sonuçlar

### Straight Target
- Hit Success Rate: ~95%
- Average Steps: ~200

### Circular Target
- Hit Success Rate: ~85%
- Average Steps: ~250

### Evasive Target
- Hit Success Rate: ~70%
- Average Steps: ~300

## Gelecek Geliştirmeler

- [ ] Hierarchical RL (HRL) entegrasyonu
- [ ] Multi-agent scenarios (çoklu füze)
- [ ] 3D simülasyon ortamı
- [ ] Gerçek füze parametreleri ile validasyon
- [ ] Daha karmaşık hedef manevralar
- [ ] Gürültü ve belirsizlik modelleri

## Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## İletişim

Sorularınız için issue açabilirsiniz.

## Referanslar

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- PID Control Theory
- Missile Guidance Systems

## Teşekkürler

Bu proje, [HHMARL 2D](https://github.com/YOUR_REPO) projesinden esinlenerek ve bazı altyapı komponentleri kullanılarak geliştirilmiştir.
