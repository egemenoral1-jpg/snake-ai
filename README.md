# Snake AI — Pekiştirmeli Öğrenme (DQN)

Deep Q-Learning ile Snake oynamayı öğrenen bir yapay zeka.

## Dosyalar
- `game.py`    — Oyun ortamı (environment)
- `model.py`   — Sinir ağı + eğitici (Q-learning / Bellman denklemi)
- `agent.py`   — State çıkarma, epsilon-greedy karar verme, hafıza
- `helper.py`  — Eğitim sırasında canlı skor grafiği
- `train.py`   — Her şeyi birleştiren ana döngü

## Kurulum
```bash
pip install -r requirements.txt
```

## Çalıştırma
```bash
python train.py
```

Pencere açılacak ve yılan başta rastgele hareket edip duracak.
Birkaç yüz oyun sonra elmaları düzgün bulmaya, birkaç bin oyun
sonra da gayet iyi oynamaya başlayacak. Durdurmak için Ctrl+C.

En iyi model otomatik olarak `model/model.pth` içine kaydedilir.

## Nasıl çalışıyor (özet)
1. Ajan oyunu 11 sayıya indirger (tehlike + yön + elma konumu).
2. Bu 11 sayıyı sinir ağına verip 3 eylem (düz/sağ/sol) için
   Q-değeri tahmini alır.
3. Başta rastgele hareket eder (keşif), zamanla ağın önerdiğini
   uygulamaya başlar (epsilon-greedy).
4. Her adımdan ve her oyun sonunda (hafızadan rastgele örnekleyerek)
   Bellman denklemine göre ağı günceller.
