"""
model.py — Ajanın "beyni".

Klasik Q-learning'de her (state, action) çifti için bir Q-değeri
tablosu tutulur. Ama Snake'in olası durum sayısı çok büyük olduğu için
tablo yerine bir SİNİR AĞI kullanıyoruz: state'i girdi olarak alıp
3 eylemin (düz/sağ/sol) her biri için bir Q-değeri tahmin ediyor.

Q-değeri = "bu durumda bu eylemi yaparsam, gelecekte toplam ne kadar
ödül toplarım?" tahmini.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    """
    Basit bir ileri beslemeli (feed-forward) sinir ağı:
    input_size  -> hidden_size -> output_size

    input_size  = 11 (state.py / agent.py'da tanımlayacağımız 11 özellik)
    hidden_size = 256 (deneyerek bulunmuş makul bir değer)
    output_size = 3  (düz git / sağa dön / sola dön için Q-değerleri)
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # ReLU: negatif değerleri sıfırlayan aktivasyon fonksiyonu,
        # ağın doğrusal olmayan ilişkiler öğrenmesini sağlar.
        x = F.relu(self.linear1(x))
        x = self.linear2(x)  # son katmanda aktivasyon yok, ham Q-değerleri
        return x

    def save(self, file_name='model.pth'):
        """Eğitilmiş ağırlıkları diske kaydeder."""
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """
    Ağı Bellman denklemine göre eğiten sınıf.

    Bellman denklemi (Q-learning'in kalbi):
        Q_yeni(s, a) = ödül + gamma * max(Q(s_sonraki))

    Yani: "bu eylemin gerçek değeri, aldığım anlık ödül + gelecekte
    alabileceğim en iyi ödülün (indirimli) tahminidir."
    gamma (0-1 arası) gelecekteki ödülleri ne kadar önemsediğimizi belirler.
    """

    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()  # tahmin ile hedef arasındaki fark

    def train_step(self, state, action, reward, next_state, done):
        # Tek bir örnek de gelebilir, bir batch (küme) de — ikisini de
        # aynı şekilde işleyebilmek için hepsini "batch" formatına çeviriyoruz.
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # tek boyutluysa (tek örnek) başına bir batch boyutu ekle
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1. Şu anki tahminler: Q(s, a)
        pred = self.model(state)

        # 2. Hedef Q-değerlerini hesapla (Bellman denklemi)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # oyun bitmediyse, gelecekteki en iyi Q-değerini de ekle
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # sadece seçilen eylemin Q-değerini güncelle, diğerleri aynı kalsın
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 3. Tahmin ile hedef arasındaki farkı azaltacak şekilde ağı güncelle
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()