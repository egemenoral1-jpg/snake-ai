"""
agent.py — Ajan.

Bu dosya üç şeyi bir araya getiriyor:
  1) Oyun ekranını 11 sayıya çevirmek (state)
  2) O 11 sayıyı sinir ağına verip bir karar almak (epsilon-greedy)
  3) Sonuçtan öğrenmek (hafızaya kaydet + eğit)

Yorumlarda her adımı "neden böyle" diye açıklıyorum, kod tek başına
biraz soyut kalabilir.
"""

import torch
import random
import numpy as np
from collections import deque

from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000   # hafızada en fazla kaç deneyim tutulacak
BATCH_SIZE = 1000      # her eğitim adımında hafızadan kaç deneyim örneklenecek
LR = 0.001              # öğrenme oranı — ağırlıkların ne kadar büyük adımlarla güncelleneceği


class Agent:

    def __init__(self):
        self.n_games = 0     # kaç oyun oynandı
        self.epsilon = 0     # keşif (exploration) oranını kontrol edecek
        self.gamma = 0.9     # Bellman denklemindeki "gelecek ne kadar önemli" katsayısı

        # deque = "çift uçlu kuyruk". Doluysa en eski deneyimi otomatik siler,
        # böylece hafıza sonsuza kadar büyümez.
        self.memory = deque(maxlen=MAX_MEMORY)

        # state 11 sayı, gizli katman 256 nöron, çıktı 3 eylem
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        """
        Oyun ekranını ajanın anlayacağı 11 sayıya (0 ya da 1) çeviriyoruz.
        Bu 11 sayı şunları söylüyor:

          [0-2]  Tehlike: düz gidersem / sağa dönersem / sola dönersem ölür müyüm?
          [3-6]  Şu an hangi yöne gidiyorum? (sağ, sol, yukarı, aşağı)
          [7-10] Elma benden hangi yönde? (sol, sağ, yukarı, aşağı)

        Neden bu şekilde? Çünkü yönleri MUTLAK değil, yılanın kendi
        bakış açısına göre (göreceli) verirsek, ağın öğrenmesi gereken
        örüntü sayısı çok azalır — "önümde tehlike varsa dönme" kuralı
        yılan hangi yöne bakarsa baksın aynı kalır.
        """
        head = game.snake[0]

        # başın 20 piksel (bir hücre) uzağındaki 4 nokta
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Tehlike düz ilerde
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Tehlike sağda
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Tehlike solda
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Şu anki hareket yönü
            dir_l, dir_r, dir_u, dir_d,

            # Elma nerede
            game.food.x < game.head.x,  # elma solda
            game.food.x > game.head.x,  # elma sağda
            game.food.y < game.head.y,  # elma yukarıda
            game.food.y > game.head.y   # elma aşağıda
        ]

        # True/False değerlerini 1/0'a çeviriyoruz, ağ sadece sayılarla çalışır
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Bir deneyimi ('anıyı') hafızaya ekle."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        Her oyun bittiğinde çağrılır. Hafızadan RASTGELE bir küme
        (batch) deneyim çekip toplu eğitim yapar.

        Neden rastgele? Art arda gelen deneyimler birbirine çok benzer
        (yılan hep aynı yöne gidiyordur mesela). Rastgele karıştırmak
        ağın "son ne olduysa onu ezberlemesini" değil, genel bir
        strateji öğrenmesini sağlar.
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Her TEK adımdan sonra hemen küçük bir eğitim yap (hızlı geri bildirim)."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Epsilon-greedy strateji: ajan bazen ağın önerdiğini yapar
        (exploitation / bildiğini uygulama), bazen rastgele bir şey
        dener (exploration / keşif).

        Neden rastgelelik lazım? Eğer ajan başta hep "en iyi bilinen"
        hareketi yaparsa, hiç denemediği ama aslında daha iyi olabilecek
        stratejileri asla keşfedemez. Oyun sayısı arttıkça epsilon
        küçülür — ajan tecrübelendikçe daha az rastgele, daha çok
        "bildiğini" yapar.
        """
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            # keşif: rastgele bir eylem seç
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # bildiğini uygula: ağın en yüksek Q-değeri verdiği eylemi seç
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

