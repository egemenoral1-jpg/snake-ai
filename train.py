"""
train.py — Her şeyi birleştiren ana döngü.
 
Bu dosyayı çalıştırdığında:
  1) Bir Agent (ajan) ve SnakeGameAI (oyun) oluşturulur.
  2) Sonsuz bir döngüde: durum oku -> eylem seç -> uygula -> öğren.
  3) Her oyun bittiğinde skor yazdırılır ve grafik güncellenir.
  4) Yeni bir rekor kırıldığında model diske kaydedilir (model/model.pth).
 
Durdurmak için Ctrl+C.
"""
 
from agent import Agent
from game import SnakeGameAI
from helper import plot
 
 
def train():
    plot_scores = []       # her oyunun skoru
    plot_mean_scores = []  # o ana kadarki ortalama skor
    total_score = 0
    record = 0              # şimdiye kadarki en yüksek skor
 
    agent = Agent()
    game = SnakeGameAI()
 
    while True:
        # 1. şu anki durumu (11 sayı) al
        state_old = agent.get_state(game)
 
        # 2. bir eylem seç (epsilon-greedy: keşif ya da bildiğini uygulama)
        final_move = agent.get_action(state_old)
 
        # 3. eylemi oyuna uygula, sonucu al
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
 
        # 4. bu tek adımdan hemen öğren (kısa vadeli)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
 
        # 5. bu deneyimi hafızaya kaydet, sonra tekrar kullanılacak
        agent.remember(state_old, final_move, reward, state_new, done)
 
        if done:
            # oyun bitti (yılan öldü): oyunu sıfırla, hafızadan toplu öğren
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
 
            if score > record:
                record = score
                agent.model.save()  # yeni rekor -> ağırlıkları kaydet
 
            print(f'Oyun {agent.n_games}  Skor {score}  Rekor: {record}')
 
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
 
 
if __name__ == '__main__':
    train()