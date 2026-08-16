"""
helper.py — Eğitim ilerlerken skorları canlı bir grafikte gösterir.

Bu dosya ML mantığıyla ilgili değil, sadece "ajan gerçekten öğreniyor mu?"
sorusunun cevabını göz ile görebilmemiz için bir grafik çiziyor.
Ortalama skor (mean_scores) zamanla yukarı doğru bir eğim gösteriyorsa,
ajan gerçekten öğreniyor demektir.
"""

import matplotlib.pyplot as plt
from IPython import display

plt.ion()  # interaktif mod: grafiği her seferinde yeniden çizip güncelleyebilelim


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Eğitim İlerlemesi')
    plt.xlabel('Oyun Sayısı')
    plt.ylabel('Skor')
    plt.plot(scores, label='Skor')
    plt.plot(mean_scores, label='Ortalama Skor')
    plt.ylim(ymin=0)
    plt.legend()
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)