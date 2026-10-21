"""
02 - FFT temelleri
==================
FFT (Fast Fourier Transform) karışık bir sinyalin içindeki frekansları ve
genliklerini bulur. Bu script üç şeyi gösterir:
  1. Gürültülü sinyalde bile FFT tepeleri doğru yerde çıkar (işleme kazancı).
  2. FFT'nin "benzerlik testi" mantığı: çarp ve ortala.
  3. Frekans çözünürlüğü: Δf = 1 / T

FFT basics: noise robustness, the multiply-and-average intuition and
frequency resolution (Δf = 1/T).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

fs = 1000                                   # örnekleme frekansı [Hz]


def genlik_spektrumu(sinyal, fs):
    """Tek taraflı genlik spektrumu (tepe yüksekliği = sinüsün genliği)."""
    spektrum = np.fft.rfft(sinyal)
    frekanslar = np.fft.rfftfreq(len(sinyal), 1 / fs)
    return frekanslar, np.abs(spektrum) * 2 / len(sinyal)


# --- 1. Gürültülü sinyal ve FFT ---
t = np.arange(0, 1, 1 / fs)
temiz = 1.0 * np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)
gurultulu = temiz + 1.5 * rng.standard_normal(len(t))
f, A = genlik_spektrumu(gurultulu, fs)

# --- 2. Benzerlik testi: FFT'yi elle yapmak ---
print("Benzerlik testi (çarp ve ortala):")
for f_test in range(1, 7):
    benzerlik = np.mean(temiz * np.sin(2 * np.pi * f_test * t)) * 2
    print(f"  {f_test} Hz -> {benzerlik:.2f}")

# --- 3. Çözünürlük: 5 Hz ve 5.3 Hz ---
cozunurluk = []
for T in (1, 10):
    tt = np.arange(0, T, 1 / fs)
    s = np.sin(2 * np.pi * 5 * tt) + np.sin(2 * np.pi * 5.3 * tt)
    cozunurluk.append((T, *genlik_spektrumu(s, fs)))

fig, ax = plt.subplots(3, 1, figsize=(9, 9))
ax[0].plot(t, gurultulu, lw=0.8)
ax[0].set(title="Zaman ekseni: gürültü sinyali boğuyor", xlabel="Zaman (s)")
ax[1].plot(f, A)
ax[1].set(title="Frekans ekseni (FFT): 2 Hz ve 5 Hz tepeleri hâlâ belirgin",
          xlabel="Frekans (Hz)", ylabel="Genlik", xlim=(0, 10))
for T, ff, AA in cozunurluk:
    ax[2].plot(ff, AA, label=f"T = {T} s  (Δf = {1 / T:g} Hz)")
ax[2].set(title="Çözünürlük: 5 Hz ve 5.3 Hz ancak uzun sinyalde ayrılıyor",
          xlabel="Frekans (Hz)", ylabel="Genlik", xlim=(4, 6.5))
ax[2].legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "02_fft_temelleri.png", dpi=150)
plt.show()
