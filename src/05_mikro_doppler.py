"""
05 - Mikro-Doppler: dron ve kuşun parmak izi
=============================================
Gövde tek bir hızla ilerler; dönen pervaneler ve çırpan kanatlar ise ek,
zamanla değişen hızlar üretir (mikro-Doppler). Bu deseni görmek için yavaş
zaman sinyaline STFT (Short-Time Fourier Transform) uygulanır ve spektrogram
çizilir. Hedefler, küçük yansıtıcı noktaların (scatterer) toplamı olarak
modellenir.

  Dron : düzenli dikey "blade flash"lar; aralık -> devir, yükseklik -> kanat ucu hızı
  Kuş  : gövde çizgisi etrafında yavaş sinüs dalgası

Not: Pervane ucu ≈ 60 m/s olduğundan 77 GHz'teki ±48.7 m/s sınırı aşılır.
Bu yüzden burada 24 GHz (λ = 12.5 mm) ve PRF = 50 kHz kullanılır (±156 m/s).

Micro-Doppler signatures of a drone and a bird via STFT spectrograms.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

# --- Radar parametreleri ---
c = 3e8
fc = 24e9                      # K bandı
lam = c / fc                   # 12.5 mm
PRF = 50_000                   # saniyedeki chirp sayısı
t = np.arange(0, 1.0, 1 / PRF) # yavaş zaman, 1 saniye

print(f"Maksimum ölçülebilir hız: ±{lam * PRF / 4:.0f} m/s")


def yanki(R, A):
    """Tek bir yansıtıcı noktanın yavaş zaman sinyali."""
    return A * np.exp(1j * 4 * np.pi * R / lam)


def dron_sinyali(R0=50, v=5, yaricap=0.12, devir=80, kanat_sayisi=2, n_nokta=15):
    s = yanki(R0 + v * t, 1.0)                                   # gövde
    for k in range(kanat_sayisi):
        aci = 2 * np.pi * devir * t + k * 2 * np.pi / kanat_sayisi
        for l in np.linspace(0.01, yaricap, n_nokta):          # kanat boyunca noktalar
            s += yanki(R0 + v * t + l * np.cos(aci), 0.05)
    return s


def kus_sinyali(R0=50, v=5, kanat_boyu=0.30, cirpma=5, genlik_aci=0.8, n_nokta=15):
    s = yanki(R0 + v * t, 1.0)                                   # gövde
    kanat_acisi = genlik_aci * np.sin(2 * np.pi * cirpma * t)   # yukarı-aşağı
    for l in np.linspace(0.02, kanat_boyu, n_nokta):
        s += 2 * yanki(R0 + v * t + l * np.sin(kanat_acisi), 0.05)  # 2 kanat
    return s


def spektrogram(sinyal, pencere):
    f, zaman, Z = stft(sinyal, fs=PRF, nperseg=pencere, noverlap=pencere * 7 // 8,
                       return_onesided=False)
    f, Z = np.fft.fftshift(f), np.fft.fftshift(Z, axes=0)
    dB = 20 * np.log10(np.abs(Z) + 1e-12)
    return zaman, f * lam / 2, dB - dB.max()


def gurultu_ekle(s, seviye=0.05):
    return s + seviye * (rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s)))


if __name__ == "__main__":
    # (ad, sinyal, pencere, gösterilen süre [s], hız sınırı [m/s])
    ayarlar = [("Dron (2 kanat, 80 tur/s)", gurultu_ekle(dron_sinyali()), 64, 0.1, 80),
               ("Kuş (5 çırpma/s)", gurultu_ekle(kus_sinyali()), 1024, 1.0, 20)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, (ad, s, pencere, goster, vlim) in zip(ax, ayarlar):
        zaman, hiz, dB = spektrogram(s, pencere)
        im = a.pcolormesh(zaman, hiz, dB, vmin=-40, cmap="jet", shading="auto")
        a.set(title=f"{ad}: mikro-Doppler", xlabel="Zaman (s)", ylabel="Hız (m/s)",
              xlim=(0, goster), ylim=(-vlim, vlim))
        fig.colorbar(im, ax=a, label="Güç (dB)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_mikro_doppler.png", dpi=150)
    plt.show()
