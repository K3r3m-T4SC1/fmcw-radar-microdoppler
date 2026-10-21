"""
04 - Menzil-Doppler haritası (Range-Doppler Map)
================================================
Tek chirp Doppler kaymasını göremez (20 µs'lik chirp'te FFT çözünürlüğü 50 kHz).
Bunun yerine art arda çok chirp gönderilir; hareketli hedefin fazı her chirp'te
biraz döner. Veriler bir tabloya dizilir (satır = chirp, sütun = örnek):

  1. FFT satırlar boyunca (hızlı zaman)  -> mesafe
  2. FFT sütunlar boyunca (yavaş zaman)  -> hız

    f_D   = 2v / λ          (Doppler kayması)
    v_max = λ / (4 Tc)      (bunu aşan hız aliasing yapar)

Range-Doppler processing with a 2D FFT over fast time and slow time.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

# --- Radar parametreleri ---
c = 3e8
fc = 77e9                      # taşıyıcı frekans [Hz]
lam = c / fc                   # dalga boyu ≈ 3.9 mm
B, Tc, fs = 150e6, 20e-6, 10e6
S = B / Tc
t = np.arange(0, Tc, 1 / fs)   # hızlı zaman (200 örnek)
n_chirp = 64                   # yavaş zaman

print(f"Maksimum ölçülebilir hız: ±{lam / (4 * Tc):.1f} m/s")
print(f"Hız çözünürlüğü: {lam / (2 * n_chirp * Tc):.2f} m/s")

# --- Hedefler: (mesafe [m], hız [m/s], genlik); hız < 0 = yaklaşıyor ---
hedefler = [(50, -10, 1.0),
            (80, 15, 0.5),
            (30, 0, 0.7)]       # duran nesne (ör. bina, ağaç)

veri = np.zeros((n_chirp, len(t)), dtype=complex)
for k in range(n_chirp):
    for R0, v, A in hedefler:
        R = R0 + v * k * Tc                    # hedef her chirp'te biraz ilerler
        f_b = 2 * S * R / c
        faz = 4 * np.pi * R / lam              # mesafeye çok hassas faz
        veri[k] += A * np.exp(1j * (2 * np.pi * f_b * t + faz))
veri += 0.5 * (rng.standard_normal(veri.shape) + 1j * rng.standard_normal(veri.shape))

# --- 2D FFT (her iki eksende Hanning penceresi) ---
pencere = np.outer(np.hanning(n_chirp), np.hanning(len(t)))
menzil_fft = np.fft.fft(veri * pencere, axis=1)
rdm = np.fft.fftshift(np.fft.fft(menzil_fft, axis=0), axes=0)
rdm_dB = 20 * np.log10(np.abs(rdm) + 1e-12)

mesafeler = np.fft.fftfreq(len(t), 1 / fs) * c / (2 * S)
hizlar = np.fft.fftshift(np.fft.fftfreq(n_chirp, Tc)) * lam / 2
yarim = len(t) // 2

# 50 m'deki hedefin fazı chirp'ten chirp'e (pencere olmadan)
kutu = np.argmin(np.abs(mesafeler - 50))
fazlar = np.unwrap(np.angle(np.fft.fft(veri, axis=1)[:, kutu]))

# --- Çizim ---
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(fazlar, "o-", ms=3)
ax[0].set(title="50 m'deki dronun fazı düzenli dönüyor", xlabel="Chirp numarası",
          ylabel="Faz (radyan)")
im = ax[1].imshow(rdm_dB[:, :yarim], aspect="auto", origin="lower", cmap="jet",
                  extent=[0, mesafeler[yarim - 1], hizlar[0], hizlar[-1]],
                  vmin=rdm_dB.max() - 40)
fig.colorbar(im, ax=ax[1], label="Güç (dB)")
ax[1].set(title="Menzil-Doppler haritası", xlabel="Mesafe (m)", ylabel="Hız (m/s)")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_menzil_doppler_haritasi.png", dpi=150)
plt.show()
