"""
03 - FMCW radar ile menzil ölçümü
=================================
FMCW (Frequency Modulated Continuous Wave) radar, frekansı doğrusal artan bir
sinyal (chirp) gönderir. Yankı, chirp'in gecikmiş kopyasıdır; ikisinin
karıştırılmasıyla elde edilen beat (fark) frekansı mesafeyle doğru orantılıdır:

    τ   = 2R / c          (yankı gecikmesi)
    f_b = S · τ = 2SR / c (S = B / Tc, chirp eğimi)
    R   = f_b · c / (2S)

Range FFT, beat frekansını bulur; eksen formülle metreye çevrilir.

FMCW range measurement: chirp, beat frequency and the range FFT.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

# --- Radar parametreleri ---
c = 3e8            # ışık hızı [m/s]
B = 150e6          # chirp bant genişliği [Hz]
Tc = 20e-6         # chirp süresi [s]
S = B / Tc         # chirp eğimi [Hz/s]
fs = 10e6          # ADC örnekleme frekansı [Hz]
t = np.arange(0, Tc, 1 / fs)

print(f"Menzil çözünürlüğü ΔR = c/2B = {c / (2 * B):.2f} m")

# --- Hedefler: (mesafe [m], genlik) ---
hedefler = [(50, 1.0), (80, 0.5)]

sinyal = np.zeros(len(t), dtype=complex)
for R, A in hedefler:
    f_b = 2 * S * R / c
    print(f"{R} m -> beat frekansı {f_b / 1e6:.2f} MHz")
    sinyal += A * np.exp(1j * 2 * np.pi * f_b * t)
sinyal += 0.5 * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))

# --- Range FFT (Hanning penceresi yan lobları bastırır) ---
N_fft = 1024
spektrum = np.abs(np.fft.fft(sinyal * np.hanning(len(t)), N_fft))
mesafeler = np.fft.fftfreq(N_fft, 1 / fs) * c / (2 * S)
pozitif = mesafeler >= 0

# --- Çizim ---
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

# Sol: chirp ve yankının zaman-frekans grafiği (oyuncak sayılarla)
tt = np.linspace(0, 1, 500)
gecikme = 0.2
ax[0].plot(tt, 100 * tt, label="Gönderilen chirp")
ax[0].plot(tt[tt >= gecikme], 100 * (tt[tt >= gecikme] - gecikme), label="Yankı (gecikmiş kopya)")
ax[0].annotate("", xy=(0.6, 60), xytext=(0.6, 40),
               arrowprops=dict(arrowstyle="<->", color="red"))
ax[0].text(0.62, 48, "beat frekansı\n(her an sabit)", color="red")
ax[0].set(title="FMCW fikri: iki paralel rampa", xlabel="Zaman", ylabel="Frekans")
ax[0].legend(loc="upper left")
ax[0].grid(alpha=0.3)

# Sağ: range FFT
dB = 20 * np.log10(spektrum[pozitif] + 1e-12)
ax[1].plot(mesafeler[pozitif], dB - dB.max())
for R, _ in hedefler:
    ax[1].axvline(R, color="r", ls="--", lw=0.8)
ax[1].set(title="Range FFT: her tepe bir hedef", xlabel="Mesafe (m)",
          ylabel="Bağıl güç (dB)", ylim=(-50, 3))
plt.tight_layout()
plt.savefig(FIG_DIR / "03_fmcw_menzil.png", dpi=150)
plt.show()
