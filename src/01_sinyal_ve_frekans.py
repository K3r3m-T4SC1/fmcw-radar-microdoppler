"""
01 - Sinyal ve frekans
======================
Sinüs dalgası, frekans ve genlik kavramları; iki sinüsün toplanmasıyla
oluşan karışık sinyal.

Signal & frequency: sine waves, frequency, amplitude and a mixed signal.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

t = np.linspace(0, 1, 1000)            # 1 saniye, 1000 zaman noktası

yavas = np.sin(2 * np.pi * 2 * t)      # 2 Hz: saniyede 2 tam tur
hizli = 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz, genlik 0.5
karisik = yavas + hizli

fig, ax = plt.subplots(2, 1, figsize=(9, 6))
ax[0].plot(t, yavas, label="2 Hz (genlik 1)")
ax[0].plot(t, hizli, label="5 Hz (genlik 0.5)")
ax[0].set(title="İki ayrı sinüs", xlabel="Zaman (s)", ylabel="Genlik")
ax[0].legend()
ax[1].plot(t, karisik, color="k")
ax[1].set(title="Toplamları: karışık sinyal (içinde hangi frekanslar var?)",
          xlabel="Zaman (s)", ylabel="Genlik")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_sinyal_ve_frekans.png", dpi=150)
plt.show()
