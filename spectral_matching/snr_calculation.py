import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nano_txt = np.loadtxt("/dirs/data/tirs/axhcis/Projects/NURI/Data/nano_blue_panel_radiance_spectra.txt", skiprows=7)
swir_txt = np.loadtxt("/dirs/data/tirs/axhcis/Projects/NURI/Data/swir_blue_panel_radiance_spectra.txt", skiprows=7)


plt.figure(figsize = (7,7))
plt.plot(nano_txt[:,0], nano_txt[:,4]/(nano_txt[:,5] - nano_txt[:,4] + 1e-5), label = "SNR Nano Blue Panel")
plt.plot(swir_txt[:,0], swir_txt[:,4]/(swir_txt[:,5] - swir_txt[:,4] + 1e-5), label = "SNR SWIR Blue Panel")
plt.xlabel("Wavelength (nm)")
plt.ylabel("SNR (Mean/std)")
plt.grid("on")
plt.legend()
plt.savefig("/dirs/data/tirs/axhcis/Projects/NURI/Data/blue_panel_snr.pdf")


nano_txt = np.loadtxt("/dirs/data/tirs/axhcis/Projects/NURI/Data/nano_red_panel_radiance_spectra.txt", skiprows=7)
swir_txt = np.loadtxt("/dirs/data/tirs/axhcis/Projects/NURI/Data/swir_red_panel_radiance_spectra.txt", skiprows=7)

plt.figure(figsize = (7,7))
plt.plot(nano_txt[:,0], nano_txt[:,4]/(nano_txt[:,5] - nano_txt[:,4] + 1e-5), label = "SNR Nano Red Panel")
plt.plot(swir_txt[:,0], swir_txt[:,4]/(swir_txt[:,5] - swir_txt[:,4] + 1e-5), label = "SNR SWIR Red Panel")
plt.xlabel("Wavelength (nm)")

plt.ylabel("SNR (Mean/std)")
plt.grid("on")
plt.legend()
plt.savefig("/dirs/data/tirs/axhcis/Projects/NURI/Data/red_panel_snr.pdf")
