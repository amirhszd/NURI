import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nano_txt = np.loadtxt("/dirs/data/tirs/axhcis/Projects/NURI/Data/nano_blue_panel_reflectance_spectra.txt", skiprows=7)
swir_txt = np.loadtxt("/dirs/data/tirs/axhcis/Projects/NURI/Data/swir_blue_panel_reflectance_spectra.txt", skiprows=7)
swir_txt[:,4][swir_txt[:,4]>1] = 1
swir_txt[:,4][swir_txt[:,4]<0] = 0


plt.figure(figsize = (7,7))
plt.plot(nano_txt[:,0], nano_txt[:,4], label = "Mean Nano Blue Panel")
plt.plot(swir_txt[:,0], swir_txt[:,4], label = "Mean SWIR Blue Panel")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.grid("on")
plt.legend()
plt.savefig("/dirs/data/tirs/axhcis/Projects/NURI/Data/blue_panel_reflectance_difference.pdf")


nano_txt = np.loadtxt("/dirs/data/tirs/axhcis/Projects/NURI/Data/nano_red_panel_reflectance_spectra.txt", skiprows=7)
swir_txt = np.loadtxt("/dirs/data/tirs/axhcis/Projects/NURI/Data/swir_red_panel_reflectance_spectra.txt", skiprows=7)
swir_txt[:,4][swir_txt[:,4]>1] = 1
swir_txt[:,4][swir_txt[:,4]<0] = 0

import matplotlib.pyplot as plt

plt.figure(figsize = (7,7))
plt.plot(nano_txt[:,0], nano_txt[:,4], label = "Mean Nano Red Panel")
plt.plot(swir_txt[:,0], swir_txt[:,4], label = "Mean SWIR Red Panel")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.grid("on")
plt.legend()
plt.savefig("/dirs/data/tirs/axhcis/Projects/NURI/Data/red_panel_reflectance_difference.pdf")




