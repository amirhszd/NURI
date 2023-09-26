import tifffile
import matplotlib.pyplot as plt
import numpy as np
import cmocean

def plot_imagefile(filename):

    import os
    output_folder = filename.split(".")[0] + "bands"
    os.makedirs(output_folder, exist_ok=True)

    image = tifffile.imread(filename)
    if len(image.shape) == 3:
        n_bands = image.shape[2]
    else:
        n_bands = 1

    for band in range(n_bands):
        fig, ax = plt.subplots(1,1, figsize=(10,10))

        ax.imshow(image[...,band], cmap = cmocean.cm.thermal)
        ax.set_title(f"band {band + 1}")
        ax.axis("off")
        plt.tight_layout()

        plt.savefig(os.path.join(output_folder, f"band_{band + 1}.pdf"), dpi = 300)



