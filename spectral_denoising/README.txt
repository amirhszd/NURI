the main method for performing spectral denoising is spectral_denoising_multiprocessing_mmap.py,
which uses memory maps to access parts of the large array, this makes sure there isnt wasted
time putting everyhting into memory all at once.

disregard spectral_denoising_multiprocesing.py, that was the basis for the main method.