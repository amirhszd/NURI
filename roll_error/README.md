# SPLASH: Spatial eLAstic Harmonization Technique
<img alt="Alt text" height="300" src="static/logo1.png" width="300"/>
This repository contains the implementation of SPLASH, a Spatial eLAstic Harmonization technique developed for spatial harmonization and registration of hyperspectral images.

## Overview
SPLASH is designed to harmonize VNIR, SWIR, and Micasense hyperspectral data. It involves the following steps:
1. Upsampling the Micasense data to match the VNIR and SWIR resolutions.
2. Warping VNIR and SWIR to match the Micasense data.
3. Setting zero values in the Micasense image based on zero values in the hyperspectral image.
4. Running shape-shifter algorithms on the datasets.
5. Running Antspy registration for precise alignment.

## Prerequisites
- Python 3.7+
- Main required Python libraries:
  - rasterio
  - spectral.io.envi
  - GDAL

## Installation
if want to install independently, set up your environment using conda, install gdal using ```conda install gdal```
```bash
pip install -r requirements.txt
```

## Author
Amirhossein Hassanzadeh (axhcis@rit.edu)
