from spectral.io import envi
import numpy as np
import numpy.ma as ma
import os
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import minmax_scale
import matplotlib.cm as cm
import matplotlib.widgets as widgets
from scipy import interpolate
import pyelastix
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric, CCMetric
import dipy.align.imwarp as imwarp
from dipy.data import get_fnames
from dipy.io.image import load_nifti_data
from dipy.segment.mask import median_otsu
from dipy.viz import regtools
from scipy.ndimage import gaussian_filter
import matplotlib
import ipywidgets as widgets
from IPython.display import display


def load_image_envi(waterfall_path):
    vnir_ds = envi.open(waterfall_path)
    vnir_profile = vnir_ds.metadata
    vnir_arr = vnir_ds.load()

    return vnir_arr, vnir_profile

def callback_CC(sdr, status):
    # Status indicates at which stage of the optimization we currently are
    # For now, we will only react at the end of each resolution of the scale
    # space
    if status == imwarp.RegistrationStages.SCALE_END:
        # get the current images from the metric
        wmoving = sdr.metric.moving_image
        wstatic = sdr.metric.static_image
        # draw the images on top of each other with different colors
        regtools.overlay_images(wmoving, wstatic, 'Warped moving', 'Overlay',
                                'Warped static')

def main():
    global mica_patch, swir_patch, ax
    # load the SWIR image and select band 38
    or_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/SWIR/raw_1504_nuc_or_plusindices3.hdr"
    swir_arr, swir_profile= load_image_envi(or_hdr)
    swir_patch = swir_arr[1064:1181, 493:668, 37].squeeze()


    # load the micasense and select last band
    mica_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/Micasense/NURI_micasense_1133_transparent_mosaic_stacked_warped.hdr"
    mica_arr, mica_profile = load_image_envi(mica_hdr)
    mica_patch = mica_arr[2655:2772, 812:987, -1].squeeze()
    mica_patch = gaussian_filter(mica_patch, sigma=3)

    parameters = {
        'sigma_diff': 3.0,
        'radius': 2,
        'step_length': 1.0,
        'level_iters': [100, 50],
        'inv_iter': 50,
        'ss_sigma_factor': 0.1,
        'opt_tol': 1.e-1
    }

    # Your SymmetricDiffeomorphicRegistration setup here with default parameters
    metric = CCMetric(2, parameters['sigma_diff'], parameters['radius'])
    sdr = SymmetricDiffeomorphicRegistration(metric=metric,
                                             step_length=parameters['step_length'],
                                             level_iters=parameters['level_iters'],
                                             inv_iter=parameters['inv_iter'],
                                             ss_sigma_factor=parameters['ss_sigma_factor'],
                                             opt_tol=parameters['opt_tol'])
    mapping = sdr.optimize(mica_patch, swir_patch)
    swir_patch_warped = mapping.transform(swir_patch)

    fig, ax = plt.subplots(1, 3, figsize=(10, 8))
    ax[0].imshow(swir_patch)
    ax[0].axis("off")
    ax[1].imshow(mica_patch)
    ax[1].axis("off")



    def update_parameters(sigma_diff, radius, step_length, level_iters_1, level_iters_2, inv_iter, ss_sigma_factor,
                          opt_tol):
        global fig, ax
        parameters['sigma_diff'] = sigma_diff
        parameters['radius'] = radius
        parameters['step_length'] = step_length
        parameters['level_iters'] = [level_iters_1, level_iters_2]
        parameters['inv_iter'] = inv_iter
        parameters['ss_sigma_factor'] = ss_sigma_factor
        parameters['opt_tol'] = opt_tol
        metric = CCMetric(2, parameters['sigma_diff'], parameters['radius'])
        sdr = SymmetricDiffeomorphicRegistration(metric=metric,
                                                 step_length=parameters['step_length'],
                                                 level_iters=parameters['level_iters'],
                                                 inv_iter=parameters['inv_iter'],
                                                 ss_sigma_factor=parameters['ss_sigma_factor'],
                                                 opt_tol=parameters['opt_tol'])

        mapping = sdr.optimize(mica_patch, swir_patch)
        swir_patch_warped = mapping.transform(swir_patch)

        ax[2].imshow(swir_patch_warped)
        ax[2].axis("off")
        fig.canvas.draw()


    # Create sliders for each parameter using ipywidgets
    sigma_slider = widgets.FloatSlider(value=parameters['sigma_diff'], min=0.1, max=10.0, step=0.1, description='Sigma')
    radius_slider = widgets.IntSlider(value=parameters['radius'], min=1, max=10, step=1, description='Radius')
    step_length_slider = widgets.FloatSlider(value=parameters['step_length'], min=0.1, max=2.0, step=0.1,
                                             description='Step Length')
    level_iters_1_slider = widgets.IntSlider(value=parameters['level_iters'][0], min=10, max=200, step=10,
                                             description='Level 1')
    level_iters_2_slider = widgets.IntSlider(value=parameters['level_iters'][1], min=10, max=200, step=10,
                                             description='Level 2')
    inv_iter_slider = widgets.IntSlider(value=parameters['inv_iter'], min=10, max=200, step=10, description='Inv Iter')
    ss_sigma_factor_slider = widgets.FloatSlider(value=parameters['ss_sigma_factor'], min=0.01, max=1.0, step=0.01,
                                                 description='SS Sigma Factor')
    opt_tol_slider = widgets.FloatSlider(value=parameters['opt_tol'], min=0.01, max=1.0, step=0.01,
                                         description='Opt Tol')

    # Define the interactive function
    interactive_plot = widgets.interactive(update_parameters,
                                           sigma_diff=sigma_slider,
                                           radius=radius_slider,
                                           step_length=step_length_slider,
                                           level_iters_1=level_iters_1_slider,
                                           level_iters_2=level_iters_2_slider,
                                           inv_iter=inv_iter_slider,
                                           ss_sigma_factor=ss_sigma_factor_slider,
                                           opt_tol=opt_tol_slider)

    # Display the interactive plot
    display(interactive_plot)

    # sigma_diff = 3.0
    # radius = 2
    # metric = CCMetric(2, sigma_diff, radius)
    # sdr = SymmetricDiffeomorphicRegistration(metric=metric,
    #                                          step_length=1.0,
    #                                          level_iters=[100,50],
    #                                          inv_iter=50,
    #                                          ss_sigma_factor=0.1,
    #                                          opt_tol=1.e-1)
    #
    # sdr.callback = callback_CC
    #
    #
    # mapping = sdr.optimize(mica_patch, swir_patch)
    # # regtools.plot_2d_diffeomorphic_map(mapping, 4, 'diffeomorphic_map.png')
    # swir_patch_warped = mapping.transform(swir_patch)
    # fig, ax = plt.subplots(1,3, figsize = (10,8))
    # ax[0].imshow(swir_patch)
    # ax[0].axis("off")
    # ax[1].imshow(mica_patch)
    # ax[1].axis("off")
    # ax[2].imshow(swir_patch_warped)
    # ax[2].axis("off")
    # plt.show()

if __name__ == "__main__":

    """
    The problem with this registration is that it requires a symmetric registration, this is mainly used in meidcal imaging
    """
    main()







