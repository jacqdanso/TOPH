import numpy as np
from astropy.io import fits 
from toph_params import toph_params
from create_kernels import create_kernels
from convolve_image import convolve_image
from scipy.io import readsav
 
params = toph_params()

psf_points = readsav(params['SAV_FILE'])
xpoints = psf_points['spx'].flatten()
ypoints = psf_points['spy'].flatten()

img_filepath = params['IMG_FILEPATH']
img = fits.open(img_filepath)[0].data

img_psfs = fits.open(params['IMG_PSFGRID'])[0].data
ref_psfs = fits.open(params['REFERENCE_PSFGRID'])[0].data
regfact = params['REGFACT']

kernels = create_kernels(img_psfs=img_psfs, ref_psfs=ref_psfs, regfact=regfact)
buffer_size = np.shape(kernels[0])[0]
conv_type = params['CONVOLUTION_TYPE']

convol_img = convolve_image(params, img, img_filepath, buffer_size, kernels, xpoints, ypoints, conv_type)


