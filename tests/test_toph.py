import unittest
import numpy as np
from astropy.io import fits 
from toph_scripts.toph_params import toph_params
from toph_scripts.create_kernels import create_kernels
from toph_scripts.convolve_image import convolve_image
from scipy.io import readsav

def test_toph():
	params = toph_params()
	params['SAVE_IMAGE'] = False
	params['CONVOLUTION_TYPE'] = 'fft'
	regfact = params['REGFACT']
	conv_type = params['CONVOLUTION_TYPE']

	psf_points = readsav(params['SAV_FILE'])
	xpoints = psf_points['spx'].flatten()
	ypoints = psf_points['spy'].flatten()

	img_filepath = params['IMG_FILEPATH']
	img = fits.open(img_filepath)[0].data

	img_psfs = fits.open(params['IMG_PSFGRID'])[0].data
	ref_psfs = fits.open(params['REFERENCE_PSFGRID'])[0].data

	assert len(img_psfs) == len(ref_psfs), "Number of reference and science PSFs should be the same"
	assert np.shape(img_psfs) == np.shape(ref_psfs), "Reference and science PSFs should have the same size"

	kernels = create_kernels(img_psfs=img_psfs, ref_psfs=ref_psfs, regfact=regfact)
	buffer_size = np.shape(kernels[0])[0]

	convol_img = convolve_image(params, img, img_filepath, buffer_size, kernels, xpoints, ypoints, conv_type)

