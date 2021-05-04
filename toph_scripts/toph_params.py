import numpy as np
from astropy.io.ascii import read 
from astropy.io import fits 
import sys 

params = {}

filepath = '/tests/input/'
def toph_params():
	params['IMG_FILEPATH'] = filepath+'COSMOS_352_Kb_0.80pp_sci.fits'
	params['IMG_PSFGRID'] = filepath+'COSMOS_352_Kb_psfgrid_v0.8.fits'
	params['STARS_FILEPATH'] = filepath+'COSMOS_352_Kb_0.80pp_stars_green2.txt'
	params['REFERENCE_FILEPATH'] = filepath+'COSMOS_352_Ks_cutout.fits'
	params['REFERENCE_PSFGRID'] = filepath+'COSMOS_352_Ks_psfgrid_v0.8.fits'
	params['SAV_FILE'] = filepath+'COSMOS_352_Kb_0.80pp_sci.sav'
	params['PIXEL_SCALE'] = 0.15
	params['CHECK_OFFSET_BEFORE'] = False 
	params['CHECK_OFFSET_AFTER'] = False 
	params['KERNEL_METHOD'] = 'FFD'
	params['REGFACT'] = 1e-2
	params['USE_MODEL_REFERENCE'] = False 
	params['CHECK_KERNEL'] = False 
	params['SHOW_GRID'] = False 
	params['SHOW_CONVOLVED_IMAGE'] = True 
	params['SHOW_GROWTHCURVES'] = False 
	params['APERTURE_SIZE'] = 28*params['PIXEL_SCALE']
	params['INNER_ANNULUS'] = 32*params['PIXEL_SCALE']
	params['OUTER_ANNULUS'] = 51*params['PIXEL_SCALE']
	params['SAVE_IMAGE'] = True
	params['CONVOLUTION_TYPE'] = 'convolve2d'

	# Checks 
	img_psfs = fits.open(params['IMG_PSFGRID'])[0].data
	ref_psfs = fits.open(params['REFERENCE_PSFGRID'])[0].data

	if np.shape(img_psfs) != np.shape(ref_psfs):
		print('Different number of science and reference PSFs OR they have different shapes')
		sys.exit()
	else:
		return params


