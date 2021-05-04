import numpy as np
from astropy.io.ascii import read 
from astropy.io import fits 
import sys 

params = {}

def toph_params():
	params['IMG_FILEPATH'] = '/Users/jadanso/Desktop/DRP_temp/COSMOS_55/COSMOS_55_Kb_4_mp_avg_bgsub.fits'
	params['IMG_PSFGRID'] = '/Users/jadanso/Desktop/jacq/input/COSMOS_55_Kb_4_mp_avg_bgsub_psfgrid.fits'
	params['STARS_FILEPATH'] = '/Users/jadanso/Desktop/jacq/input/COSMOS_55_Ks_cutout_stars.cat'
	params['REFERENCE_FILEPATH'] = '/Users/jadanso/Desktop/DRP_temp/COSMOS_55/COSMOS_55_Ks_cutout_psfmatched.fits'
	params['REFERENCE_PSFGRID'] = '/Users/jadanso/Desktop/jacq/input/COSMOS_55_Ks_cutout_psfmatched_psfgrid.fits'
	params['SAV_FILE'] = '/Users/jadanso/Desktop/jacq/input/COSMOS_55_Kb_4_mp_avg_bgsub.sav'
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


