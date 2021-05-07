import numpy as np
from astropy.io.ascii import read 
from astropy.io import fits 
import sys 

params = {}

#filepath = '/Users/jadanso/Desktop/TOPH/tests/input/'
filepath = '/tests/input/'
def toph_params():
	params['IMG_FILEPATH'] = filepath+'COSMOS_55_Kb_4_mp_avg_bgsub.fits'
	params['IMG_PSFGRID'] = filepath+'COSMOS_55_Kb_4_mp_avg_bgsub_psfgrid.fits'
	params['STARS_FILEPATH'] = filepath+'COSMOS_55_Ks_cutout_stars.cat'
	params['REFERENCE_FILEPATH'] = filepath+'COSMOS_55_Ks_cutout.fits'
	params['REFERENCE_PSFGRID'] = filepath+'COSMOS_55_Ks_cutout_psfgrid.fits'
	params['SAV_FILE'] = filepath+'COSMOS_55_Kb_4_mp_avg_bgsub.sav'
	params['PIXEL_SCALE'] = 0.15
	params['KERNEL_METHOD'] = 'FFD'
	params['REGFACT'] = 1e-2
	params['USE_MODEL_REFERENCE'] = False 
	params['CHECK_OFFSET_BEFORE'] = False 
	params['CHECK_OFFSET_AFTER'] = False 
	params['CHECK_KERNELS'] = True 
	params['CHECK_CONVOLVED_PSFS'] = True 
	params['SHOW_GRID'] = False 
	params['SHOW_CONVOLVED_IMAGE'] = True 
	params['SHOW_GROWTHCURVES'] = False 
	params['APERTURE_SIZE'] = 28*params['PIXEL_SCALE']
	params['INNER_ANNULUS'] = 32*params['PIXEL_SCALE']
	params['OUTER_ANNULUS'] = 51*params['PIXEL_SCALE']
	#params['CATALOG_APERTURE'] = 0.6 #in arcseconds
	params['SAVE_IMAGE'] = True
	params['CONVOLUTION_TYPE'] = 'convolve2d'
	params['OUTDIR'] = 'output/'

	return params


