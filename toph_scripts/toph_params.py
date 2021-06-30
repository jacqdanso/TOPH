import numpy as np
from astropy.io.ascii import read 
from astropy.io import fits 
import sys 

params = {}

#filepath = '/Users/jadanso/Desktop/TOPH/tests/input/'
filepath = 'tests/input/'

def toph_params():

	# Input data
	params['IMG_FILEPATH'] = filepath+'COSMOS_55_Kb_4_mp_avg_bgsub.fits'
	params['IMG_PSFGRID'] = filepath+'COSMOS_55_Kb_4_mp_avg_bgsub_psfgrid.fits'
	params['STARS_FILEPATH'] = filepath+'COSMOS_55_Kb_4_mp_avg_bgsub_stars_green.cat'
	params['REFERENCE_FILEPATH'] = filepath+'COSMOS_55_Ks_cutout.fits'
	params['REFERENCE_PSFGRID'] = filepath+'COSMOS_55_Ks_cutout_psfgrid.fits'
	params['SAV_FILE'] = filepath+'COSMOS_55_Kb_4_mp_avg_bgsub.sav'

	# Diagnostic plots 
	params['CHECK_OFFSET_BEFORE'] = False 
	params['CHECK_OFFSET_AFTER'] = False 
	params['CHECK_KERNELS'] = True
	params['CHECK_CONVOLVED_PSFS'] = False
	params['SHOW_GRID'] = False 
	params['SHOW_CONVOLVED_IMAGE'] = False 
	params['SHOW_STAR_STAMPS'] = False
	params['SAVE_IMAGE'] = True

	# Kernels 
	params['KERNEL_METHOD'] = 'ffd' # options: 'ffd, matrix_reg'
	params['ALPHA'] = 0.4 # alpha parameter of split cosine bell window
	params['BETA'] = 0.3 # beta parameter of split cosine bell window
	params['REGFACT'] = 1e-2
	params['USE_MODEL_REFERENCE'] = False 
	
	# Growth curves and background subtraction
	params['PIXEL_SCALE'] = 0.15
	params['APERTURE_SIZE'] = 28*params['PIXEL_SCALE']
	params['INNER_ANNULUS'] = 32*params['PIXEL_SCALE']
	params['OUTER_ANNULUS'] = 51*params['PIXEL_SCALE']
	params['CATALOG_APERTURE'] = 0.6 #in arcseconds
	
	params['CONVOLUTION_TYPE'] = 'fft' # options : 'fft, convolve_2d'
	params['OUTDIR'] = 'output/'

	# If running diagnostics only
	params['DIAGNOSTICS_ONLY'] = False
	params['CONVOL_IMG_FILEPATH'] = params['OUTDIR']+'COSMOS_55_Kb_4_mp_avg_bgsub_psfmatched.fits'

	return params


