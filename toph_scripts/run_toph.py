import numpy as np
from astropy.io import fits 
from toph_params import toph_params
from diagnostics import *
from create_kernels import create_kernels
from convolve_image import convolve_image
from scipy.io import readsav
from photometry.psfutils import fit_moffat, peak2d, bgsub
 
params = toph_params()

psf_points = readsav(params['SAV_FILE']) 
xpoints = psf_points['spx'].flatten()
ypoints = psf_points['spy'].flatten()

file_basename = params['IMG_FILEPATH'].split('/')[-1].split('.fits')[0]
img = fits.open(params['IMG_FILEPATH'])[0].data

img_psfs = fits.open(params['IMG_PSFGRID'])[0].data
ref_psfs = fits.open(params['REFERENCE_PSFGRID'])[0].data

star_cat = read(params['STARS_FILEPATH'])

# run TOPH -- main routines

kernels = create_kernels(img_psfs=img_psfs, ref_psfs=ref_psfs, regfact=params['REGFACT'])
buffer_size = np.shape(kernels[0])[0]

convol_img = convolve_image(params, img, params['IMG_FILEPATH'], buffer_size, kernels, xpoints, ypoints, \
	params['CONVOLUTION_TYPE'], file_basename)

# make diagnostic plots 

if params['CHECK_KERNELS']: 
	check_kernels(kernels=kernels, img_psfs=img_psfs, ref_psfs=ref_psfs, \
		file_basename = file_basename)

if params['CHECK_CONVOLVED_PSFS']:
	check_convolved_psfs(ref_psfs = ref_psfs, stars=star_cat, pixscale = params['PIXEL_SCALE'])
	
if params['SHOW_CONVOLVED_IMAGE']:
	convol_img = fits.open(file_basename+'_psfmatched.fits')[0].data
	check_convolved_image(convol_img)




