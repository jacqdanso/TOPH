import numpy as np
from astropy.io import fits 
from toph_params import toph_params
from diagnostics import *
from create_kernels import create_kernels
from convolve_image import convolve_image
from scipy.io import readsav
from astropy.io.ascii import read 
 
params = toph_params()

psf_points = readsav(params['SAV_FILE']) 
xpoints = psf_points['spx'].flatten()
ypoints = psf_points['spy'].flatten()

file_basename = params['IMG_FILEPATH'].split('/')[-1].split('.fits')[0]
img = fits.open(params['IMG_FILEPATH'])[0].data

img_psfs = fits.open(params['IMG_PSFGRID'])[0].data
ref_psfs = fits.open(params['REFERENCE_PSFGRID'])[0].data

star_cat = read(params['STARS_FILEPATH'])

if not params['DIAGNOSTICS_ONLY']:

    # run TOPH -- main routines

    kernels, shifted_kernels = create_kernels(img_psfs=img_psfs, ref_psfs=ref_psfs, file_basename=file_basename, \
                                                params=params)
    # set size of padding for image slices
    buffer_size = np.shape(kernels[0])[0]

    convol_img = convolve_image(params, img, buffer_size, shifted_kernels, xpoints, ypoints, file_basename)
 
else:
    kernels = fits.open(params['OUTDIR']+file_basename+'_kernels.fits')[0].data
    shifted_kernels = fits.open(params['OUTDIR']+file_basename+'_shifted_kernels.fits')[0].data
    # set size of padding for image slices
    buffer_size = np.shape(kernels[0])[0]


# make diagnostic plots 
if params['CHECK_KERNELS']: 
    check_kernels(kernels=shifted_kernels, img_psfs=img_psfs, ref_psfs=ref_psfs, \
        file_basename = file_basename, params = params)

if params['CHECK_CONVOLVED_PSFS']:
    check_convolved_psfs(kernels = shifted_kernels, img_psfs = img_psfs, ref_psfs = ref_psfs, stars=star_cat, \
        file_basename = file_basename, params = params)

if params['SHOW_CONVOLVED_IMAGE']:
    convol_img = fits.open(params['OUTDIR']+file_basename+'_psfmatched.fits')[0].data
    check_convolved_image(convol_img=convol_img, file_basename = file_basename, outdir = params['OUTDIR'])

if params['SHOW_GRID']:
    show_image_grid(img, xpoints, ypoints, params, file_basename, buffer_size)

if params['SHOW_STAR_STAMPS']:
    convol_img = fits.open(params['CONVOL_IMG_FILEPATH'])[0].data
    ref_img = fits.open(params['REFERENCE_FILEPATH'])[0].data
    show_star_stamps(convol_img, ref_img, star_cat, file_basename, params, verbose=False)





