from pypher import pypher 
from scipy.ndimage import shift
from astropy.io import fits 
from photutils import SplitCosineBellWindow, create_matching_kernel

def create_kernels(img_psfs, ref_psfs, file_basename, params, save_kernels = True):
    print('>>> Creating kernels')
    if params['KERNEL_METHOD'] == 'matrix_reg':
        kernels = [pypher.homogenization_kernel(reference_psf, img_psf, reg_fact=params['REGFACT'])[0] \
              for  reference_psf, img_psf  in  zip(ref_psfs, img_psfs)]

    elif params['KERNEL_METHOD'] == 'ffd':
        window = SplitCosineBellWindow(alpha=params['ALPHA'],beta=params['BETA'])
        kernels = [create_matching_kernel(img_psf, ref_psf, window=window) for img_psf, ref_psf in \
                    zip(img_psfs, ref_psfs)]

    shifted_kernels = [shift(ker, [-1, -1], mode = 'constant') for ker in kernels] 

    if save_kernels:
        hdu_shifted = fits.PrimaryHDU(shifted_kernels)
        hdu_shifted.header = fits.open(params['IMG_FILEPATH'])[0].header
        shifted_save_string = params['OUTDIR']+file_basename+'_shifted_kernels.fits'

        hdu = fits.PrimaryHDU(kernels)
        hdu.header = fits.open(params['IMG_FILEPATH'])[0].header
        save_string = params['OUTDIR']+file_basename+'_kernels.fits'

        hdu_shifted.writeto(shifted_save_string, overwrite = True)
        hdu.writeto(save_string, overwrite = True)

        print('>>> Kernels will be saved here: '+save_string)
        print('>>> Shifted Kernels will be saved here: '+shifted_save_string)

    return kernels, shifted_kernels






