from pypher import pypher 
from scipy.ndimage import shift

def create_kernels(img_psfs, ref_psfs, regfact):
	print('Creating kernels')
	kernels = [pypher.homogenization_kernel(reference_psf, img_psf, reg_fact=regfact)[0] \
              for  reference_psf, img_psf  in  zip(ref_psfs, img_psfs)]

	shifted_kernels = [shift(ker, [-1, -1], mode = 'constant') for ker in kernels] 

	return kernels, shifted_kernels






