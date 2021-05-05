import matplotlib.pyplot as plt 
from astropy.convolution import convolve_fft
import pidly

plt.style.use('seaborn-poster')
kwargs = {'vmin' : -1e-3, 'vmax' : 1e-3, 'cmap' : 'Greys_r', 'origin' : 'lower'}

def convolve_psfs(kernels, img_psfs, refpsfs):
	convol_img_psfs = [convolve_fft(chunk, kernel, boundary='fill', mode='same') \
                  for chunk, kernel in zip(img_psfs, kernels)]
    return convol_img_psfs

def check_kernels(kernels, img_psfs, refpsfs, file_basename):
	convol_img_psfs = convolve_psfs(kernels, img_psfs, refpsfs)

	fig, ax = plt.subplots(len(img_psfs), 5, figsize = (8,180), sharex=True, sharey=True, \
                       gridspec_kw={'hspace': 0, 'wspace': 0})

	ax[0,0].text(5,80, '(1) Science PSF', fontsize = 10, c = 'blue')
	ax[0,1].text(5,80, '(2) Reference PSF', fontsize = 10, c='blue')
	ax[0,2].text(5,80, '(3) Kernel', fontsize = 10, c = 'blue')
	ax[0,3].text(5,80, '(4) Convolved', fontsize = 10, c = 'blue')
	ax[0,4].text(5,80, 'Residual: (4)-(2)', fontsize = 10, c = 'blue')

	for i in range(len(convol_img_psfs)):
    	ax[i,0].imshow(img_psfs[i], **kwargs)
    	ax[i,1].imshow(ref_psfs[i], **kwargs)
    	ax[i,2].imshow(kernels[i], **kwargs)
    	ax[i,3].imshow(convol_img_psfs[i], **kwargs)
    	ax[i,4].imshow(convol_img_psfs[i]-ref_psfs[i], **kwargs)

    fig.text(0.5, 0.01, 'x (pixels)', ha='center', fontsize = 15)
	fig.text(0.04, 0.5, 'y (pixels)', va='center', rotation='vertical', fontsize = 15)
	plt.savefig(file_basename+'_kernels.png')

	return 

def check_convolved_image(convol_img):
	plt.figure(figsize = (10,10))
	plt.imshow(convol_img, **kwargs)
	plt.xlabel('x (pixels)', fontsize = 30)
	plt.ylabel('y (pixels)', fontsize = 30)
	plt.savefig(file_basename+'_psfmatched.png')

def check_convolved_psfs(ref_psfs, stars, pixscale):
	convol_img_psfs = convolve_psfs(kernels, img_psfs, refpsfs)
	bgsub_img_psfs = [bgsub(stamp, id_i, rin = 30, rout = 50)[0] for stamp, id_i \
                in zip(convol_img_psfs, stars['id'])]
	bgsub_ref_psfs = [bgsub(stamp, id_i, rin = 30, rout = 50)[0] for stamp, id_i \
                in zip(ref_psfs, stars['id'])]

    rnorm = 4 # in arcseconds
	raper = np.linspace(0.5, rnorm/pixscale, num=100)

	idl = pidly.IDL()

	ratio_gc = []; ref_gcs = []; img_gcs = []

	for i in range(len(kb_stars)):
    	img_gc = idl.func('growthcurve', bgsub_img_psfs[i], rnorm=True, raper = raper)  
    	ref_gc = idl.func('growthcurve', bgsub_ref_psfs[i], rnorm=True, raper = raper)
    	ratio = img_gc/ref_gc
    	ref_gcs.append(ref_gc)
    	img_gcs.append(img_gc)
    	ratio_gc.append(ratio)

    fig, ax = plt.subplots(1,2, figsize = (15, 6))
	ax = ax.ravel()

	for i in range(len(ref_gcs)):
    	ax[0].plot(raper*pixscale, ref_gcs[i], c='green', alpha = 0.5, \
    		label = 'Reference')
    	ax[0].plot(raper*pixscale, img_gcs[i], c='blue', alpha = 0.8, \
    		label = 'Science')
    	ax[0].axhline(1.0, color='k', linestyle = '-', lw = 0.5)
    	ax[0].axhline(1.01, color='k', linestyle = '--', lw = 0.5)
    	ax[0].axhline(0.99, color='k', linestyle = '--', lw = 0.5)
    	ax[0].axvline(1.5, c='red', ls='dashed')
    	ax[0].set_ylabel('curve of growth ', fontsize = 20)
    	ax[0].set_xlabel('radius (arcsec)')

	for i in range(len(ratio_gc)):
    	ax[1].plot(raper*pixscale, ratio_gc[i], c = 'gray')
    	ax[1].set_ylim(0.97,1.03)
    	ax[1].axhline(1.0, color='k', linestyle = '-', lw = 0.5)
    	ax[1].axhline(1.01, color='k', linestyle = '--', lw = 0.5)
    	ax[1].axhline(0.99, color='k', linestyle = '--', lw = 0.5)
    	ax[1].axvline(1.5, c='black', ls='dashed')
    	ax[1].set_ylabel('curve of growth / reference', fontsize = 20)
    	ax[1].set_xlabel('radius (arcsec)')

    plt.savefig(file_basename+'_psf_growthcurves.png')

	
   




