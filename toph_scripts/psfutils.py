import numpy as np 
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from median_fit import median_fit
from astropy.io import fits
from photutils import detect_threshold, detect_sources
from astropy.convolution import Gaussian2DKernel
from matplotlib.patches import Circle

# find the area of the photometry circle 
def photometry_area(img, rout):
	center = int(np.shape(img)[0]/2)
	imin = center - rout
	imax = center + rout 
	jmin = center - rout
	jmax = center + rout
    
	phot_vals = []
	for i in np.arange(imin, imax):
	    for j in np.arange(jmin, jmax):
	        ij = np.array([i,j])
	        dist = np.linalg.norm(ij - np.array(center))
	        if dist <= rout:
	            phot_vals.append([i,j,img[i][j]])
	phot_area = len(phot_vals)

	return phot_area 

def detect_objects(image):
    threshold = detect_threshold(image, nsigma=3)
    sigma = 3.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # FWHM = 3
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    segm = detect_sources(image, threshold, npixels=5,
                      filter_kernel=kernel)
    
    return segm

def bgsub(img, idx, rin, rout, display = False):
    center = int(np.shape(img)[0]/2)
    imin = center - rout
    imax = center + rout 
    jmin = center - rout
    jmax = center + rout 
    
    segm = detect_objects(img)
    masked_img_obj = np.ma.masked_array(img, ~segm.data_ma.mask)
    mask = masked_img_obj.mask
    masked_img = np.copy(img)
    masked_img[mask]= -99
    
    vals = []
    for i in np.arange(imin, imax):
        for j in np.arange(jmin, jmax):
            ij = np.array([i,j])
            dist = np.linalg.norm(ij - np.array(center))
            if dist > rin and dist <= rout:
                if not mask[i][j]:
                    vals.append([i,j,masked_img[i][j]])  
                             
    vals = np.array(vals)        

    x = np.array([int(xpix) for xpix in vals[:,1]])
    y = np.array([int(ypix) for ypix in vals[:,0]])
    
    skyvals = vals[:,2]
    skyvals = skyvals[~np.isnan(skyvals)] # remove NaNs (for stars on the edge]
    
    # sigma clipping
    mode, min_num, max_num = median_fit(skyvals, idx)

    sky = mode
    
    if display == True: 
        # diagnostic plots 
        fig, ax = plt.subplots(1, 3, figsize=(20,5))
        ax = ax.ravel()
        ax[0].imshow(segm.data, interpolation='nearest')
        
        ax[1].imshow(img, cmap = 'Greys_r', vmin=-1E-3, vmax=1E-3)
        ax[1].text(20, 20,  str(idx), color='blue', fontsize = 25)
        circ = Circle((center,center),4.0/0.15, fill = False, linestyle = '--', color = 'white', lw=2)
        ax[1].add_patch(circ)
        phot_area = photometry_area(img)
        ax[1].set_title('annulus area: '+str(len(vals)/phot_area)+'x phot_area')
        
        for xi, yi in zip(x, y):
            ax[1].plot(xi, yi, '.', c='red')
            
        ax[2].hist(skyvals, bins = 30)
        ax[2].axvline(sky, ls='dashed', c='black')
        ax[2].axvline(min_num, ls='dashed', c='green')
        ax[2].axvline(max_num, ls='dashed', c='green')
        ax[2].set_xlabel('Sky Value (counts)')
        ax[2].set_ylabel('Number of pixels')
        
    print('Sky Value:', sky, ' counts (normalized)')
    bg_sub_img = img - sky
    return [bg_sub_img, sky]