import matplotlib.pyplot as plt 
from astropy.convolution import convolve_fft
from scipy.signal import convolve2d
import pidly
from psfutils import bgsub
import numpy as np
from convolve_image import slice_image
from datetime import datetime
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
import sys


plt.style.use('seaborn-poster')
kwargs = {'vmin' : -1e-3, 'vmax' : 1e-3, 'cmap' : 'Greys_r', 'origin' : 'lower'}

def save_toph_figure(filename):
    try:
        plt.savefig(filename)
    except OSError:
        datetime_tag = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        plt.savefig(filename.split('.png')[0]+datetime_tag+'.png')

def convolve_psfs(kernels, img_psfs, params):
    if params['CONVOLUTION_TYPE'] == 'convolve2d':
        convol_img_psfs = [convolve2d(chunk, kernel, boundary='fill', mode = 'same') \
                  for chunk, kernel in zip(img_psfs, kernels)]
    elif params['CONVOLUTION_TYPE'] == 'fft':
        convol_img_psfs = [convolve_fft(chunk, kernel, boundary='fill') \
                  for chunk, kernel in zip(img_psfs, kernels)]                
    return convol_img_psfs

def check_kernels(kernels, img_psfs, ref_psfs, file_basename, params):
    print('>>> Visually inspecting PSF and kernel quality')
    convol_img_psfs = convolve_psfs(kernels, img_psfs, params)

    fig, ax = plt.subplots(len(img_psfs[:3]), 5, figsize = (10,5), sharex=True, sharey=True, \
                       gridspec_kw={'hspace': 0, 'wspace': 0})

    ax[0,0].text(5,80, '(1) Science PSF', fontsize = 10, c = 'blue')
    ax[0,1].text(5,80, '(2) Reference PSF', fontsize = 10, c='blue')
    ax[0,2].text(5,80, '(3) Kernel', fontsize = 10, c = 'blue')
    ax[0,3].text(5,80, '(4) Convolved', fontsize = 10, c = 'blue')
    ax[0,4].text(5,80, 'Residual: (4)-(2)', fontsize = 10, c = 'blue')

    for i in range(len(convol_img_psfs[:3])):
        ax[i,0].imshow(img_psfs[i], **kwargs)
        ax[i,1].imshow(ref_psfs[i], **kwargs)
        ax[i,2].imshow(kernels[i], **kwargs)
        ax[i,3].imshow(convol_img_psfs[i], **kwargs)
        ax[i,4].imshow(convol_img_psfs[i]-ref_psfs[i], **kwargs)

    fig.text(0.5, 0.005, 'x (pixels)', ha='center', fontsize = 15)
    fig.text(0.04, 0.5, 'y (pixels)', va='center', rotation='vertical', fontsize = 15)
    plt.suptitle('Only displaying the first three rows')
    print('>>> Kernel check plots will be saved here: '+params['OUTDIR']+file_basename+'_kernels.png')
    save_toph_figure(params['OUTDIR']+file_basename+'_kernels.png')


def check_convolved_image(convol_img, file_basename, outdir):
    plt.figure(figsize = (10,10))
    plt.imshow(convol_img, **kwargs)
    plt.xlabel('x (pixels)', fontsize = 30)
    plt.ylabel('y (pixels)', fontsize = 30)
    print('>>> Convolved image plot will be saved here: '+outdir+file_basename+'_psfmatched.png')
    save_toph_figure(outdir+file_basename+'_psfmatched.png')

def check_convolved_psfs(kernels, img_psfs, ref_psfs, stars, file_basename, params):
    print('>>> Checking kernel quality using growthcurves of convolved PSFs')
    convol_img_psfs = convolve_psfs(kernels, img_psfs, params)

    print('>>> Subtracting background of PSF stamps')
    bgsub_img_psfs = [bgsub(stamp, id_i, rin = 30, rout = 50, display = False)[0] for stamp, id_i \
                in zip(convol_img_psfs, stars['id'])]
    bgsub_ref_psfs = [bgsub(stamp, id_i, rin = 30, rout = 50, display = False)[0] for stamp, id_i \
                in zip(ref_psfs, stars['id'])]

    rnorm = 4 # in arcseconds
    raper = np.linspace(0.5, rnorm/params['PIXEL_SCALE'], num=100)

    idl = pidly.IDL()

    ratio_gc = []; ref_gcs = []; img_gcs = []

    print('>>> Measuring growthcurves')
    for i in range(len(stars)):
        img_gc = idl.func('growthcurve', bgsub_img_psfs[i], rnorm=True, raper = raper)  
        ref_gc = idl.func('growthcurve', bgsub_ref_psfs[i], rnorm=True, raper = raper)
        ratio = img_gc/ref_gc
        ref_gcs.append(ref_gc)
        img_gcs.append(img_gc)
        ratio_gc.append(ratio)

    fig, ax = plt.subplots(1,2, figsize = (15, 6))
    ax = ax.ravel()

    for i in range(len(ref_gcs)):
        ax[0].plot(raper*params['PIXEL_SCALE'], ref_gcs[i], c='green', alpha = 0.5, \
            label = 'Reference')
        ax[0].plot(raper*params['PIXEL_SCALE'], img_gcs[i], c='blue', alpha = 0.5, \
            label = 'Science')
        ax[0].axhline(1.0, color='k', linestyle = '-', lw = 0.5)
        ax[0].axhline(1.01, color='k', linestyle = '--', lw = 0.5)
        ax[0].axhline(0.99, color='k', linestyle = '--', lw = 0.5)
        ax[0].axvline(1.5, c='red', ls='dashed')
        ax[0].set_ylabel('curve of growth ', fontsize = 20)
        ax[0].set_xlabel('radius (arcsec)')

    ax[0].legend()

    for i in range(len(ratio_gc)):
        ax[1].plot(raper*params['PIXEL_SCALE'], ratio_gc[i], c = 'gray')
        ax[1].set_ylim(0.97,1.03)
        ax[1].axhline(1.0, color='k', linestyle = '-', lw = 0.5)
        ax[1].axhline(1.01, color='k', linestyle = '--', lw = 0.5)
        ax[1].axhline(0.99, color='k', linestyle = '--', lw = 0.5)
        ax[1].axvline(1.5, c='black', ls='dashed')
        ax[1].set_ylabel('curve of growth / reference', fontsize = 20)
        ax[1].set_xlabel('radius (arcsec)')

    print('>>> Convolved PSF plots will be saved here: '+params['OUTDIR']+file_basename+'_psf_growthcurves.png')
    save_toph_figure(params['OUTDIR']+file_basename+'_psf_growthcurves.png')

def get_eqdist(x1, x2, y1, y2):
    x3 = (x2**2 - x1**2)/(2*x2 - 2*x1)
    num = y2**2 - y1**2 + (x3 - x2)**2 - (x3 - x1)**2
    denom = 2*y2 - 2*y1
    y3 = num/denom
    return x3, y3

def show_image_grid(img, xpoints, ypoints, params, file_basename, buffer_size):
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap = 'Greys_r', alpha = 0.3, origin = 'lower')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')

    slice_num, xpts, ypts, grid = slice_image(img, xpoints, ypoints, buffer_size, diagnostics = True)

    midpoints = []
    for i in range(len(xpts)-1):
        for j in range(len(ypts)-1):
            x1 = xpts[i]; y1 = ypts[j]
            x2 = xpts[i+1]; y2 = ypts[j+1]
     
        if np.logical_and((x2-x1) > 1, (y2-y1 > 1)):
            x3, y3 = get_eqdist(x1, x2, y1, y2) 
            midpoints.append([x3,y3])

    #for i in range(len(midpoints)):
        #plt.text(midpoints[i][0]-30, midpoints[i][1]-10, str(slice_num[i]), c='k')

    for i in range(len(xpoints)):
        plt.plot(xpoints[i], ypoints[i], 'o', c='brown')

    plt.plot(xpoints[0], ypoints[0], 'o', c = 'brown', label = 'PSF grid points')

    top = np.linspace(0, np.shape(img)[1], num=(grid[1]*2)-1)
    edge = np.linspace(0, np.shape(img)[0], num=(grid[0]*2)-1)

    for i in range(len(top)):
        plt.axvline(top[i], ymax=0.95, c = 'brown', ls = 'dashed')
    for i in range(len(edge)):
        plt.axhline(edge[i], xmax=0.95, c='brown', ls = 'dashed')

    plt.legend()
    print('>>> Image grid will be saved here: '+params['OUTDIR']+file_basename+'_grid.png')
    save_toph_figure(params['OUTDIR']+file_basename+'_grid.png')

def show_star_stamps(convol_img, ref_img, stars, file_basename, params, verbose = False):
    img_stamps = [Cutout2D(convol_img, (np.array(stars['x'])[i],np.array(stars['y'])[i]),\
                       (110,110)).data for i in range(len(stars))]
    try:
        ref_stamps = [Cutout2D(ref_img, (np.array(stars['x'])[i],np.array(stars['y'])[i]),\
                       (110, 110)).data for i in range(len(stars))]
    except NoOverlapError:
        print("Reference and science image should have the same x y grid")
        sys.exit()

    for img, ref in zip(img_stamps, ref_stamps):
        img[np.isnan(img)] = -99
        ref[np.isnan(ref)] = -99

    idl = pidly.IDL() 

    cstamps = [idl.func('center', stamp, cubic = True, missing=0) \
              for stamp in np.array(img_stamps)]

    crefstamps = [idl.func('center', stamp, cubic = True, missing=0) \
               for stamp in np.array(ref_stamps)]

    for img, ref in zip(cstamps, crefstamps):
        img[img < - 98] = np.nan
        ref[ref < - 98] = np.nan

    cnorm_stamps = [img/np.sum(img[~np.isnan(img)]) for img in cstamps]
    cnorm_refs = [stamp/np.sum(stamp[~np.isnan(stamp)]) for stamp in crefstamps]

    bgsub_imgs = [bgsub(stamp, id_i, rin = 30, rout = 50, verbose = verbose)[0] for stamp, id_i \
                in zip(cnorm_stamps, stars['id'])]
    bgsub_refs = [bgsub(stamp, id_i, rin = 30, rout = 50, verbose = verbose)[0] for stamp, id_i \
                in zip(cnorm_refs, stars['id'])]
   

    fig, ax = plt.subplots(len(stars[:4]),4, figsize = (8,10), sharex=True, sharey=True, \
                       gridspec_kw={'hspace': 0, 'wspace': 0})
    color = 'lavender'
    
    ax[0,0].text(15,100, 'Science (1)', color=color, fontsize = 12)
    ax[0,1].text(15,100,'Reference (2)', color=color, fontsize = 12)
    ax[0,2].text(15,100,'Residual (2-1)',  color=color, fontsize = 12)
    ax[0,3].text(2,100,'Residual \n (centered)',  color=color, fontsize = 12)

    plt.suptitle('Only showing the first four stars')

    for i in range(len(stars[:4])):
        img_vmin = np.mean(bgsub_imgs[i])-np.std(bgsub_imgs[i])
        img_vmax = np.mean(bgsub_imgs[i])+np.std(bgsub_imgs[i])

        ref_vmin = np.mean(bgsub_refs[i])-np.std(bgsub_refs[i])
        ref_vmax = np.mean(bgsub_refs[i])+np.std(bgsub_refs[i])

        ax[i,0].imshow(bgsub_imgs[i], cmap='Greys_r', vmin=img_vmin, vmax = img_vmax)
        ax[i,0].text(15,15, str(stars['id'][i]), fontsize = 18, color = color)
        ax[i,1].imshow(bgsub_refs[i], cmap='Greys_r', vmin=ref_vmin, vmax = ref_vmax)
        ax[i,2].imshow(ref_stamps[i]/np.sum(ref_stamps[i])-img_stamps[i]/np.sum(img_stamps[i]), \
             cmap='Greys_r', vmin=-1e-3, vmax = 1e-3)
        ax[i,3].imshow(bgsub_refs[i]-bgsub_imgs[i], cmap='Greys_r', vmin=-1e-3, vmax = 1e-3)

    print('>>> Star stamps will be saved here: '+params['OUTDIR']+file_basename+'_star_stamps.png')
    save_toph_figure(params['OUTDIR']+file_basename+'_star_stamps.png')










   




