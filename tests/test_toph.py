import unittest
import numpy as np
from astropy.io import fits 
from toph_scripts.toph_params import toph_params
from toph_scripts.create_kernels import create_kernels
from toph_scripts.convolve_image import get_grid
from scipy.io import readsav
from toph_scripts.include_buffer import include_buffer
from astropy.convolution import convolve_fft

params = toph_params()

def test_params(test_ker=False):
    regfact = params['REGFACT']
    psf_points = readsav(params['SAV_FILE'])
    xpoints = psf_points['spx'].flatten()
    ypoints = psf_points['spy'].flatten()

    img_filepath = params['IMG_FILEPATH']
    img = fits.open(img_filepath)[0].data

    img_psfs = fits.open(params['IMG_PSFGRID'])[0].data
    ref_psfs = fits.open(params['REFERENCE_PSFGRID'])[0].data

    assert len(img_psfs) == len(ref_psfs), "Number of reference and science PSFs should be the same"
    assert np.shape(img_psfs) == np.shape(ref_psfs), "Reference and science PSFs should have the same size"

    if test_ker:
        return img_psfs, ref_psfs, regfact
    else: 
        return params, img, img_filepath, xpoints, ypoints 

def test_kernels():
    test_ker = True
    img_psfs, ref_psfs, regfact = test_params(test_ker)
    kernels, shifted_kernels = create_kernels(img_psfs=img_psfs, ref_psfs=ref_psfs, file_basename = None, \
        params = params, save_kernels = False)
    buffer_size = np.shape(shifted_kernels[0])[0]

    return buffer_size, shifted_kernels

def test_slicing():
    test_ker = False
    params, img, img_filepath, xpoints, ypoints = test_params(test_ker)

    # note kernels is being overwritten here
    buffer_size, kernels = test_kernels()

    grid = get_grid(xpoints, ypoints)

    x = np.linspace(0, max(xpoints), num = (grid[1]*2)-1, dtype = int)
    y = np.linspace(0, max(ypoints), num = (grid[0]*2)-1, dtype = int)
    xpts = np.repeat(x, 2)
    ypts = np.repeat(y, 2)
    xpts = np.delete(xpts, [0, len(xpts)-1])
    ypts = np.delete(ypts, [0, len(ypts)-1])
    
    print('Slicing image')
    ## slice image and include buffer 
    slices = []; buffers = []
    ### the slicing is done in row-order
    for i in range(len(xpts)-1):
        for j in range(len(ypts)-1):
            x1 = xpts[i]; y1 = ypts[j]
            x2 = xpts[i+1]; y2 = ypts[j+1]
            if np.logical_and((x2-x1) > 1, (y2-y1 > 1)):
                newx1, newy1, newx2, newy2, x1buff, y1buff, x2buff, y2buff \
                = include_buffer(img, buffer_size, xpts, ypts, x1, x2, y1, y2)
                slices.append(img[newy1:newy2, newx1:newx2])
                buffers.append([x1buff, y1buff, x2buff, y2buff])
                slice_num = np.arange(0, len(slices))+1

    print('There are '+str(len(slice_num))+' slices')

    return slice_num, grid, slices, buffers

def test_reordering():
    slice_num, grid, slices, buffers = test_slicing()
    # put slices in the same order as psf points
    slice_num = np.array(slice_num).reshape((grid[1]*2)-2, (grid[0]*2)-2)
    slice_num = slice_num.T
    slice_num = np.flip(slice_num, axis = 0)
    slice_num = slice_num.flatten()
    np.flip(slice_num.reshape((grid[0]*2)-2,(grid[1]*2)-2), axis = 0).T.flatten()
    new_slice_num = np.arange(0, len(slices))+1

    buff_index = np.array([np.where(new_slice_num == i)[0][0] for i in slice_num])
    new_slice_index =  [np.where(buff_index == i)[0][0] for i in new_slice_num-1]

    return buff_index, new_slice_index 

def test_kernel_grid():
    test_ker = False
    params, img, img_filepath, xpoints, ypoints = test_params(test_ker)
    buffer_size, kernels = test_kernels()
    slice_num, grid, slices, buffers = test_slicing()
    buff_index, new_slice_index = test_reordering()

    print('Creating kernel grid')
    # reshape and flip slice array
    slices = np.array(slices, dtype = object).reshape((grid[1]*2)-2, (grid[0]*2)-2)
    slices = np.flip(slices.T, axis = 0).flatten()

    # do the same thing for the buffers
    buffers = np.array(buffers)[buff_index]

    N_left = (grid[0]-2)*2 + 2
    right_edge = [] 
    right_edge.append(grid[1]-1)
    for i in range(N_left-1):
        right_edge.append(right_edge[len(right_edge)-1]+grid[1])

    left_edge = [i+1 for i in right_edge] # doing it this way so they stay as ints
    left_edge.append(0)
    left_edge = [i for i in np.sort(left_edge)[:-1]] #remove last element because we need right and left edge to be same length, and we're appending 0

    # remove extras because those will be taken care of by duplicates 
    right_edge = right_edge[:grid[0]]
    left_edge = left_edge[:grid[0]]

    indices = np.linspace(0, len(xpoints)-1, num = len(xpoints), dtype = int)
    indices = np.repeat(indices,2)

    duplicates = np.sort(np.concatenate([left_edge, right_edge])) # sorting not necessary, just to help me know if I've done the right thing
    indices = [i for i in indices]

    for i in duplicates:
        indices.remove(i)
    indices = np.array(indices)

    indices = np.reshape(indices, (grid[0], (grid[1]*2)-2)) # num of columns*2 (since we duplicated) - 2 since we deleted left and right edge duplicates
    dup_arr = np.array([np.broadcast_to(row, (2, *np.shape(row))) for row in indices])
    dup_arr = np.reshape(dup_arr, (np.shape(dup_arr)[0]* np.shape(dup_arr)[1], np.shape(dup_arr)[2]))
    ## delete top and bottom rows 
    del_rows = [0, (grid[0]*2)-2] #-2 because the first row would have been taken out by the time that row is being removed
    for row_index in del_rows:
        dup_arr = np.delete(dup_arr, row_index, axis = 0)

    # to put kernel array in the correct order
    ker_indices = np.flip(dup_arr, axis = 0).flatten()
    ker_grid = np.array(kernels)[ker_indices]

    return kernels, ker_grid, slices, dup_arr, buffers

def test_convolution(conv_type = 'fft'):
    kernels, ker_grid, slices, dup_arr, buffers = test_kernel_grid()
    print('Convolving image slices with kernel grid')
    convol_grid = []
    for chunk, kernel, num in zip(slices, ker_grid, np.arange(len(slices))+1):
        if conv_type == 'convolve2d':
            convol_grid.append(convolve2d(chunk, kernel, boundary='fill', mode='same'))
        elif conv_type == 'fft':
            convol_grid.append(convolve_fft(chunk, kernel, boundary='fill', \
                preserve_nan = True))
        print('Convolving slice #'+str(num))

    return convol_grid

def test_stitching():
    slice_num, grid, slices, buffers = test_slicing()
    buff_index, new_slice_index = test_reordering()
    kernels, ker_grid, slices, dup_arr, buffers = test_kernel_grid()
    convol_grid = test_convolution()

    print('Stitching image back together')
    new_slices = []

    for chunk, buffs in zip(convol_grid, np.array(buffers)):
        x1 = 0; y1 = 0
        x2 = np.shape(chunk)[1]; y2 = np.shape(chunk)[0]
        newx1 = x1 + buffs[0]
        newy1 = y1 + buffs[1]
        newx2 = x2 - buffs[2]
        newy2 = y2 - buffs[3]
        new_slices.append(chunk[newy1:newy2, newx1:newx2])

    # flip and transpose slice array
    new_slices = np.array(new_slices, dtype = object)[new_slice_index]

    ### plus 1 because np.arange does not include stop value
    row_indices = np.arange(0, (np.shape(dup_arr)[0]*np.shape(dup_arr)[1])+1, step = np.shape(dup_arr)[0]) 
    row_indices = np.repeat(row_indices, 2)
    row_indices = np.delete(row_indices, [0, len(row_indices)-1])

    img_rows = []
    ##
    for i in range(len(row_indices)-1):
        if (row_indices[i+1]-row_indices[i]) > 1:
            img_rows.append(np.concatenate(new_slices[row_indices[i]:row_indices[i+1]]))

    #put columns together 
    convol_img = np.concatenate(img_rows, axis=1)
    
