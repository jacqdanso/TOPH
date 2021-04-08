# includes buffer in image slices so that we don't end up with sharp edges after convolution

import numpy as np

def include_buffer(image, kersize, xpts, ypts, x1, x2, y1, y2):
	#if slice lies on the left edge, not including top left and bottom left corners
	if np.logical_and(np.logical_and(x1 == 0, x2 == xpts[1]), np.logical_and(y2 != ypts[len(ypts)-1], y1 != 0)):
		x1buff = 0 ; x2buff = kersize 
		y1buff = kersize ; y2buff = kersize 
	#if slice lies on the top edge, not including top left and right corners
	elif np.logical_and(np.logical_and(x1 > 0, y1 == 0), x2 != xpts[len(xpts)-1]):
		x1buff = kersize ; x2buff = kersize 
		y1buff = 0 ; y2buff = kersize
	#if slice lies on the right edge, not including bottom and top right corners
	elif np.logical_and(x2+kersize > xpts[len(xpts)-1], np.logical_and(y1 != 0, y2+kersize < ypts[len(ypts)-1])):
		x1buff = kersize ; x2buff = 0
		y1buff = kersize ; y2buff = kersize
	#if slice lies on the bottom edge, not including bottom left and right corners 
	elif np.logical_and(np.logical_and(y2 == ypts[len(ypts)-1], x2+kersize < xpts[len(xpts)-1]), x2 != xpts[1]):
		x1buff = kersize ; x2buff = kersize
		y1buff = kersize ; y2buff = 0
	#top left corner
	elif np.logical_and(np.logical_and(x1 == 0, x2 == xpts[1]), y1 == 0):
		x1buff = 0 ; x2buff = kersize
		y1buff = 0 ; y2buff = kersize
	#top right corner 
	elif np.logical_and(np.logical_and(y1 == 0, x1 == xpts[len(xpts)-2]), y2 == ypts[1]):
		x1buff = kersize ; x2buff = 0
		y1buff = 0 ; y2buff = kersize
	#bottom left corner 
	elif np.logical_and(np.logical_and(y1 == ypts[len(ypts)-2], x1 == 0), x2 == xpts[1]):
		x1buff = 0 ; x2buff = kersize
		y1buff = kersize ; y2buff = 0
	#bottom right corner 
	elif np.logical_and(y2+kersize > ypts[len(ypts)-1], x2+kersize > xpts[len(xpts)-1]):
		x1buff = kersize ; x2buff = 0
		y1buff = kersize ; y2buff = 0
	#rest of image
	else: 
		x1buff = kersize ; x2buff = kersize
		y1buff = kersize ; y2buff = kersize

	newx1 = x1 - x1buff ; newy1 = y1-y1buff
	newx2 = x2 + x2buff ; newy2 = y2+y2buff
	
	return newx1, newy1, newx2, newy2, x1buff, y1buff, x2buff, y2buff