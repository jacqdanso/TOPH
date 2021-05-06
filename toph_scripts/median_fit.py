################################################################################################
# This script iteratively find the median of a 3-sigma clipped distribution and uses the mean and
# median to determine the mode via Karl Pearson's formula. 
#
# Written by Jacqueline Antwi-Danso sometime in summer 2018, modified 01/04/2019
################################################################################################

import numpy as np
from astropy.stats import mad_std

def median_fit(array,  id_i):    
    median  = np.median(array)
    std = mad_std(array)
    mean = np.mean(array)
    
    converged = False 
    
    while not converged:
        new_arr = [i for i in array if np.logical_and(i <= median+(2.3*std), \
                                                          i >= median-(2.3*std))]
    
        new_median = np.median(new_arr)
        new_mean = np.mean(new_arr)
        new_std = mad_std(new_arr)
            
        if np.isclose(median,new_median):
            converged = True 
            
            if np.isclose(new_median, new_mean): 
                mode = new_median
            else:
                mode = 3*median - 2*mean     
                
            min_num = min(new_arr)
            max_num = max(new_arr)
                
        else: 
            median = new_median
            mean = new_mean
            std = new_std
           
    return mode, min_num, max_num
