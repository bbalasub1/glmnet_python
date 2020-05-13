# -*- coding: utf-8 -*-
"""
Calculate nan-removed weighted mean. The mean is  
computed in the axis=0 direction along each column.

INPUT ARGUMENTS:
---------------
    mat:     must be a 2D scipy array of size N x K
    weights: must be a 2D scipy array of size N x 1 or a 1-D array of size N
    
OUTPUT ARGUMENTS:
----------------
    returns nan-removed weighted mean as a 1D array of size K

"""
import numpy as np

def wtmean(mat,weights):
    if len(weights.shape) == 1:
        weights = np.reshape(weights, [np.size(weights), 1])
    wmat = isfinite(mat)*weights
    mat[isnan(mat)] = 0
    swmat = mat*wmat
    tf = weights != 0
    tf = tf[:,0]    
    y = np.sum(swmat[tf, :], axis = 0)/np.sum(wmat, axis = 0)        
    return y
# end of wtmean

def isnan(x):
    return ~np.isfinite(x)    
# end of isnan

def isfinite(x):
    return np.isfinite(x)        
# end of isfinite    
    
