# -*- coding: utf-8 -*-
"""
calculate nan-removed weighted mean

PRE:
    mat:     must be n x k
    weights: must be n x 1
    
POST:


@author: bbalasub
"""
import scipy

def wtmean(mat,weights):
    wmat = isfinite(mat)*weights
    mat[isnan(mat)] = 0
    swmat = mat*wmat
    tf = weights != 0
    tf = tf[:,0]    
    y = scipy.sum(swmat[tf, :], axis = 0)/scipy.sum(wmat, axis = 0)        
    return y
# end of wtmean

def isnan(x):
    return ~scipy.isfinite(x)    
# end of isnan

def isfinite(x):
    return scipy.isfinite(x)        
# end of isfinite    
    