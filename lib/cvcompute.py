# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 07:31:19 2016

@author: bbalasub
"""
import scipy
from wtmean import wtmean

def cvcompute(mat, weights, foldid, nlams):
    if len(weights.shape) > 1:
        weights = scipy.reshape(weights, [weights.shape[0], ])
    wisum = scipy.bincount(foldid, weights = weights)
    nfolds = scipy.amax(foldid) + 1
    outmat = scipy.ones([nfolds, mat.shape[1]])*scipy.NaN
    good = scipy.zeros([nfolds, mat.shape[1]])
    mat[scipy.isinf(mat)] = scipy.NaN
    for i in range(nfolds):
        tf = foldid == i
        mati = mat[tf, ]
        wi = weights[tf, ]
        outmat[i, :] = wtmean(mati, wi)
        good[i, 0:nlams[i]] = 1
    N = scipy.sum(good, axis = 0)
    cvcpt = dict()
    cvcpt['cvraw'] = outmat
    cvcpt['weights'] = wisum
    cvcpt['N'] = N

    return(cvcpt)

# end of cvcompute
#=========================    
    