# -*- coding: utf-8 -*-
"""
Internal glmnet function. See also cvglmnet.

Compute the weighted mean and SD within folds, and hence the SE of the mean
"""
import numpy as np
from wtmean import wtmean

def cvcompute(mat, weights, foldid, nlams):
    if len(weights.shape) > 1:
        weights = np.reshape(weights, [weights.shape[0], ])
    wisum = np.bincount(foldid, weights = weights)
    nfolds = np.amax(foldid) + 1
    outmat = np.ones([nfolds, mat.shape[1]])*np.NaN
    good = np.zeros([nfolds, mat.shape[1]])
    mat[np.isinf(mat)] = np.NaN
    for i in range(nfolds):
        tf = foldid == i
        mati = mat[tf, ]
        wi = weights[tf, ]
        outmat[i, :] = wtmean(mati, wi)
        good[i, 0:nlams[i]] = 1
    N = np.sum(good, axis = 0)
    cvcpt = dict()
    cvcpt['cvraw'] = outmat
    cvcpt['weights'] = wisum
    cvcpt['N'] = N

    return(cvcpt)

# end of cvcompute
#=========================    
    
