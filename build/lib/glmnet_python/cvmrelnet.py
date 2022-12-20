# -*- coding: utf-8 -*-
"""
Internal function called by cvglmnet. See also cvglmnet

"""
import numpy as np
from glmnetPredict import glmnetPredict
from wtmean import wtmean
from cvcompute import cvcompute

def cvmrelnet(fit, \
            lambdau, \
            x, \
            y, \
            weights, \
            offset, \
            foldid, \
            ptype, \
            grouped, \
            keep = False):
    
    typenames = {'deviance':'Mean-Squared Error', 'mse':'Mean-Squared Error', 
                 'mae':'Mean Absolute Error'}
    if ptype == 'default':
        ptype = 'mse'

    ptypeList = ['mse', 'mae', 'deviance']    
    if not ptype in ptypeList:
        print('Warning: only ', ptypeList, 'available for Gaussian models; ''mse'' used')
        ptype = 'mse'

    nobs, nc = y.shape
    
    if len(offset) > 0:
        y = y - offset

    predmat = np.ones([nobs, nc, lambdau.size])*np.NAN               
    nfolds = np.amax(foldid) + 1
    nlams = [] 
    for i in range(nfolds):
        which = foldid == i
        fitobj = fit[i].copy()
        fitobj['offset'] = False
        preds = glmnetPredict(fitobj, x[which, ])
        nlami = np.size(fit[i]['lambdau'])
        predmat[which, 0:nlami] = preds
        nlams.append(nlami)
    # convert nlams to scipy array
    nlams = np.array(nlams, dtype = np.integer)

    N = nobs - np.reshape(np.sum(np.isnan(predmat[:, 1, :]), axis = 0), (1, -1))
    bigY = np.tile(y[:, :, None], [1, 1, lambdau.size])

    if ptype == 'mse':
        cvraw = np.sum((bigY - predmat)**2, axis = 1).squeeze()
    elif ptype == 'mae':
        cvraw = np.sum(np.absolute(bigY - predmat), axis = 1).squeeze()
        
    if y.size/nfolds < 3 and grouped == True:
        print('Option grouped=false enforced in cv.glmnet, since < 3 observations per fold')
        grouped = False
        
    if grouped == True:
        cvob = cvcompute(cvraw, weights, foldid, nlams)
        cvraw = cvob['cvraw']
        weights = cvob['weights']
        N = cvob['N']
        
    cvm = wtmean(cvraw, weights)
    sqccv = (cvraw - cvm)**2
    cvsd = np.sqrt(wtmean(sqccv, weights)/(N-1))

    result = dict()
    result['cvm'] = cvm
    result['cvsd'] = cvsd
    result['name'] = typenames[ptype]

    if keep:
        result['fit_preval'] = predmat
        
    return(result)

# end of cvelnet
#=========================    
