# -*- coding: utf-8 -*-
"""
Internal function called by cvglmnet. See also cvglmnet

"""
import numpy as np
from glmnetPredict import glmnetPredict
from wtmean import wtmean
from cvcompute import cvcompute

def cvmultnet(fit, \
            lambdau, \
            x, \
            y, \
            weights, \
            offset, \
            foldid, \
            ptype, \
            grouped, \
            keep = False):
    
    typenames = {'deviance':'Multinomial Deviance', 'mse':'Mean-Squared Error', 
                 'mae':'Mean Absolute Error', 'class':'Misclassification Error'}
    if ptype == 'default':
        ptype = 'deviance'
        
    ptypeList = ['mse', 'mae', 'deviance', 'class']    
    if not ptype in ptypeList:
        print('Warning: only ', ptypeList, 'available for multinomial models; ''deviance'' used')
        ptype = 'deviance'

    prob_min = 1.0e-5
    prob_max = 1 - prob_min
    nc = y.shape        
    if nc[1] == 1:
        classes, sy = np.unique(y, return_inverse = True)
        nc = len(classes)
        indexes = np.eye(nc, nc)
        y = indexes[sy, :]
    else:
        nc = nc[1]
        
    is_offset = not(len(offset) == 0)
    predmat = np.ones([y.shape[0], nc, lambdau.size])*np.NAN               
    nfolds = np.amax(foldid) + 1
    nlams = []    
    for i in range(nfolds):
        which = foldid == i
        fitobj = fit[i].copy()
        if is_offset:
            off_sub = offset[which, ]
        else:
            off_sub = np.empty([0])
        preds = glmnetPredict(fitobj, x[which, ], np.empty([0]), 'response', False, off_sub)
        nlami = np.size(fit[i]['lambdau'])
        predmat[which, 0:nlami] = preds
        nlams.append(nlami)
    # convert nlams to np array
    nlams = np.array(nlams, dtype = np.integer)

    ywt = np.sum(y, axis = 1, keepdims = True)
    y = y/np.tile(ywt, [1, y.shape[1]])
    weights = weights*ywt
    N = y.shape[0] - np.sum(np.isnan(predmat[:,1,:]), axis = 0, keepdims = True)
    bigY = np.tile(y[:, :, None], [1, 1, lambdau.size])

    if ptype == 'mse':
        cvraw = np.sum((bigY - predmat)**2, axis = 1).squeeze()
    elif ptype == 'deviance':
        predmat = np.minimum(np.maximum(predmat, prob_min), prob_max)
        lp = bigY*np.log(predmat)
        ly = bigY*np.log(bigY)
        ly[y == 0] = 0
        cvraw = np.sum(2*(ly - lp), axis = 1).squeeze()
    elif ptype == 'mae':
        cvraw = np.sum(np.absolute(bigY - predmat), axis = 1).squeeze()
    elif ptype == 'class':
        classid = np.zeros([y.shape[0], lambdau.size])*np.NaN
        for i in range(lambdau.size):
            classid[:, i] = glmnet_softmax(predmat[:,:,i])
        classid = classid.reshape([classid.size,1])    
        yperm = bigY.transpose((0,2,1))
        yperm = yperm.reshape([yperm.size, 1])
        idx =  sub2ind(yperm.shape, range(len(classid)), classid.transpose())
        cvraw = np.reshape(1 - yperm[idx], [-1, lambdau.size]);
        
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
#
#=========================    
# Helper functions
#=========================    
def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols
#=========================    
def glmnet_softmax(x):
    d = x.shape
    nas = np.any(np.isnan(x), axis = 1)
    if np.any(nas):
        pclass = np.zeros([d[0], 1])*np.NaN
        if np.sum(nas) < d[0]:
            pclass2 = glmnet_softmax(x[~nas, :])
            pclass[~nas] = pclass2
            result = pclass
    else:
        maxdist = x[:, 1]
        pclass = np.ones([d[0], 1])
        for i in range(1, d[1], 1):
            t = x[:, i] > maxdist
            pclass[t] = i
            maxdist[t] = x[t, i]
        result = pclass
        
    return(result)    
#=========================    
