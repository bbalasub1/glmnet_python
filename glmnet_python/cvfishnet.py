# -*- coding: utf-8 -*-
"""
Internal cvglmnet function. See also cvglmnet.

"""
import numpy 
from glmnetPredict import glmnetPredict
from wtmean import wtmean
from cvcompute import cvcompute

def cvfishnet(fit, \
            lambdau, \
            x, \
            y, \
            weights, \
            offset, \
            foldid, \
            ptype, \
            grouped, \
            keep = False):
    
    typenames = {'deviance':'Poisson Deviance', 'mse':'Mean-Squared Error', 
                 'mae':'Mean Absolute Error'}
    if ptype == 'default':
        ptype = 'deviance'

    ptypeList = ['mse', 'mae', 'deviance']    
    if not ptype in ptypeList:
        print('Warning: only ', ptypeList, 'available for Poisson models; ''deviance'' used')
        ptype = 'deviance'
        
    if len(offset) > 0:
        is_offset = True
    else:
        is_offset = False 

    predmat = numpy.ones([y.size, lambdau.size])*numpy.NAN               
    nfolds = numpy.amax(foldid) + 1
    nlams = [] 
    for i in range(nfolds):
        which = foldid == i
        fitobj = fit[i].copy()
        if is_offset:
            off_sub = offset[which]
        else:
            off_sub = numpy.empty([0])
        preds = glmnetPredict(fitobj, x[which, ], offset = off_sub)
        nlami = numpy.size(fit[i]['lambdau'])
        predmat[which, 0:nlami] = preds
        nlams.append(nlami)
    # convert nlams to scipy array
    nlams = numpy.array(nlams, dtype = numpy.integer)

    N = y.shape[0] - numpy.sum(numpy.isnan(predmat), axis = 0)
    yy = numpy.tile(y, [1, lambdau.size])

    if ptype == 'mse':
        cvraw = (yy - predmat)**2
    elif ptype == 'deviance':
        cvraw = devi(yy, predmat)
    elif ptype == 'mae':
        cvraw = numpy.absolute(yy - predmat)
        
    if y.size/nfolds < 3 and grouped == True:
        print('Option grouped=false enforced in cvglmnet, since < 3 observations per fold')
        grouped = False
        
    if grouped == True:
        cvob = cvcompute(cvraw, weights, foldid, nlams)
        cvraw = cvob['cvraw']
        weights = cvob['weights']
        N = cvob['N']
        
    cvm = wtmean(cvraw, weights)
    sqccv = (cvraw - cvm)**2
    cvsd = numpy.sqrt(wtmean(sqccv, weights)/(N-1))

    result = dict()
    result['cvm'] = cvm
    result['cvsd'] = cvsd
    result['name'] = typenames[ptype]

    if keep:
        result['fit_preval'] = predmat
        
    return(result)

# end of cvfishnet
#=========================    
def devi(yy, eta):
    deveta = yy*eta - numpy.exp(eta)
    devy = yy*numpy.log(yy) - yy
    devy[yy == 0] = 0
    result = 2*(devy - deveta)
    return(result)
    



