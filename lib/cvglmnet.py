# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:56:06 2016

@author: bbalasub
"""
import joblib
import multiprocessing
from glmnetSet import glmnetSet
from glmnetPredict import glmnetPredict
import scipy
from glmnet import glmnet
from cvelnet import cvelnet
from cvlognet import cvlognet
from cvmultnet import cvmultnet

def cvglmnet(x, \
             y, \
             family = 'gaussian', \
             ptype = 'default', \
             nfolds = 10, \
             foldid = scipy.empty([0]), \
             parallel = False, \
             keep = False, \
             grouped = True, \
             **options):

    options = glmnetSet(options)

    if len(options['lambdau']) != 0 and len(options['lambda'] < 2):
        raise ValueError('Need more than one value of lambda for cv.glmnet')
    
    nobs = x.shape[0]

    # we should not really need this. user must supply the right shape
    # if y.shape[0] != nobs:
    #    y = scipy.transpose(y)
        
    # convert 1d python array of size nobs to 2d python array of size nobs x 1
    if len(y.shape) == 1:
        y = scipy.reshape(y, [y.size, 1])

    # we should not really need this. user must supply the right shape       
    # if (len(options['offset']) > 0) and (options['offset'].shape[0] != nobs):
    #    options['offset'] = scipy.transpose(options['offset'])
    
    if len(options['weights']) == 0:
        options['weights'] = scipy.ones([nobs, 1], dtype = scipy.float64)

    # main call to glmnet        
    glmfit = glmnet(x = x, y = y, family = family, **options)    

    is_offset = glmfit['offset']
    options['lambdau'] = glmfit['lambdau']
    
    nz = glmnetPredict(glmfit, scipy.empty([0]), scipy.empty([0]), 'nonzero')
    if glmfit['class'] == 'multnet':        
        nnz = scipy.zeros([len(options['lambdau']), len(nz)])
        for i in range(len(nz)):
            nnz[:, i] = scipy.transpose(scipy.sum(nz[i], axis = 0))
        nz = scipy.ceil(scipy.median(nnz, axis = 1))    
    elif glmfit['class'] == 'mrelnet':
        nz = scipy.transpose(scipy.sum(nz[0], axis = 0))
    else:
        nz = scipy.transpose(scipy.sum(nz, axis = 0))
    
    if len(foldid) == 0:
        ma = scipy.tile(scipy.arange(nfolds), [1, scipy.floor(nobs/nfolds)])
        mb = scipy.arange(scipy.mod(nobs, nfolds))
        mb = scipy.reshape(mb, [1, mb.size])
        population = scipy.append(ma, mb, axis = 1)
        mc = scipy.random.permutation(len(population))
        mc = mc[0:nobs]
        foldid = population[mc]
        foldid = scipy.reshape(foldid, [foldid.size,])
    else:
        nfolds = scipy.amax(foldid) + 1
        
    if nfolds < 3:
        raise ValueError('nfolds must be bigger than 3; nfolds = 10 recommended')        
        
    cpredmat = list()
    foldid = scipy.reshape(foldid, [foldid.size, ])
    if parallel == True:
        # TODO: parallel not yet implemented
        raise NotImplementedError('Parallel for cvglmnet not yet implemented')
    else:
        for i in range(nfolds):
            which = foldid == i
            opts = options.copy()
            opts['weights'] = opts['weights'][~which, ]
            opts['lambdau'] = options['lambdau']
            if is_offset:
                if opts['offset'].size > 0:
                    opts['offset'] = opts['offset'][~which, ]
            xr = x[~which, ]
            yr = y[~which, ]
            newFit = glmnet(x = xr, y = yr, family = family, **opts)
            cpredmat.append(newFit)
    
    if cpredmat[0]['class'] == 'elnet':
        cvstuff = cvelnet( cpredmat, options['lambdau'], x, y \
                          , options['weights'], options['offset'] \
                          , foldid, ptype, grouped, keep)
    elif cpredmat[0]['class'] == 'lognet':
        cvstuff = cvlognet(cpredmat, options['lambdau'], x, y \
                          , options['weights'], options['offset'] \
                          , foldid, ptype, grouped, keep)
    elif cpredmat[0]['class'] == 'multnet':
        cvstuff = cvmultnet(cpredmat, options['lambdau'], x, y \
                          , options['weights'], options['offset'] \
                          , foldid, ptype, grouped, keep)
#    elif cpredmat[0]['class'] == 'coxnet':
#        cvstuff = cvcoxnet(cpredmat, options['lambdau'], x, y \
#                          , options['weights'], options['offset'] \
#                          , foldid, ptype, grouped, keep)
#    elif cpredmat[0]['class'] == 'mrelnet':
#        cvstuff = cmrelnet(cpredmat, options['lambdau'], x, y \
#                          , options['weights'], options['offset'] \
#                          , foldid, ptype, grouped, keep)
#    elif cpredmat[0]['class'] == 'fishnet':
#        cvstuff = cvfishnet(cpredmat, options['lambdau'], x, y \
#                          , options['weights'], options['offset'] \
#                          , foldid, ptype, grouped, keep)
 
    cvm = cvstuff['cvm']
    cvsd = cvstuff['cvsd']
    cvname = cvstuff['name']

    CVerr = dict()
    CVerr['lambdau'] = options['lambdau']       
    CVerr['cvm'] = scipy.transpose(cvm)
    CVerr['cvsd'] = scipy.transpose(cvsd)
    CVerr['cvup'] = scipy.transpose(cvm + cvsd)
    CVerr['cvlo'] = scipy.transpose(cvm - cvsd)
    CVerr['nzero'] = nz
    CVerr['name'] = cvname
    CVerr['glmnet_fit'] = glmfit
    if keep:
        CVerr['fit_preval'] = cvstuff['fit_preval']
        CVerr['foldid'] = foldid
    if ptype == 'auc':
        cvm = -cvm
    CVerr['lambda_min'] = scipy.amax(options['lambdau'][cvm <= scipy.amin(cvm)]).reshape([1])  
    idmin = options['lambdau'] == CVerr['lambda_min']
    semin = cvm[idmin] + cvsd[idmin]
    CVerr['lambda_1se'] = scipy.amax(options['lambdau'][cvm <= semin]).reshape([1])
    CVerr['class'] = 'cvglmnet'
    
    return(CVerr)
        
# end of cvglmnet
#==========================

# TODO: parallel stuff
# This is future reference code for parallel implementation 
#    inputs = range(10)
#    num_cores = multiprocessing.cpu_count()
#    results = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(childFunc)(i) for i in inputs)

#def childFunc(i):
#	    return i * i
    