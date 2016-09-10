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
    
    N = x.shape[0]

    if y.shape[0] != N:
        y = scipy.transpose(y)
       
    if (len(options['offset']) > 0) and (options['offset'].shape[0] != N):
        options['offset'] = scipy.transpose(options['offset'])
        
    if len(options['weights']) == 0:
        options['weights'] = scipy.ones([N, 1], dtype = scipy.float64)

    # main call to glmnet        
    glmfit = glmnet(x = x, y = y, family = family, **options)    

    is_offset = glmfit['offset']
    options['lambdau'] = glmfit['lambdau']
    
    nz = glmnetPredict(glmfit, scipy.empty([0]), scipy.empty([0]), 'nonzero')
    if glmnet['class'] == 'multnet':        
        nnz = scipy.zeros([len(options['lambdau']), nz.shape[1]])
        for i in range(nz.shape[1]):
            nnz[:, i] = scipy.transpose(scipy.sum(nz[i], axis = 0))
        nz = scipy.ceil(scipy.median(nnz, axis = 1))    
    elif glmnet['class'] == 'mrelnet':
        nz = scipy.transpose(scipy.sum(nz[0], axis = 0))
    else:
        nz = scipy.transpose(scipy.sum(nz, axis = 0))
    
    if len(foldid) == 0:
        ma = scipy.tile(scipy.arange(nfolds), [1, scipy.floor(N/nfolds)])
        mb = scipy.arange(scipy.mod(N, nfolds))
        population = scipy.append(ma, mb, axis = 1)
        mc = scipy.random.permutation(len(population))
        mc = mc[0:N]
        foldid = population(mc)
    else:
        nfolds = scipy.amax(foldid)
        
    if nfolds < 3:
        raise ValueError('nfolds must be bigger than 3; nfolds = 10 recommended')        
        
    cpredmat = list()
    if parallel == True:
        # TODO: parallel not yet implemented
        raise NotImplementedError('Parallel for cvglmnet not yet implemented')
    else:
        for i in range(nfolds):
            which = foldid == i
            opts = options
            opts['weights'] = opts['weights'][~which, :]
            opts['lambdau'] = options['lambdau']
            if is_offset:
                opts['offset'] = opts['offset'][~which, :]
            xr = x[~which, :]
            yr = y[~which, :]
            cpredmat.append(glmnet(x = xr, y = yr, family = family, **opts))
            
    if cpredmat[0]['class'] == 'elnet':
        cvstuff = cvelnet( cpredmat, options['lambda'], x, y \
                          , options['weights'], options['offset'] \
                          , foldid, ptype, grouped, keep)
    elif cpredmat[0]['class'] == 'lognet':
        cvstuff = cvlognet(cpredmat, options['lambda'], x, y \
                          , options['weights'], options['offset'] \
                          , foldid, ptype, grouped, keep)
    elif cpredmat[0]['class'] == 'multnet':
        cvstuff = cvmultnet(cpredmat, options['lambda'], x, y \
                          , options['weights'], options['offset'] \
                          , foldid, ptype, grouped, keep)
    elif cpredmat[0]['class'] == 'coxnet':
        cvstuff = cvcoxnet(cpredmat, options['lambda'], x, y \
                          , options['weights'], options['offset'] \
                          , foldid, ptype, grouped, keep)
    elif cpredmat[0]['class'] == 'mrelnet':
        cvstuff = cmrelnet(cpredmat, options['lambda'], x, y \
                          , options['weights'], options['offset'] \
                          , foldid, ptype, grouped, keep)
    elif cpredmat[0]['class'] == 'fishnet':
        cvstuff = cvfishnet(cpredmat, options['lambda'], x, y \
                          , options['weights'], options['offset'] \
                          , foldid, ptype, grouped, keep)
 
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
    CVerr['lambda_min'] = scipy.amax(options['lambdau'][cvm <= scipy.amin(cvm)])    
    idmin = options['lambdau'] == CVerr['lambda_min']
    semin = cvm[idmin] + cvsd[idmin]
    CVerr['lambda_1se'] = scipy.amax(options['lambdau'][cvm <= semin])
    CVerr['class'] = 'cv.glmnet'
    
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
    