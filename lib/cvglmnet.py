# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------
 cvglmnet.m: cross-validation for glmnet
--------------------------------------------------------------------------

 DESCRIPTION:
    Does k-fold cross-validation for glmnet, produces a plot, and returns
    a value for lambdau. Cross-validation is not implemented for Cox model yet.

 USAGE:

    Note that like glmnet, all arguments are keyword-only:
 
    CVerr = cvglmnet(x, y, family, options, type, nfolds, foldid,
    parallel, keep, grouped);

    Fewer input arguments(more often) are allowed in the call. Default values
    for the arguments are used unless specified by the user.
        
=======================
INPUT ARGUMENTS
 x           nobs x nvar scipy 2D array of x parameters (as in glmnet).
 y           nobs x nc scipy Response y as in glmnet.
 family      Response type as family in glmnet.
 options     Options as in glmnet.
 ptype       loss to use for cross-validation. Currently five options, not
             all available for all models. The default is ptype='deviance', which uses
             squared-error for Gaussian models (a.k.a ptype='mse' there), deviance for
             logistic and Poisson regression, and partial-likelihood for the Cox
             model (Note that CV for cox model is not implemented yet). 
             ptype='class' applies to binomial and multinomial logistic
             regression only, and gives misclassification error. ptype='auc' is for
             two-class logistic regression only, and gives area under the ROC curve.
             ptype='mse' or ptype='mae' (mean absolute error) can be used by all models
             except the 'cox'; they measure the deviation from the fitted mean to the
             response.  
 nfolds      number of folds - default is 10. Although nfolds can be as
             large as the sample size (leave-one-out CV), it is not recommended for
             large datasets. Smallest value allowable is nfolds=3.
 foldid      an optional vector of values between 1 and nfold identifying
             what fold each observation is in. If supplied, nfold can be
             missing.
 parallel    If True, use parallel computation to fit each fold. 
 keep        If keep=True, a prevalidated array is returned containing
             fitted values for each observation and each value of lambda.
             This means these fits are computed with this observation and
             the rest of its fold omitted. The foldid vector is also
             returned. Default is keep=False.   
 grouped     This is an experimental argument, with default true, and can
             be ignored by most users. For all models except the 'cox',
             this refers to computing nfolds separate statistics, and then
             using their mean and estimated standard error to describe the
             CV curve. If grouped=false, an error matrix is built up at
             the observation level from the predictions from the nfold
             fits, and then summarized (does not apply to
             type='auc'). For the 'cox' family, grouped=true obtains the 
             CV partial likelihood for the Kth fold by subtraction; by
             subtracting the log partial likelihood evaluated on the full
             dataset from that evaluated on the on the (K-1)/K dataset.
             This makes more efficient use of risk sets. With
             grouped=FALSE the log partial likelihood is computed only on
             the Kth fold.

=======================
OUTPUT ARGUMENTS:
 A dict() is returned with the following fields.
 lambdau     the values of lambda used in the fits.
 cvm         the mean cross-validated error - a vector of length
             length(lambdau). 
 cvsd        estimate of standard error of cvm.
 cvup        upper curve = cvm+cvsd.
 cvlo        lower curve = cvm-cvsd.
 nzero       number of non-zero coefficients at each lambda.
 name        a text string indicating type of measure (for plotting
             purposes). 
 glmnet_fit  a fitted glmnet object for the full data.
 lambda_min  value of lambda that gives minimum cvm.
 lambda_1se  largest value of lambda such that error is within 1 standard
             error of the minimum. 
 class       Type of regression - internal usage.
 fit_preval  if keep=true, this is the array of prevalidated fits. Some
             entries can be NA, if that and subsequent values of lambda
             are not reached for that fold.
 foldid      if keep=true, the fold assignments used.

 DETAILS:
    The function runs glmnet nfolds+1 times; the first to get the lambda
    sequence, and then the remainder to compute the fit with each of the 
    folds omitted. The error is accumulated, and the average error and 
    standard deviation over the folds is computed. Note that cvglmnet 
    does NOT search for values for alpha. A specific value should be 
    supplied, else alpha=1 is assumed by default. If users would like to 
    cross-validate alpha as well, they should call cvglmnet with a 
    pre-computed vector foldid, and then use this same fold vector in 
    separate calls to cvglmnet with different values of alpha. 

 LICENSE: GPL-2

 AUTHORS:
    Algorithm was designed by Jerome Friedman, Trevor Hastie and Rob Tibshirani
    Fortran code was written by Jerome Friedman
    R wrapper (from which the MATLAB wrapper was adapted) was written by Trevor Hasite
    The original MATLAB wrapper was written by Hui Jiang,
    and is updated and maintained by Junyang Qian.
    This Python wrapper (adapted from the Matlab and R wrappers) is written by Balakumar B.J., 
    Department of Statistics, Stanford University, Stanford, California, USA.

 REFERENCES:
    Friedman, J., Hastie, T. and Tibshirani, R. (2008) Regularization Paths for Generalized Linear Models via Coordinate Descent, 
    http://www.jstatsoft.org/v33/i01/
    Journal of Statistical Software, Vol. 33(1), 1-22 Feb 2010
    
    Simon, N., Friedman, J., Hastie, T., Tibshirani, R. (2011) Regularization Paths for Cox's Proportional Hazards Model via Coordinate Descent,
    http://www.jstatsoft.org/v39/i05/
    Journal of Statistical Software, Vol. 39(5) 1-13

    Tibshirani, Robert., Bien, J., Friedman, J.,Hastie, T.,Simon, N.,Taylor, J. and Tibshirani, Ryan. (2010) Strong Rules for Discarding Predictors in Lasso-type Problems,
    http://www-stat.stanford.edu/~tibs/ftp/strong.pdf
    Stanford Statistics Technical Report

 SEE ALSO:
    cvglmnetPlot, cvglmnetCoef, cvglmnetPredict, and glmnet.

 EXAMPLES:
 
      # Gaussian
      x = scipy.random.rand(100, 10)
      y = scipy.random.rand(100, 1)
      cvfit = cvglmnet(x = x, y = y)
      cvglmnetPlot(cvfit)
      print( cvglmnetCoef(cvfit) )
      print( cvglmnetPredict(cvfit, x[0:5, :], 'lambda_min') )
      cvfit1 = cvglmnet(x = x, y = y, ptype = 'mae')
      cvglmnetPlot(cvfit1)
      
      # Binomial
      x = scipy.random.rand(100, 10)
      y = scipy.random.rand(100,1)
      y = (y > 0.5)*1.0
      fit = cvglmnet(x = x, y = y, family = 'binomial', ptype = 'class')    
      cvglmnetPlot(fit)
      
      # poisson
      x = scipy.random.rand(100,10)
      y = scipy.random.poisson(size = [100, 1])*1.0
      cvfit = cvglmnet(x = x, y = y, family = 'poisson')
      cvglmnetPlot(cvfit)
      
      # Multivariate Gaussian:
      x = scipy.random.rand(100, 10)
      y = scipy.random.rand(100,3)
      cvfit = cvglmnet(x = x, y = y, family = 'mgaussian')      
      cvglmnetPlot(cvfit)
       
      # Multinomial
      x = scipy.random.rand(100,10)
      y = scipy.random.rand(100,1)
      y[y < 0.3] = 1.0
      y[y < 0.6] = 2.0
      y[y < 1.0] = 3.0
      cvfit = cvglmnet(x = x, y = y, family = 'multinomial')
      cvglmnetPlot(cvfit) 
      
      #cox
      Not implemented for cvglmnet.py


    
 % Cox
    n=1000;p=30;
    nzc=p/3;
    x=randn(n,p);
    beta=randn(nzc,1);
    fx=x(:,1:nzc)*beta/3;
    hx=exp(fx);
    ty=exprnd(1./hx,n,1);
    tcens=binornd(1,0.3,n,1);
    y=cat(2,ty,1-tcens);
    foldid=randsample(10,n,true);
    fit1_cv=cvglmnet(x,y,'cox',[],[],[],foldid);
    cvglmnetPlot(fit1_cv);
    
 % Parallel
    matlabpool;
    x=randn(1e3,100);
    y=randn(1e3,1);
    tic;
    cvglmnet(x,y);
    toc;
    tic;
    cvglmnet(x,y,[],[],[],[],[],true);
    toc;

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
from cvmrelnet import cvmrelnet
from cvfishnet import cvfishnet

def cvglmnet(*, x, \
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
        num_cores = multiprocessing.cpu_count()
        cpredmat = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(doCV)(i, x, y, family, foldid, nfolds, is_offset, **options) for i in range(nfolds))
    else:
        for i in range(nfolds):
            newFit = doCV(i, x, y, family, foldid, nfolds, is_offset, **options)
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
    elif cpredmat[0]['class'] == 'mrelnet':
        cvstuff = cvmrelnet(cpredmat, options['lambdau'], x, y \
                          , options['weights'], options['offset'] \
                          , foldid, ptype, grouped, keep)
    elif cpredmat[0]['class'] == 'fishnet':
        cvstuff = cvfishnet(cpredmat, options['lambdau'], x, y \
                           , options['weights'], options['offset'] \
                           , foldid, ptype, grouped, keep)
    elif cpredmat[0]['class'] == 'coxnet':
        raise NotImplementedError('Cross-validation for coxnet not implemented yet.')
        #cvstuff = cvcoxnet(cpredmat, options['lambdau'], x, y \
        #                  , options['weights'], options['offset'] \
        #                  , foldid, ptype, grouped, keep)
 
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
def doCV(i, x, y, family, foldid, nfolds, is_offset, **options):
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
    return(newFit)
    
