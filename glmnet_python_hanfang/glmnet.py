# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------
glmnet.py: 
    Fit a GLM with lasso or elastic-net regularization.
    glmnet.py provides a wrapper to the glmnet fortran routines. All
    variables in the arguments are keyword-only. (see examples below). 
--------------------------------------------------------------------------

DESCRIPTION:
-----------
    Fit a generalized linear model via penalized maximum likelihood. The 
    regularization path is computed for the lasso or elasticnet penalty 
    at a grid of values for the regularization parameter lambda. Can deal 
    with all shapes of data, including very large sparse data matrices. 
    Fits linear, logistic and multinomial, Poisson, and Cox regression 
    models.
    
EXTERNAL FUNCTIONS:
------------------
    options = glmnetSet()   # provided with this (glmnet python) package    
    
INPUT ARGUMENTS:
---------------
  x        Input scipy 2D array of nobs x nvars (required). Each row is an 
           observation vector. Can be in sparse matrix format. Must be in 
           scipy csc_matrix format
           
  y        Response variable (scipy 2D array of size nobs x 1, nobs x nc, etc). (required) 
           For family = 'gaussian', Quantitative column vector
           For family = 'poisson' (non-negative counts), Quantitative column vector
           For family = 'binomial', should be either a column vector with two
             levels or a two column matrix of counts of proportions. 
           For family = 'multinomial', can be a column vector of nc >= 2 levels
             or a matrix with nc columns of counts or proportions.
           For family = 'cox', y should be a two-column array with the first column
             for time and the second for status. The latter is a binary variable, 
             with 1 indicating death and 0 indicating right censored. 
           For family = 'mgaussian', y is an array of quantitative responses.
           (see examples for illustrations)
           
  family   Response type. Default is 'gaussian'. (optional)
           Currently, 'gaussian', 'poisson', 'binomial', 'multinomial', 'mgaussian'
           and 'cox' are supported

  options  optional parameters that can be set and altered by glmnetSet()
           Default values for some often used parameters:
             alpha = 1.0 (elastic-net mixing parameter)
             nlambda = 100 (number of lambda values)
             lambdau depends on data, nlambda and lambda_min (user supplied lambda sequence)
             standardize = True (variable standardization)
             weights = all ones scipy vector (observation weights)
           For more details see help for glmnetSet   

OUTPUT ARGUMENTS: 
----------------
fit        glmnet(...) outputs a dict() of fit parameters with the following keys:

a0         Intercept sequence of length len(fit['lambdau'])

beta       For 'elnet' and 'lognet' models, nvars x len(lambdau) array of coefficients
           For 'multnet', a list of nc such matrices, one for each class

lambdau    The actual sequence of lambdau values used

dev        The fraction of (null) deviance explained (for 'elnet', this is the R-squared)

nulldev    Null deviance (per observation)

df         The number of nonzero coefficients for each value of lambdau.
           For 'multnet', this is the number of variables with a nonezero 
           coefficient for any class

dfmat      For 'multnet' only: A 2D array consisting of the number of nonzero 
           coefficients per class

dim        Dimension of coefficient matrix (ices)

npasses    Total passes over the data summed over all lambdau values

offset     A logical variable indicating whether an offset was included in the model

jerr       Error flag, for warnings and errors (largely for internal debugging)

class      Type of regression - internal usage

EXAMPLES:
--------
      # Gaussian
      x = scipy.random.rand(100, 10)
      y = scipy.random.rand(100, 1)
      fit = glmnet(x = x, y = y)
      fit = glmnet(x = x, y = y, alpha = 0.5)
      glmnetPrint(fit)
      glmnetPredict(fit, scipy.empty([0]), scipy.array([0.01]), 'coef') # extract coefficients at a single value of lambdau
      glmnetPredict(fit, x[0:10,:], scipy.array([0.01, 0.005])) # make predictions

      # Multivariate Gaussian:
      x = scipy.random.rand(100, 10)
      y = scipy.random.rand(100,3)
      fit = glmnet(x, y, 'mgaussian')      
      glmnetPlot(fit, 'norm', False, '2norm')
      
      # Binomial
      x = scipy.random.rand(100, 10)
      y = scipy.random.rand(100,1)
      y = (y > 0.5)*1.0
      fit = glmnet(x = x, y = y, family = 'binomial', alpha = 0.5)    
      
      # Multinomial
      x = scipy.random.rand(100,10)
      y = scipy.random.rand(100,1)
      y[y < 0.3] = 1.0
      y[y < 0.6] = 2.0
      y[y < 1.0] = 3.0
      fit = glmnet(x = x, y = y, family = 'multinomial', mtype = 'grouped')

      # poisson
      x = scipy.random.rand(100,10)
      y = scipy.random.poisson(size = [100, 1])*1.0
      fit = glmnet(x = x, y = y, family = 'poisson')
      
      # cox
      N = 1000; p = 30;
      nzc = p/3;
      x = scipy.random.normal(size = [N, p])
      beta = scipy.random.normal(size = [nzc, 1])
      fx = scipy.dot(x[:, 0:nzc], beta/3)
      hx = scipy.exp(fx)
      ty = scipy.random.exponential(scale = 1/hx, size = [N, 1])
      tcens = scipy.random.binomial(1, 0.3, size = [N, 1])
      tcens = 1 - tcens
      y = scipy.column_stack((ty, tcens))
      fit = glmnet(x = x.copy(), y = y.copy(), family = 'cox')
      glmnetPlot(fit)
      
      # sparse example
      N = 1000000;
      x = scipy.random.normal(size = [N,10])
      x[x < 3.0] = 0.0
      xs = scipy.sparse.csc_matrix(x, dtype = scipy.float64)
      y = scipy.random.binomial(1, 0.5, size =[N,1])
      y = y*1.0
      st = time.time()
      fit = glmnet.glmnet(x = xs, y = y, family = 'binomial')
      en = time.time()
      print("time elapsed (sparse) = ", en - st)
      print("nbytes = ", xs.data.nbytes)
      # non-sparse (same as sparse case)      
      st = time.time()
      fit = glmnet.glmnet(x = x, y = y, family = 'binomial')
      en = time.time()
      print("time elapsed (full) = ", en - st)
      print("nbytes = ", x.data.nbytes)
 
DETAILS:
-------
   The sequence of models implied by lambda is fit by coordinate descent.
   For family='gaussian' this is the lasso sequence if alpha=1, else it
   is the elasticnet sequence. For the other families, this is a lasso or
   elasticnet regularization path for fitting the generalized linear
   regression paths, by maximizing the appropriate penalized
   log-likelihood (partial likelihood for the 'cox' model). Sometimes the
   sequence is truncated before nlambda values of lambda have been used,
   because of instabilities in the inverse link functions near a
   saturated fit. glmnet(...,family='binomial') fits a traditional
   logistic regression model for the log-odds.
   glmnet(...,family='multinomial') fits a symmetric multinomial model,
   where each class is represented by a linear model (on the log-scale).
   The penalties take care of redundancies. A two-class 'multinomial'
   model will produce the same fit as the corresponding 'binomial' model,
   except the pair of coefficient matrices will be equal in magnitude and
   opposite in sign, and half the 'binomial' values. Note that the
   objective function for 'gaussian' is

                   1/2 RSS / nobs + lambda * penalty,
                   
   and for the logistic models it is

                    -loglik / nobs + lambda * penalty.

    Note also that for 'gaussian', glmnet standardizes y to have unit
    variance before computing its lambda sequence (and then unstandardizes
    the resulting coefficients); if you wish to reproduce/compare results
    with other software, best to supply a standardized y. The latest two
    features in glmnet are the family='mgaussian' family and the
    mtype='grouped' in options for multinomial fitting. The former
    allows a multi-response gaussian model to be fit, using a "group
    -lasso" penalty on the coefficients for each variable. Tying the
    responses together like this is called "multi-task" learning in some
    domains. The grouped multinomial allows the same penalty for the
    family='multinomial' model, which is also multi-responsed. For both of
    these the penalty on the coefficient vector for variable j is

            (1-alpha)/2 * ||beta_j||_2^2 + alpha * ||beta_j||_2

    When alpha=1 this is a group-lasso penalty, and otherwise it mixes
    with quadratic just like elasticnet. 

LICENSE:
-------
    GPL-2

AUTHORS:
-------
    Algorithm was designed by Jerome Friedman, Trevor Hastie and Rob Tibshirani
    Fortran code was written by Jerome Friedman
    R wrapper (from which the MATLAB wrapper was adapted) was written by Trevor Hasite
    The original MATLAB wrapper was written by Hui Jiang,
    and is updated and maintained by Junyang Qian.
    This Python wrapper (adapted from the Matlab and R wrappers) 
    is written by Balakumar B.J., bbalasub@stanford.edu 
    Department of Statistics, Stanford University, Stanford, California, USA.

REFERENCES:
---------- 
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
--------
    glmnetPrint, glmnetPlot, glmnetCoef, glmnetPredict,
    glmnetSet, glmnetControl and cvglmnet.

"""

# import packages/methods
from glmnetSet import glmnetSet
from glmnetControl import glmnetControl
import scipy
from elnet import elnet
from lognet import lognet
from coxnet import coxnet
from mrelnet import mrelnet
from fishnet import fishnet

def glmnet(*, x, y, family='gaussian', **options):
        
    # check inputs: make sure x and y are scipy, float64 arrays
    # fortran order is not checked as we force a convert later 
    if not( isinstance(x, scipy.sparse.csc.csc_matrix) ):
        if not( isinstance(x, scipy.ndarray) and x.dtype == 'float64'):
            raise ValueError('x input must be a scipy float64 ndarray')
    else:
        if not (x.dtype == 'float64'):
            raise ValueError('x input must be a float64 array')
            
    if not( isinstance(y, scipy.ndarray) and y.dtype == 'float64'):
            raise ValueError('y input must be a scipy float64 ndarray')

    # create options
    if options is None:
        options = glmnetSet();
    
    ## match the family, abbreviation allowed
    fambase = ['gaussian','binomial','poisson','multinomial','cox','mgaussian'];
    # find index of family in fambase
    indxtf = [x.startswith(family.lower()) for x in fambase] # find index of family in fambase
    famind = [i for i in range(len(indxtf)) if indxtf[i] == True]
    if len(famind) == 0:
        raise ValueError('Family should be one of ''gaussian'', ''binomial'', ''poisson'', ''multinomial'', ''cox'', ''mgaussian''')
    elif len(famind) > 1:
        raise ValueError('Family could not be uniquely determined : Use a longer description of the family string.')        
    else:
        family = fambase[famind[0]] 
    
    ## prepare options
    options = glmnetSet(options)
    #print('glmnet.py options:')
    #print(options)
    
    ## error check options parameters
    alpha = scipy.float64(options['alpha'])
    if alpha > 1.0 :
        print('Warning: alpha > 1.0; setting to 1.0')
        options['alpha'] = scipy.float64(1.0)
 
    if alpha < 0.0 :
        print('Warning: alpha < 0.0; setting to 0.0')
        options['alpha'] = scipy.float64(0.0)

    parm  = scipy.float64(options['alpha'])
    nlam  = scipy.int32(options['nlambda'])
    nobs, nvars  = x.shape
    
    # check weights length
    weights = options['weights']
    if len(weights) == 0:
        weights = scipy.ones([nobs, 1], dtype = scipy.float64)
    elif len(weights) != nobs:
        raise ValueError('Error: Number of elements in ''weights'' not equal to number of rows of ''x''')
    # check if weights are scipy nd array
    if not( isinstance(weights, scipy.ndarray) and weights.dtype == 'float64'):
        raise ValueError('weights input must be a scipy float64 ndarray')
    
    # check y length
    nrowy = y.shape[0]
    if nrowy != nobs:
        raise ValueError('Error: Number of elements in ''y'' not equal to number of rows of ''x''')
    
    # check ne   
    ne = options['dfmax']
    if len(ne) == 0:
        ne = nvars + 1

    # check nx
    nx = options['pmax']
    if len(nx) == 0:
        nx = min(ne*2 + 20, nvars)

    # check jd
    exclude = options['exclude']
    # TBD: test this
    if not (len(exclude) == 0):
        exclude = scipy.unique(exclude)
        if scipy.any(exclude < 0) or scipy.any(exclude >= nvars):
            raise ValueError('Error: Some excluded variables are out of range')
        else:    
            jd = scipy.append(len(exclude), exclude + 1) # indices are 1-based in fortran
    else:
        jd = scipy.zeros([1,1], dtype = scipy.integer)

    # check vp    
    vp = options['penalty_factor']
    if len(vp) == 0:
        vp = scipy.ones([1, nvars])
    
    # inparms
    inparms = glmnetControl()
    
    # cl
    cl = options['cl']
    if any(cl[0,:] > 0):
        raise ValueError('Error: The lower bound on cl must be non-positive')

    if any(cl[1,:] < 0):
        raise ValueError('Error: The lower bound on cl must be non-negative')
        
    cl[0, cl[0, :] == scipy.float64('-inf')] = -1.0*inparms['big']    
    cl[1, cl[1, :] == scipy.float64('inf')]  =  1.0*inparms['big']    
    
    if cl.shape[1] < nvars:
        if cl.shape[1] == 1:
            cl = cl*scipy.ones([1, nvars])
        else:
            raise ValueError('Error: Require length 1 or nvars lower and upper limits')
    else:
        cl = cl[:, 0:nvars]
        
        
    exit_rec = 0
    if scipy.any(cl == 0.0):
        fdev = inparms['fdev']
        if fdev != 0:
            optset = dict()
            optset['fdev'] = 0
            glmnetControl(optset)
            exit_rec = 1
             
    isd  = scipy.int32(options['standardize'])
    intr = scipy.int32(options['intr'])
    if (intr == True) and (family == 'cox'):
        print('Warning: Cox model has no intercept!')
        
    jsd        = scipy.int32(options['standardize_resp'])
    thresh     = options['thresh']    
    lambdau    = options['lambdau']
    lambda_min = options['lambda_min']
    
    if len(lambda_min) == 0:
        if nobs < nvars:
            lambda_min = 0.01
        else:
            lambda_min = 1e-4
    
    lempty = (len(lambdau) == 0)
    if lempty:
        if (lambda_min >= 1):
            raise ValueError('ERROR: lambda_min should be less than 1')
        flmin = lambda_min
        ulam  = scipy.zeros([1,1], dtype = scipy.float64)
    else:
        flmin = 1.0
        if any(lambdau < 0):
            raise ValueError('ERROR: lambdas should be non-negative')
        
        ulam = -scipy.sort(-lambdau)    # reverse sort
        nlam = lambdau.size
    
    maxit =  scipy.int32(options['maxit'])
    gtype = options['gtype']
    if len(gtype) == 0:
        if (nvars < 500):
            gtype = 'covariance'
        else:
            gtype = 'naive'
    
    # ltype
    ltype = options['ltype']
    ltypelist = ['newton', 'modified.newton']
    indxtf    = [x.startswith(ltype.lower()) for x in ltypelist]
    indl      = [i for i in range(len(indxtf)) if indxtf[i] == True]
    if len(indl) != 1:
        raise ValueError('ERROR: ltype should be one of ''Newton'' or ''modified.Newton''')
    else:
        kopt = indl[0]
    
    if family == 'multinomial':
        mtype = options['mtype']
        mtypelist = ['ungrouped', 'grouped']
        indxtf    = [x.startswith(mtype.lower()) for x in mtypelist]
        indm      = [i for i in range(len(indxtf)) if indxtf[i] == True]
        if len(indm) == 0:
            raise ValueError('Error: mtype should be one of ''ungrouped'' or ''grouped''')
        elif (indm == 2):
            kopt = 2
    #
    offset = options['offset']
    # sparse (if is_sparse, convert to compressed sparse row format)   
    is_sparse = False
    if scipy.sparse.issparse(x):
        is_sparse = True
        tx = scipy.sparse.csc_matrix(x, dtype = scipy.float64)
        x = tx.data; x = x.reshape([len(x), 1])
        irs = tx.indices + 1
        pcs = tx.indptr + 1
        irs = scipy.reshape(irs, [len(irs),])
        pcs = scipy.reshape(pcs, [len(pcs),])        
    else:
        irs = scipy.empty([0])
        pcs = scipy.empty([0])
        
    if scipy.sparse.issparse(y):
        y = y.todense()
    
    ## finally call the appropriate fit code
    if family == 'gaussian':
        # call elnet
        fit = elnet(x, is_sparse, irs, pcs, y, weights, offset, gtype, parm, 
                    lempty, nvars, jd, vp, cl, ne, nx, nlam, flmin, ulam, 
                    thresh, isd, intr, maxit, family)
    elif (family == 'binomial') or (family == 'multinomial'):
        # call lognet
        fit = lognet(x, is_sparse, irs, pcs, y, weights, offset, parm,
                     nobs, nvars, jd, vp, cl, ne, nx, nlam, flmin, ulam,
                     thresh, isd, intr, maxit, kopt, family)
    elif family == 'cox':
        # call coxnet
        fit = coxnet(x, is_sparse, irs, pcs, y, weights, offset, parm,
                     nobs, nvars, jd, vp, cl, ne, nx, nlam, flmin, ulam,
                     thresh, isd, maxit, family)
    elif family == 'mgaussian':
        # call mrelnet
        fit = mrelnet(x, is_sparse, irs, pcs, y, weights, offset, parm, 
                      nobs, nvars, jd, vp, cl, ne, nx, nlam, flmin, ulam, 
                      thresh, isd, jsd, intr, maxit, family)
    elif family == 'poisson':
        # call fishnet
        fit = fishnet(x, is_sparse, irs, pcs, y, weights, offset, parm,
                      nobs, nvars, jd, vp, cl, ne, nx, nlam, flmin, ulam,
                      thresh, isd, intr, maxit, family); 
    else:
        raise ValueError('calling a family of fits that has not been implemented yet')
            
    if exit_rec == 1:
        optset['fdev'] = fdev
        #TODO: Call glmnetControl(optset) to set persistent parameters
        
    # return fit
    return fit

#----------------------------------------- 
# end of method glmnet   
#----------------------------------------- 
