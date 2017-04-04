# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------
 glmnetCoef computes coefficients from a "glmnet" object.
--------------------------------------------------------------------------

 DESCRIPTION:
    This function extracts coefficients at certain lambdas if they are
    in the lambda sequence of a "glmnet" object or make predictions
    if they are not in that sequence.

 USAGE:
    glmnetCoef(object, s, exact)

    Fewer input arguments (more often) are allowed in the call, but must
    come in the order listed above. To set default values on the way, use
    scipy.empty([0]). 
    For example, ncoef = glmnetCoef(fit,scipy.empty([0]),False).

 INPUT ARGUMENTS:
    obj      Fitted "glmnet" model object.
    s        Value(s) of the penalty parameter lambda at which computation
             is required. Default is the entire sequence used to create
             the model.
    exact    If exact = False (default), then the function uses
             linear interpolation to make predictions for values of s
             that do not coincide with those used in the fitting
             algorithm. Note that exact = True is not implemented.

 OUTPUT ARGUMENTS:
    result   A (nvars+1) x length(s) scipy 2D array with each column being the 
             coefficients at an s. Note that the first row are the 
             intercepts (0 if no intercept in the original model).

 LICENSE: GPL-2

 AUTHORS:
    Algorithm was designed by Jerome Friedman, Trevor Hastie and Rob Tibshirani
    Fortran code was written by Jerome Friedman
    R wrapper (from which the MATLAB wrapper was adapted) was written by Trevor Hasite
    The original MATLAB wrapper was written by Hui Jiang,
    and is updated and maintained by Junyang Qian.
    This Python wrapper (adapted from the Matlab and R wrappers) 
    is written by Balakumar B.J., bbalasub@stanford.edu 
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
    glmnet, glmnetPrint, glmnetPredict, and cvglmnet.

 EXAMPLES:
    x = scipy.random.rand(100,20);
    y = scipy.random.rand(100,1);
    fit = glmnet(x = x.copy(),y = y.copy());
    ncoef = glmnetCoef(fit,scipy.array([0.01, 0.001]));

"""

import scipy
from glmnetPredict import glmnetPredict

def glmnetCoef(obj, s = None, exact = False):
    
    if s is None:
        s = obj['lambdau']
    
    if exact and len(s) > 0:
        raise NotImplementedError('exact = True not implemented in glmnetCoef')
        
    result = glmnetPredict(obj, scipy.empty([0]), s, 'coefficients')    
    
    return(result)
    
    