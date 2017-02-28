# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------
 cvglmnetCoef computes coefficients from a "cvglmnet" object.
--------------------------------------------------------------------------

 DESCRIPTION:
    This function extracts coefficients at certain lambdas if they are
    in the lambda sequence of a "cvglmnet" object or make predictions
    if they are not.

 USAGE:
    mcoef=cvglmnetCoef(object);
    ncoef=cvglmnetCoef(object, s);

 INPUT ARGUMENTS:
    obj      Fitted "cvglmnet" model object.
    s        Value(s) of the penalty parameter lambdau at which computation
             is required. Default is the value s='lambda_1se' stored on
             the CV object. Alternatively s='lambda_min' can be used. If s
             is numeric, it is taken as the value(s) of lambda to be used.

 OUTPUT ARGUMENTS:
    result   If s is 'lambda_1se' or 'lambda_min', the coefficients at 
             that s is returned. If s is numeric, a (nvars+1) x length(s) 
             matrix is returned with each column being the coefficients 
             at an s. Note that the first row are the intercepts (0 if no 
             intercept in the original model).

 DETAILS:
    The function uses linear interpolation to make predictions for values 
    of s that do not coincide with those used in the fitting algorithm. 
    Exact prediction is not supported currently.

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
    cvglmnet, cvglmnetPrint, and cvglmnetPredict.

 EXAMPLES:
    x=randn(100,20);
    y=randn(100,1);
    cvfit=cvglmnet(x,y);
    ncoef=cvglmnetCoef(cvfit,'lambda_min');
    
"""

import scipy
from glmnetCoef import glmnetCoef

def cvglmnetCoef(obj, s = None):
    
    if s is None or len(s) == 0:
        s = obj['lambda_1se']
        
    if isinstance(s, scipy.ndarray):
        lambdau = s
    elif isinstance(s, str):
        sbase = ['lambda_1se', 'lambda_min']
        indxtf = [x.startswith(s.lower()) for x in sbase] # find index of family in fambase
        sind= [i for i in range(len(indxtf)) if indxtf[i] == True]
        s = sbase[sind[0]]
        lambdau = obj[s]
    else:
        raise ValueError('Invalid form of s')
        
    result = glmnetCoef(obj['glmnet_fit'], lambdau)
    
    return(result)
    
    