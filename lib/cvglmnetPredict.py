# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------
 cvglmnetPredict makes predictions from a "cvglmnet" object.
--------------------------------------------------------------------------

 DESCRIPTION:
    This function makes predictions from a cross-validated glmnet model,
    using the stored "glmnet_fit" object, and the optimal value chosen for
    lambda.

 USAGE:
    pred = cvglmnetPredict(cvfit)
    pred = cvglmnetPredict(cvfit, newx)
    pred = cvglmnetPredict(cvfit, newx, s)
    pred = cvglmnetPredict(cvfit, newx, s, ...)

 INPUT ARGUMENTS:
 object      Fitted "glmnet" model object.
 newx        2D array of new values for x at which predictions are to be
             made. Must be a 2D array; can be sparse. See documentation for 
             glmnetPredict.
 s           Value(s) of the penalty parameter lambda at which predictions
             are required. Default is the value s='lambda_1se' stored on
             the CV object. Alternatively s='lambda_min' can be used. If s
             is numeric, it is taken as the value(s) of lambda to be used.
             If s is numeric, it must be a scipy 1D array.
 options     Other arguments to predict (see glmnetPredict).

 OUTPUT ARGUMENTS:
    If only the cvglmnet object is provided, the function returns the 
    coefficients at the default s = 'lambda_1se'. Otherwise, the object 
    returned depends on the ... argument which is passed on to the 
    glmnetPredict for glmnet objects.
             

 DETAILS:
    This function makes it easier to use the results of cross-validation
    to make a prediction. 

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
    cvglmnet and glmnetPredict.

 EXAMPLES:
      x = scipy.random.rand(100, 10)
      y = scipy.random.rand(100, 1)
      cvfit = cvglmnet(x = x, y = y)
      cvglmnetPredict(cvfit, x[0:5, :], 'lambda_min')
      cvglmnetPredict(cvfit, x[0:5, :], scipy.array([0.0866, 0.2323]))

"""
from cvglmnetCoef import cvglmnetCoef
from glmnetPredict import glmnetPredict
import scipy

def cvglmnetPredict(obj, newx = None, s = 'lambda_1se', **options):
    if newx is None:
        CVpred = cvglmnetCoef(obj)
        return(CVpred)
        
    if type(s) == scipy.ndarray and s.dtype == 'float64':
        lambdau = s
    elif s in ['lambda_1se', 'lambda_min']:
            lambdau = obj[s]
    else:
            raise ValueError('Invalid form for s')
    
    CVpred = glmnetPredict(obj['glmnet_fit'], newx, lambdau, **options)
    
    return(CVpred)
    