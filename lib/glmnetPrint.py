# -*- coding: utf-8 -*-
"""

--------------------------------------------------------------------------
 glmnetPrint.m: print a glmnet object
--------------------------------------------------------------------------

 DESCRIPTION:
    Print a summary of the glmnet path at each step along the path.

 USAGE: 
    glmnetPrint(fit)

 INPUT ARGUMENTS:
    fit         fitted glmnet object

 DETAILS:
    Three-column matrix with columns Df, %Dev and Lambda is printed. The Df
    column is the number of nonzero coefficients (Df is a reasonable name
    only for lasso fits). %Dev is the percent deviance explained (relative
    to the null deviance).

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

 SEE ALSO:
    glmnet, glmnetSet, glmnetPredict and glmnetCoef methods.
 
 EXAMPLES:
    x = scipy.random.normal(size=[100,20])
    y = scipy.random.normal(size=[100,1])
    fit=glmnet(x = x,y = y);
    glmnetPrint(fit);

"""

def glmnetPrint(fit):

    print('\t df \t %dev \t lambdau\n')
    N = fit['lambdau'].size
    for i in range(N):
        line_p = '%d \t %f \t %f \t %f' % (i, fit['df'][i], fit['dev'][i], fit['lambdau'][i])
        print(line_p)


