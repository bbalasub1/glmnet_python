# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------
 cvglmnetPlot.m: plot the cross-validation curve produced by cvglmnet
--------------------------------------------------------------------------

 DESCRIPTION:
    Plots the cross-validation curve, and upper and lower standard
    deviation curves, as a function of the lambda values used. 

 USAGE:
    cvglmnetPlot(cvfit);
    cvglmnetPlot(cvfit, sign_lambda)
    cvglmnetPlot(cvfit, sign_lambda, options)

 INPUT ARGUMENTS:
 cvobject    fitted "cvglmnet" object
 sign_lambda Either plot against log(lambda) (default) or its negative if
             sign_lambda=-1. 
 varargin    Other errorbar parameters.
 
 DETAILS:
    A plot is produced, and nothing is returned.

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
    cvglmnet and glmnet.

 EXAMPLES:
 
    scipy.random.seed(1)
    x=scipy.random.normal(size = (100,20))
    y=scipy.random.normal(size = (100,1))
    g2=scipy.random.choice(2,size = (100,1))*1.0
    g4=scipy.random.choice(4,size = (100,1))*1.0

    plt.figure()     
    fit1=cvglmnet(x = x.copy(),y = y.copy())
    cvglmnetPlot(fit1)

    plt.figure()
    fit2=cvglmnet(x = x.copy(),y = g2.copy(), family = 'binomial')
    cvglmnetPlot(fit2)

    plt.figure()
    fit3=cvglmnet(x = x.copy(),y = g2.copy(), family = 'binomial', ptype = 'class')
    cvglmnetPlot(fit3)
     
"""

import scipy
import matplotlib.pyplot as plt

def cvglmnetPlot(cvobject, sign_lambda = 1.0, **options):
    
    sloglam = sign_lambda*scipy.log(cvobject['lambdau'])

    fig = plt.gcf()
    ax1 = plt.gca()
    #fig, ax1 = plt.subplots()    
    
    plt.errorbar(sloglam, cvobject['cvm'], cvobject['cvsd'], \
                 ecolor = (0.5, 0.5, 0.5), \
                 **options
                 )
    plt.hold(True)         
    plt.plot(sloglam, cvobject['cvm'], linestyle = 'dashed',\
             marker = 'o', markerfacecolor = 'r')             
    
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    
    xval = sign_lambda*scipy.log(scipy.array([cvobject['lambda_min'], cvobject['lambda_min']]))
    plt.plot(xval, ylim1, color = 'b', linestyle = 'dashed', \
             linewidth = 1)
        
    if cvobject['lambda_min'] != cvobject['lambda_1se']:
        xval = sign_lambda*scipy.log([cvobject['lambda_1se'], cvobject['lambda_1se']])
        plt.plot(xval, ylim1, color = 'b', linestyle = 'dashed', \
             linewidth = 1)

    ax2 = ax1.twiny()
    ax2.xaxis.tick_top()

    atdf = ax1.get_xticks()
    indat = scipy.ones(atdf.shape, dtype = scipy.integer)
    if sloglam[-1] >= sloglam[1]:
        for j in range(len(sloglam)-1, -1, -1):
            indat[atdf <= sloglam[j]] = j
    else:
        for j in range(len(sloglam)):
            indat[atdf <= sloglam[j]] = j

    prettydf = cvobject['nzero'][indat]
    
    ax2.set(XLim=xlim1, XTicks = atdf, XTickLabels = prettydf)
    ax2.grid()
    ax1.yaxis.grid()
    
    ax2.set_xlabel('Degrees of Freedom')
    
  #  plt.plot(xlim1, [ylim1[1], ylim1[1]], 'b')
  #  plt.plot([xlim1[1], xlim1[1]], ylim1, 'b')
    
    if sign_lambda < 0:
        ax1.set_xlabel('-log(Lambda)')
    else:
        ax1.set_xlabel('log(Lambda)')
        
    ax1.set_ylabel(cvobject['name'])
    
    #plt.show()

    
    