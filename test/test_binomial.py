# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 22:08:51 2016

@author: bbalasub
"""

#%%
import sys
sys.path.append('../test')
sys.path.append('../lib')

import scipy
import importlib
import matplotlib.pyplot as plt

import glmnet 
import glmnetPlot
import glmnetPrint
import glmnetCoef
import glmnetPredict

import cvglmnet
import cvglmnetCoef
import cvglmnetPlot
import cvglmnetPredict

importlib.reload(glmnet)
importlib.reload(glmnetPlot)    
importlib.reload(glmnetPrint)
importlib.reload(glmnetCoef)    
importlib.reload(glmnetPredict)

importlib.reload(cvglmnet)    
importlib.reload(cvglmnetCoef)
importlib.reload(cvglmnetPlot)
importlib.reload(cvglmnetPredict)

# parameters
baseDataDir= '../data/'
#testTypeList = ['gaussian', 'binomial', 'multinomial', 'cox', 'mgaussian', 'poisson']
testType = 'binomial'
runType = 'cvglmnet'  # runType is cvglmnet or glmnet

# call test functions
if testType == 'binomial':
    ##  binomial caller 
    x = scipy.loadtxt(baseDataDir + 'BinomialExampleX.dat', dtype = scipy.float64, delimiter = ',')
    y = scipy.loadtxt(baseDataDir + 'BinomialExampleY.dat', dtype = scipy.float64)

    # glmnet, glmnetPlot, glmnetPredict
    fit = glmnet.glmnet(x = x.copy(), y = y.copy(), family = testType)
    glmnetPlot.glmnetPlot(fit, label = True)
    glmnetPlot.glmnetPlot(fit, xvar = 'lambda', label = True)
    f = glmnetPredict.glmnetPredict(fit, x[0:10,:])
    
    # cvglmnet, cvglmnetPlot
    fit = cvglmnet.cvglmnet(x = x.copy(), y = y.copy(), family = testType, ptype='class')
    cvglmnetPlot.cvglmnetPlot(fit)
    # pprint
    print('fit:')
    pprint.pprint(fit)

