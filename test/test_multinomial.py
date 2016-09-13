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
import glmnet 
import importlib
import pprint
import glmnetPlot
import glmnetPredict
import cvglmnet
import cvglmnetPlot

importlib.reload(glmnet)
importlib.reload(glmnetPlot)    
importlib.reload(glmnetPredict)    
importlib.reload(cvglmnet)    
importlib.reload(cvglmnetPlot)

# parameters
baseDataDir= '../data/'
testTypeList = ['gaussian', 'binomial', 'multinomial', 'cox', 'mgaussian', 'poisson']
testType = 'multinomial'
runType = 'cvglmnet'  # runType is cvglmnet or glmnet

# call test functions
if testType == 'multinomial':
    ##  multinomial caller 
    x = scipy.loadtxt(baseDataDir + 'MultinomialExampleX.dat', dtype = scipy.float64, delimiter = ',')
    y = scipy.loadtxt(baseDataDir + 'MultinomialExampleY.dat', dtype = scipy.float64, delimiter = ',')

    # glmnet, glmnetPlot, glmnetPredict
    #fit = glmnet.glmnet(x = x, y = y, family = testType)
    #glmnetPlot.glmnetPlot(fit, label = True)
    #glmnetPlot.glmnetPlot(fit, xvar = 'lambda', label = True)
    #f = glmnetPredict.glmnetPredict(fit, x[0:1,:])
    
    # cvglmnet, cvglmnetPlot
    fit = cvglmnet.cvglmnet(x = x, y = y, family = testType, grouped = True)
    cvglmnetPlot.cvglmnetPlot(fit)
    # pprint
    print('fit:')
    pprint.pprint(fit)

