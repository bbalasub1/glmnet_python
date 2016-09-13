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
    fit = glmnet.glmnet(x = x.copy(), y = y.copy(), family = testType)
    #glmnetPlot.glmnetPlot(fit.copy(), label = True)
    glmnetPlot.glmnetPlot(fit.copy(), xvar = 'lambda', label = True)
    f = glmnetPredict.glmnetPredict(fit.copy(), x[0:10,:])
    
    # cvglmnet, cvglmnetPlot
    fit = cvglmnet.cvglmnet(x = x.copy(), y = y.copy(), family = testType, grouped = True)
    cvglmnetPlot.cvglmnetPlot(fit)

    # print fit
    print('fit:')
    pprint.pprint(fit)

