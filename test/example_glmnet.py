# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 11:39:55 2016

@author: bbalasub
"""

import sys
sys.path.append('../test')
sys.path.append('../lib')

import numpy as np
import glmnet 
import importlib
import pprint
import glmnetPlot
import glmnetPredict

importlib.reload(glmnet)
importlib.reload(glmnetPlot)    
importlib.reload(glmnetPredict)    

# parameters
baseDataDir= '../data/'
testTypeList = ['gaussian', 'binomial', 'multinomial', 'cox', 'mgaussian', 'poisson']
testType = testTypeList[0]

# call test functions
if testType == 'gaussian':
    ##  elnet caller 
    y = np.loadtxt(baseDataDir + 'QuickStartExampleY.dat', dtype = np.float64)
    x = np.loadtxt(baseDataDir + 'QuickStartExampleX.dat', dtype = np.float64)
    fit = glmnet.glmnet(x = x, y = y, family = 'gaussian')
    #fit = glmnet.glmnet(x = x, y = y, family = 'gaussian', alpha = 0.5)
    print('fit:')
    pprint.pprint(fit)

if testType == 'binomial':
    # lognet caller
    x = np.loadtxt(baseDataDir + 'BinomialExampleX.dat', dtype = np.float64, delimiter = ',')
    y = np.loadtxt(baseDataDir + 'BinomialExampleY.dat', dtype = np.float64)
    fit = glmnet.glmnet(x = x, y = y, family = 'binomial')
    print('fit:')
    pprint.pprint(fit)

if testType == 'multinomial':
    # multinomial caller
    x = np.loadtxt(baseDataDir + 'MultinomialExampleX.dat', dtype = np.float64, delimiter = ',')
    y = np.loadtxt(baseDataDir + 'MultinomialExampleY.dat', dtype = np.float64, delimiter = ',')
    fit = glmnet.glmnet(x = x, y = y, family = 'multinomial')
    print('fit:')
    pprint.pprint(fit)    

if testType == 'cox':
    # coxnet caller
    x = np.loadtxt(baseDataDir + 'CoxExampleX.dat', dtype = np.float64, delimiter = ',')
    y = np.loadtxt(baseDataDir + 'CoxExampleY.dat', dtype = np.float64, delimiter = ',')
    fit = glmnet.glmnet(x = x, y = y, family = 'cox')
    print('fit:')
    pprint.pprint(fit)

if testType == 'mgaussian':
    # mgaussian caller
    x = np.loadtxt(baseDataDir + 'MultiGaussianExampleX.dat', dtype = np.float64, delimiter = ',')
    y = np.loadtxt(baseDataDir + 'MultiGaussianExampleY.dat', dtype = np.float64, delimiter = ',')
    fit = glmnet.glmnet(x = x, y = y, family = 'mgaussian')
    print('fit:')
    pprint.pprint(fit)    
    
if testType == 'poisson':
    # poisson caller
    x = np.loadtxt(baseDataDir + 'PoissonExampleX.dat', dtype = np.float64, delimiter = ',')
    y = np.loadtxt(baseDataDir + 'PoissonExampleY.dat', dtype = np.float64, delimiter = ',')
    fit = glmnet.glmnet(x = x, y = y, family = 'poisson')
    print('fit:')
    pprint.pprint(fit)


glmnetPlot.glmnetPlot(fit, label = True)
# glmnetPlot.glmnetPlot(fit, xvar = 'lambda', label = True)

f = glmnetPredict.glmnetPredict(fit, x[0:1,:])


