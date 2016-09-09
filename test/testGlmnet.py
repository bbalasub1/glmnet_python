# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 11:39:55 2016

@author: bbalasub
"""

import sys
sys.path.append('../test')
sys.path.append('../lib')

import scipy
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
    y = scipy.loadtxt(baseDataDir + 'QuickStartExampleY.dat', dtype = scipy.float64)
    x = scipy.loadtxt(baseDataDir + 'QuickStartExampleX.dat', dtype = scipy.float64)
    fit = glmnet.glmnet(x = x, y = y, family = 'gaussian')
    #fit = glmnet.glmnet(x = x, y = y, family = 'gaussian', alpha = 0.5)
    print('fit:')
    pprint.pprint(fit)

if testType == 'binomial':
    # lognet caller
    x = scipy.loadtxt(baseDataDir + 'BinomialExampleX.dat', dtype = scipy.float64, delimiter = ',')
    y = scipy.loadtxt(baseDataDir + 'BinomialExampleY.dat', dtype = scipy.float64)
    fit = glmnet.glmnet(x = x, y = y, family = 'binomial')
    print('fit:')
    pprint.pprint(fit)

if testType == 'multinomial':
    # multinomial caller
    x = scipy.loadtxt(baseDataDir + 'MultinomialExampleX.dat', dtype = scipy.float64, delimiter = ',')
    y = scipy.loadtxt(baseDataDir + 'MultinomialExampleY.dat', dtype = scipy.float64, delimiter = ',')
    fit = glmnet.glmnet(x = x, y = y, family = 'multinomial')
    print('fit:')
    pprint.pprint(fit)    

if testType == 'cox':
    # coxnet caller
    x = scipy.loadtxt(baseDataDir + 'CoxExampleX.dat', dtype = scipy.float64, delimiter = ',')
    y = scipy.loadtxt(baseDataDir + 'CoxExampleY.dat', dtype = scipy.float64, delimiter = ',')
    fit = glmnet.glmnet(x = x, y = y, family = 'cox')
    print('fit:')
    pprint.pprint(fit)

if testType == 'mgaussian':
    # mgaussian caller
    x = scipy.loadtxt(baseDataDir + 'MultiGaussianExampleX.dat', dtype = scipy.float64, delimiter = ',')
    y = scipy.loadtxt(baseDataDir + 'MultiGaussianExampleY.dat', dtype = scipy.float64, delimiter = ',')
    fit = glmnet.glmnet(x = x, y = y, family = 'mgaussian')
    print('fit:')
    pprint.pprint(fit)    
    
if testType == 'poisson':
    # poisson caller
    x = scipy.loadtxt(baseDataDir + 'PoissonExampleX.dat', dtype = scipy.float64, delimiter = ',')
    y = scipy.loadtxt(baseDataDir + 'PoissonExampleY.dat', dtype = scipy.float64, delimiter = ',')
    fit = glmnet.glmnet(x = x, y = y, family = 'poisson')
    print('fit:')
    pprint.pprint(fit)


glmnetPlot.glmnetPlot(fit, label = True)
# glmnetPlot.glmnetPlot(fit, xvar = 'lambda', label = True)

f = glmnetPredict.glmnetPredict(fit, x[0:1,:])


