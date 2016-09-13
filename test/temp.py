# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 11:39:55 2016

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
testType = testTypeList[1]
runType = 'cvglmnet'  # runType is cvglmnet or glmnet

# call test functions
if testType == 'gaussian':
    ##  elnet caller 
    y = scipy.loadtxt(baseDataDir + 'QuickStartExampleY.dat', dtype = scipy.float64)
    x = scipy.loadtxt(baseDataDir + 'QuickStartExampleX.dat', dtype = scipy.float64)

if testType == 'binomial':
    # lognet caller
    x = scipy.loadtxt(baseDataDir + 'BinomialExampleX.dat', dtype = scipy.float64, delimiter = ',')
    y = scipy.loadtxt(baseDataDir + 'BinomialExampleY.dat', dtype = scipy.float64)

if testType == 'multinomial':
    # multinomial caller
    x = scipy.loadtxt(baseDataDir + 'MultinomialExampleX.dat', dtype = scipy.float64, delimiter = ',')
    y = scipy.loadtxt(baseDataDir + 'MultinomialExampleY.dat', dtype = scipy.float64, delimiter = ',')

if testType == 'cox':
    # coxnet caller
    x = scipy.loadtxt(baseDataDir + 'CoxExampleX.dat', dtype = scipy.float64, delimiter = ',')
    y = scipy.loadtxt(baseDataDir + 'CoxExampleY.dat', dtype = scipy.float64, delimiter = ',')

if testType == 'mgaussian':
    # mgaussian caller
    x = scipy.loadtxt(baseDataDir + 'MultiGaussianExampleX.dat', dtype = scipy.float64, delimiter = ',')
    y = scipy.loadtxt(baseDataDir + 'MultiGaussianExampleY.dat', dtype = scipy.float64, delimiter = ',')
    
if testType == 'poisson':
    # poisson caller
    x = scipy.loadtxt(baseDataDir + 'PoissonExampleX.dat', dtype = scipy.float64, delimiter = ',')
    y = scipy.loadtxt(baseDataDir + 'PoissonExampleY.dat', dtype = scipy.float64, delimiter = ',')

if runType == 'cvglmnet':
    fit = cvglmnet.cvglmnet(x = x, y = y, family = testType, ptype='class')
    cvglmnetPlot.cvglmnetPlot(fit)
elif runType == 'glmnet':
    fit = glmnet.glmnet(x = x, y = y, family = testType)
    glmnetPlot.glmnetPlot(fit, label = True)
    glmnetPlot.glmnetPlot(fit, xvar = 'lambda', label = True)
    f = glmnetPredict.glmnetPredict(fit, x[0:1,:])

print('fit:')
pprint.pprint(fit)



#%%
#import cvglmnetCoef
#import importlib
#importlib.reload(cvglmnetCoef)
#
#coef = cvglmnetCoef.cvglmnetCoef(fit, s= scipy.array([0.1]))
#print('coef=', coef)


