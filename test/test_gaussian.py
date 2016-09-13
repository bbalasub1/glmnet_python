"""
Created on Mon Sep 12 21:53:57 2016

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
import glmnetPrint
import glmnetPlot
import glmnetPredict
import cvglmnet
import cvglmnetPlot

importlib.reload(glmnet)
importlib.reload(glmnetPrint)    
importlib.reload(glmnetPlot)    
importlib.reload(glmnetPredict)    
importlib.reload(cvglmnet)    
importlib.reload(cvglmnetPlot)

# parameters
baseDataDir= '../data/'
testTypeList = ['gaussian', 'binomial', 'multinomial', 'cox', 'mgaussian', 'poisson']
testType = 'gaussian'
runType = 'cvglmnet'  # runType is cvglmnet or glmnet

# call test functions
if testType == 'gaussian':
    ##  elnet caller 
    y = scipy.loadtxt(baseDataDir + 'QuickStartExampleY.dat', dtype = scipy.float64)
    x = scipy.loadtxt(baseDataDir + 'QuickStartExampleX.dat', dtype = scipy.float64)
    
    #
    r1 = scipy.ones((50, 1), dtype = scipy.float64)
    weights = scipy.row_stack((r1, 2*r1))
    fit = glmnet.glmnet(x = x.copy(), y = y.copy(), family = testType, \
                        weights = weights, \
                        alpha = 0.2, nlambda = 20
                        )
    glmnetPrint.glmnetPrint(fit)

    
    # glmnet, glmnetPlot, glmnetPredict
    fit = glmnet.glmnet(x = x.copy(), y = y.copy(), family = testType)
    glmnetPlot.glmnetPlot(fit, label = True)
    glmnetPlot.glmnetPlot(fit, xvar = 'lambda', label = True)
    f = glmnetPredict.glmnetPredict(fit, x[0:10,:])
    
    # cvglmnet, cvglmnetPlot
    fit = cvglmnet.cvglmnet(x = x.copy(), y = y.copy(), family = testType)
    cvglmnetPlot.cvglmnetPlot(fit)
    # pprint
    print('fit:')
    pprint.pprint(fit)

