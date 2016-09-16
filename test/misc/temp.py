# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:56:56 2016

@author: bbalasub
"""

# Import relevant modules and setup for calling glmnet

import sys
sys.path.append('../../test')
sys.path.append('../../lib')

import scipy
import importlib
import matplotlib.pyplot as plt
import time

import glmnet 
from glmnetPlot import glmnetPlot
import glmnetPrint
import glmnetCoef
import glmnetPredict

import cvglmnet
import cvglmnetCoef
import cvglmnetPlot
import cvglmnetPredict

importlib.reload(glmnet)
#importlib.reload(glmnetPlot)    
importlib.reload(glmnetPrint)
importlib.reload(glmnetCoef)    
importlib.reload(glmnetPredict)

importlib.reload(cvglmnet)    
importlib.reload(cvglmnetCoef)
importlib.reload(cvglmnetPlot)
importlib.reload(cvglmnetPredict)

# parameters
baseDataDir= '../data/'

# load data
N = 40000
K = 50
Nc= 2
nfolds = 16
x = scipy.random.rand(N, K)
y = scipy.random.rand(N, Nc)

# call glmnet
t = time.time()
cvmfit = cvglmnet.cvglmnet(x , y = y, family = "mgaussian", parallel = True, nfolds = nfolds)
e1 = time.time()
cvmfit = cvglmnet.cvglmnet(x , y = y, family = "mgaussian", parallel = False, nfolds = nfolds)
e2 = time.time()

print('time elapsed (parallel) = ', e1 - t)
print('time elapsed (serial) = ', e2 - e1)


