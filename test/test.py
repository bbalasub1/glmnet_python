# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 23:17:51 2016

@author: bbalasub
"""
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

scipy.random.seed(101)
x = scipy.random.rand(100,10)
y = scipy.random.rand(100,1)
fit = glmnet.glmnet(x = x, y = y)
h = glmnetPlot.glmnetPlot(fit)
c = glmnetCoef.glmnetCoef(fit)
# remove intercept and pick fit for last lambda 
c = c[1:, -1]
ax1 = plt.gca()
xloc = plt.xlim()
yloc = plt.ylim()
xloc = xloc[1]
yloc = 0.5*(yloc[0] + yloc[1])
for i in range(len(c)):
    ax1.text(xloc, c[i], 'var' + str(i))
    