# Import relevant modules and setup for calling glmnet

import sys
sys.path.append('../test')
sys.path.append('../lib')

import numpy as np
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

# load data
x = np.loadtxt(baseDataDir + 'CoxExampleX.dat', dtype = np.float64, delimiter = ',')
y = np.loadtxt(baseDataDir + 'CoxExampleY.dat', dtype = np.float64, delimiter = ',')

print(y[0:5, :])

# call glmnet
fit = glmnet.glmnet(x = x.copy(), y = y.copy(), family = 'cox')

glmnetPlot.glmnetPlot(fit)

c = glmnetCoef.glmnetCoef(fit, s = np.float64([0.05]))
print(c)
