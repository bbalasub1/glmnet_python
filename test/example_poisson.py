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
x = np.loadtxt(baseDataDir + 'PoissonExampleX.dat', dtype = np.float64, delimiter = ',')
y = np.loadtxt(baseDataDir + 'PoissonExampleY.dat', dtype = np.float64, delimiter = ',')

# call glmnet
fit = glmnet.glmnet(x = x.copy(), y = y.copy(), family = 'poisson')

glmnetPlot.glmnetPlot(fit)

glmnetCoef.glmnetCoef(fit, s = np.float64([1.0]))

f = glmnetPredict.glmnetPredict(fit, x[0:5,:], ptype = 'response', s = np.float64([0.1, 0.01]))
print(f)

cvfit = cvglmnet.cvglmnet(x.copy(), y.copy(), family = 'poisson')
optlam = np.array([cvfit['lambda_min'], cvfit['lambda_1se']]).reshape(2,)
cvglmnetCoef.cvglmnetCoef(cvfit, s = optlam)
