# Import relevant modules and setup for calling glmnet

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

# parameters
baseDataDir= '../data/'

# load data
x = scipy.loadtxt(baseDataDir + 'PoissonExampleX.dat', dtype = scipy.float64, delimiter = ',')
y = scipy.loadtxt(baseDataDir + 'PoissonExampleY.dat', dtype = scipy.float64, delimiter = ',')

# call glmnet
fit = glmnet.glmnet(x = x.copy(), y = y.copy(), family = 'poisson')

glmnetPlot.glmnetPlot(fit)

glmnetCoef.glmnetCoef(fit, s = scipy.float64([1.0]))

f = glmnetPredict.glmnetPredict(fit, x[0:5,:], ptype = 'response', s = scipy.float64([0.1, 0.01]))
print(f)

cvfit = cvglmnet.cvglmnet(x.copy(), y.copy(), family = 'poisson')
optlam = scipy.array([cvfit['lambda_min'], cvfit['lambda_1se']]).reshape(2,)
cvglmnetCoef.cvglmnetCoef(cvfit, s = optlam)
