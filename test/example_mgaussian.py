# Import relevant modules and setup for calling glmnet

import sys
sys.path.append('../test')
sys.path.append('../lib')

import numpy as np
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
x = np.loadtxt(baseDataDir + 'MultiGaussianExampleX.dat', dtype = np.float64, delimiter = ',')
y = np.loadtxt(baseDataDir + 'MultiGaussianExampleY.dat', dtype = np.float64, delimiter = ',')

# call glmnet
mfit = glmnet.glmnet(x = x.copy(), y = y.copy(), family = 'mgaussian')

plt.figure()
glmnetPlot(mfit, xvar = 'lambda', label = True, ptype = '2norm')

f = glmnetPredict.glmnetPredict(mfit, x[0:5,:], s = np.float64([0.1, 0.01]))
print(f[:,:,0])
print(f[:,:,1])

plt.figure()
t = time.time()
cvmfit = cvglmnet.cvglmnet(x = x.copy(), y = y.copy(), family = "mgaussian", parallel = True)
e = time.time() - t
print('time elapsed = ', e)

cvglmnetPlot.cvglmnetPlot(cvmfit)
