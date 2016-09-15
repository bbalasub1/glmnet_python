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
x = scipy.loadtxt(baseDataDir + 'BinomialExampleX.dat', dtype = scipy.float64, delimiter = ',')
y = scipy.loadtxt(baseDataDir + 'BinomialExampleY.dat', dtype = scipy.float64)

# call glmnet
fit = glmnet.glmnet(x = x.copy(), y = y.copy(), family = 'binomial')
                    
glmnetPlot.glmnetPlot(fit, xvar = 'dev', label = True);

glmnetPredict.glmnetPredict(fit, newx = x[0:5,], ptype='class', s = scipy.array([0.05, 0.01]))

cvfit = cvglmnet.cvglmnet(x = x.copy(), y = y.copy(), family = 'binomial', ptype = 'class')

plt.figure()
cvglmnetPlot.cvglmnetPlot(cvfit)

cvfit['lambda_min']

cvfit['lambda_1se']

cvglmnetCoef.cvglmnetCoef(cvfit, s = 'lambda_min')

cvglmnetPredict.cvglmnetPredict(cvfit, newx = x[0:10, ], s = 'lambda_min', ptype = 'class')






