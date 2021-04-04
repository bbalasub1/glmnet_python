# Import relevant modules and setup for calling glmnet

import sys
sys.path.append('../test')
sys.path.append('../lib')

import numpy as np
import importlib
import matplotlib.pyplot as plt
import warnings

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
x = np.loadtxt(baseDataDir + 'MultinomialExampleX.dat', dtype = np.float64, delimiter = ',')
y = np.loadtxt(baseDataDir + 'MultinomialExampleY.dat', dtype = np.float64, delimiter = ',')

# call glmnet
fit = glmnet.glmnet(x = x.copy(), y = y.copy(), family = 'multinomial', mtype = 'grouped')
                    
glmnetPlot.glmnetPlot(fit, xvar = 'lambda', label = True, ptype = '2norm')

warnings.filterwarnings('ignore')
cvfit=cvglmnet.cvglmnet(x = x.copy(), y = y.copy(), family='multinomial', mtype = 'grouped')
warnings.filterwarnings('default')

f = cvglmnetPredict.cvglmnetPredict(cvfit, newx = x[0:10, :], s = 'lambda_min', ptype = 'class')

