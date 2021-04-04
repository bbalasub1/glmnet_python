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
x = np.loadtxt(baseDataDir + 'QuickStartExampleX.dat', dtype = np.float64)
y = np.loadtxt(baseDataDir + 'QuickStartExampleY.dat', dtype = np.float64)

# create weights
t = np.ones((50, 1), dtype = np.float64)
wts = np.row_stack((t, 2*t))

# call glmnet
fit = glmnet.glmnet(x = x.copy(), y = y.copy(), family = 'gaussian', \
                    weights = wts, \
                    alpha = 0.2, nlambda = 20
                    )
                    
glmnetPrint.glmnetPrint(fit)
glmnetPlot.glmnetPlot(fit, xvar = 'lambda', label = True)
glmnetPlot.glmnetPlot(fit, xvar = 'dev', label = True)
#
any(fit['lambdau'] == 0.5)
#
coefApprx = glmnetCoef.glmnetCoef(fit, s = np.float64([0.5]), exact = False)
print(coefApprx)
#
fc = glmnetPredict.glmnetPredict(fit, x[0:5,:], ptype = 'response', \
                                s = np.float64([0.05]))
print(fc)
#
cvfit = cvglmnet.cvglmnet(x = x.copy(), y = y.copy(), ptype = 'mse', nfolds = 20) 
cvfit['lambda_min']
cvglmnetCoef.cvglmnetCoef(cvfit, s = 'lambda_min')
#%%
cvglmnetPredict.cvglmnetPredict(cvfit, newx = x[0:5,], s='lambda_min')

#%%
foldid = np.random.choice(10, size = y.shape[0], replace = True)

cv1=cvglmnet.cvglmnet(x = x.copy(),y = y.copy(),foldid=foldid,alpha=1)
cv0p5=cvglmnet.cvglmnet(x = x.copy(),y = y.copy(),foldid=foldid,alpha=0.5)
cv0=cvglmnet.cvglmnet(x = x.copy(),y = y.copy(),foldid=foldid,alpha=0)

#%%
f = plt.figure()
f.add_subplot(2,2,1)
cvglmnetPlot.cvglmnetPlot(cv1)
f.add_subplot(2,2,2)
cvglmnetPlot.cvglmnetPlot(cv0p5)
f.add_subplot(2,2,3)
cvglmnetPlot.cvglmnetPlot(cv0)
f.add_subplot(2,2,4)
plt.plot( np.log(cv1['lambdau']), cv1['cvm'], 'r.')
#plt.hold(True)
plt.plot( np.log(cv0p5['lambdau']), cv0p5['cvm'], 'g.')
plt.plot( np.log(cv0['lambdau']), cv0['cvm'], 'b.')
plt.xlabel('log(Lambda)')
plt.ylabel(cv1['name'])
plt.xlim(-6, 4)
plt.ylim(0, 9)
plt.legend( ('alpha = 1', 'alpha = 0.5', 'alpha = 0'), loc = 'upper left', prop={'size':6})

#%%
plt.figure()
cl = np.array([[-0.7], [0.5]], dtype = np.float64)
tfit=glmnet.glmnet(x = x.copy(),y= y.copy(), cl = cl)
glmnetPlot.glmnetPlot(tfit)

#%%
plt.figure()
pfac = np.ones([1, 20])
pfac[0, 4] = 0; pfac[0, 9] = 0; pfac[0, 14] = 0
pfit = glmnet.glmnet(x = x.copy(), y = y.copy(), penalty_factor = pfac)
glmnetPlot.glmnetPlot(pfit, label = True)

#%%
plt.figure()
np.random.seed(101)
x = np.random.rand(100,10)
y = np.random.rand(100,1)
fit = glmnet.glmnet(x = x, y = y)
glmnetPlot.glmnetPlot(fit)

#%%
plt.figure()
c = glmnetCoef.glmnetCoef(fit)
c = c[1:, -1] # remove intercept and get the coefficients at the end of the path 
h = glmnetPlot.glmnetPlot(fit)
ax1 = h['ax1']
xloc = plt.xlim()
xloc = xloc[1]
for i in range(len(c)):
    ax1.text(xloc, c[i], 'var' + str(i)); 
