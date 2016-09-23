import sys
sys.path.append('../../test')
sys.path.append('../../lib')
import scipy
import glmnet
from glmnetPlot import glmnetPlot
from glmnetPredict import glmnetPredict
from glmnetCoef import glmnetCoef
from cvglmnet import cvglmnet
from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot
from cvglmnetPredict import cvglmnetPredict

import importlib
import time
importlib.reload(glmnet)

section = 9

if section == 1:
    # create x and y
    scipy.random.seed(1)
    x = scipy.random.normal(size = [10,3])
    y = scipy.random.binomial(1, 0.5, size =[10,1])*1.0
    x[x < 0.0] = 0.0
    
    # x is made sparse
    xs = scipy.sparse.csc_matrix(x, dtype = scipy.float64)
    print("xs = ", xs.todense())
    
    # nobs, nvars can be obtained from sparse x
    # xs is now sparse
    nobs, nvars = xs.shape
    
    # 
    tfs = xs[:,0] > 1.0
    tfs =  tfs.toarray();
    tf = scipy.reshape(tfs, [len(tfs), ])

elif section == 2:
    # sparse caller for glmnet
    baseDataDir= '../../data/'
    
    # load data
    x = scipy.loadtxt(baseDataDir + 'QuickStartExampleX.dat', dtype = scipy.float64)
    y = scipy.loadtxt(baseDataDir + 'QuickStartExampleY.dat', dtype = scipy.float64)
    xs = scipy.sparse.csc_matrix(x, dtype = scipy.float64)
    scipy.random.seed(1)
    lambda_min = scipy.random.rand(y.size)
    exclude = scipy.array([0, 1, 2, 3])
    penalty_factor = scipy.ones(x.shape[1])
    penalty_factor[0] = 500
    pmax = scipy.array([18])
    fit = glmnet.glmnet(x = xs.copy(), y = y.copy(), family = 'gaussian', exclude = exclude)
    print(fit['a0'])
    print(fit['beta'][:,-1])

elif section == 3:
      N = 1000;
      family = 'binomial'
      x = scipy.random.normal(size = [N,10])
      x[x < 2.0] = 0.0
      xs = scipy.sparse.csc_matrix(x, dtype = scipy.float64)
      y = scipy.random.normal(size = [N,1])
      y[y > 0] = 1.0
      y[y < 0] = 0.0
      st = time.time()
      fit = glmnet.glmnet(x = xs, y = y, family = family)
      en = time.time()
      print("time elapsed (sparse) = ", en - st)
      print("nbytes = ", xs.data.nbytes)
      st = time.time()
      fit = glmnet.glmnet(x = x, y = y, family = family)
      en = time.time()
      print("time elapsed (full) = ", en - st)
      print("nbytes = ", x.data.nbytes)

elif section == 4:
      N = 1000;
      family = 'binomial'
      x = scipy.random.normal(size = [N,10])
      x[x < 2.0] = 0.0
      xs = scipy.sparse.csc_matrix(x, dtype = scipy.float64)
      y = scipy.random.normal(size = [N,1])
      y[y > 0] = 1.0
      y[y < 0] = 0.0
      st = time.time()
      fit = cvglmnet.cvglmnet(x = xs, y = y, family = family)
      en = time.time()
      print("time elapsed (sparse) = ", en - st)

elif section == 5:
     import matplotlib.pyplot as plt
     scipy.random.seed(1)
     x=scipy.random.normal(size = (100,20))
     y=scipy.random.normal(size = (100,1))
     g4=scipy.random.choice(4,size = (100,1))*1.0
     fit1=glmnet.glmnet(x = x.copy(),y = y.copy())
     glmnetPlot(fit1)
     plt.figure()
     glmnetPlot(fit1, 'lambda', True);
     fit3=glmnet.glmnet(x = x.copy(),y = g4.copy(), family = 'multinomial')
     plt.figure()
     glmnetPlot(fit3)

elif section == 6:
      x = scipy.random.rand(100, 10)
      y = scipy.random.rand(100, 1)
      fit = glmnet.glmnet(x = x, y = y)
      f = glmnetPredict(fit, x[0:5, :], scipy.array([0.0866, 0.2323]))
      print(f)
      
elif section == 7:
    x = scipy.random.normal(size = [100,20])
    y = scipy.random.normal(size = [100,1])
    g2 = scipy.random.choice(2, size = [100, 1])*1.0
    g4 = scipy.random.choice(4, size = [100, 1])*1.0
    
    fit1 = glmnet.glmnet(x = x.copy(),y = y.copy());
    print( glmnetPredict(fit1,x[0:5,:],scipy.array([0.01,0.005])) )
    print( glmnetPredict(fit1, scipy.empty([0]), scipy.empty([0]), 'coefficients') )
    
    fit2 = glmnet.glmnet(x = x.copy(), y = g2.copy(), family = 'binomial');
    print(glmnetPredict(fit2, x[2:5,:],scipy.empty([0]), 'response'))
    print(glmnetPredict(fit2, scipy.empty([0]), scipy.empty([0]), 'nonzero'))
       
    fit3 = glmnet.glmnet(x = x.copy(), y = g4.copy(), family = 'multinomial');
    print(glmnetPredict(fit3, x[0:3,:], scipy.array([0.01]), 'response'))
    print(glmnetPredict(fit3, x[0:3,:], scipy.array([0.01, 0.5]), 'response'))
      
elif section == 8:
    x=scipy.random.rand(100,20);
    y=scipy.random.rand(100,1);
    fit=glmnet.glmnet(x = x.copy(),y = y.copy());
    ncoef=glmnetCoef(fit,scipy.array([0.01, 0.001]));


elif section == 9:
    scipy.random.seed(1)
    x=scipy.random.normal(size = (100,20))
    y=scipy.random.normal(size = (100,1))
    g2=scipy.random.choice(2,size = (100,1))*1.0
    g4=scipy.random.choice(4,size = (100,1))*1.0

    plt.figure()     
    fit1=cvglmnet(x = x.copy(),y = y.copy())
    cvglmnetPlot(fit1)

    plt.figure()
    fit2=cvglmnet(x = x.copy(),y = g2.copy(), family = 'binomial')
    cvglmnetPlot(fit2)

    plt.figure()
    fit3=cvglmnet(x = x.copy(),y = g2.copy(), family = 'binomial', ptype = 'class')
    cvglmnetPlot(fit3)
    
    
    