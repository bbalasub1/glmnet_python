import sys
sys.path.append('../../test')
sys.path.append('../../lib')
import scipy
import glmnet
import importlib
import time
importlib.reload(glmnet)

section = 3;

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
    x = scipy.loadtxt(baseDataDir + 'BinomialExampleX.dat', dtype = scipy.float64, delimiter = ',')
    y = scipy.loadtxt(baseDataDir + 'BinomialExampleY.dat', dtype = scipy.float64)
    xs = scipy.sparse.csc_matrix(x, dtype = scipy.float64)
    fit = glmnet.glmnet(x = xs.copy(), y = y.copy(), family = 'binomial')

elif section == 3:
      N = 1000000;
      x = scipy.random.normal(size = [N,100])
      x[x < 2.0] = 0.0
      xs = scipy.sparse.csc_matrix(x, dtype = scipy.float64)
      y = scipy.random.normal(size = [N,1])
      y[y > 0] = 1.0
      y[y < 0] = 0.0
      st = time.time()
      fit = glmnet.glmnet(x = xs, y = y, family = 'binomial')
      en = time.time()
      print("time elapsed (sparse) = ", en - st)
      print("nbytes = ", xs.data.nbytes)
      st = time.time()
      fit = glmnet.glmnet(x = x, y = y, family = 'binomial')
      en = time.time()
      print("time elapsed (full) = ", en - st)
      print("nbytes = ", x.data.nbytes)
