import sys
sys.path.append('../../test')
sys.path.append('../../lib')
import scipy
import glmnet
import importlib
importlib.reload(glmnet)

x = scipy.random.normal(size = [1000,10])
x[x < 1.0] = 0.0
xs = scipy.sparse.csc_matrix(x, dtype = scipy.float64)
y = scipy.random.binomial(1, 0.5, size =[1000,1])*1.0
fit = glmnet.glmnet(x = xs, y = y, family = 'binomial')


