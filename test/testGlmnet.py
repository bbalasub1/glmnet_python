# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 11:39:55 2016

@author: bbalasub
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 20:27:07 2016

@author: bbalasub
"""
import sys
sys.path.append('../lib')

import scipy
import ctypes 
import glmnet 
import importlib

importlib.reload(glmnet)

glmlib = ctypes.cdll.LoadLibrary('./GLMnet.so') # this is a bit of a pain. 
                                                # unless a new python console is started

baseDataDir= '/home/bbalasub/Desktop/Summer2016/glmnet/github/glmnet_python/data/'
##  elnet caller 
y = scipy.loadtxt(baseDataDir + 'QuickStartExampleY.dat', dtype = scipy.float64)
x = scipy.loadtxt(baseDataDir + 'QuickStartExampleX.dat', dtype = scipy.float64)

fit = glmnet.glmnet(x = x, y = y, family = 'gaussian')
print('fit:\n', fit)

# lognet caller
y = scipy.loadtxt(baseDataDir + 'BinomialExampleY.dat', dtype = scipy.float64)
