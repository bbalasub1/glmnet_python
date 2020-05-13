# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 20:27:07 2016

@author: bbalasub
"""
import numpy as np
import ctypes 
import glmnet 
import importlib

importlib.reload(glmnet)

glmlib = ctypes.cdll.LoadLibrary('../../lib/GLMnet.so') # this is a bit of a pain. 
                                                # unless a new python console is started
  
baseDataDir= '/home/bbalasub/Desktop/Summer2016/glmnet/glmnet_R/'
y = np.loadtxt(baseDataDir + 'QuickStartExampleY.dat', dtype = np.float64)
x = np.loadtxt(baseDataDir + 'QuickStartExampleX.dat', dtype = np.float64)

fit = glmnet.glmnet(x = x, y = y)
