# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 20:27:07 2016

@author: bbalasub
"""
import scipy
import ctypes 
import glmnet 
from glmnetControl import glmnetControl
from glmnetSet import glmnetSet
import importlib

importlib.reload(glmnet)

glmlib = ctypes.cdll.LoadLibrary('./GLMnet.so') # this is a bit of a pain. 
                                                # unless a new python console is started
  
baseDataDir= '/home/bbalasub/Desktop/Summer2016/glmnet/glmnet_R/'
y = scipy.loadtxt(baseDataDir + 'QuickStartExampleY.dat', dtype = scipy.float64)
x = scipy.loadtxt(baseDataDir + 'QuickStartExampleX.dat', dtype = scipy.float64)

# convert x and y to 'F' (fortran) order and scipy float64
y = y.astype(dtype = scipy.float64, order = 'F', copy = True)
x = x.astype(dtype = scipy.float64, order = 'F', copy = True)


glmnet.glmnet(x = x, y = y)
