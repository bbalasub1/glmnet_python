# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 10:55:19 2016

@author: bbalasub
"""

# loads all modules so they can be tested
rootDir = '/home/bbalasub/Desktop/Summer2016/glmnet/github/glmnet_python/lib/'
from glmnet import glmnet
from glmnetSet import glmnetSet
glmnet(1,2,family = 'gaussian')

x = {'a': 1, 'b': 2, 'd' : 4}
y = {'a': 100, 'b': 200, 'c' : 300}
