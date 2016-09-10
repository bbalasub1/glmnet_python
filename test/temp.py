# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 22:03:32 2016

@author: bbalasub
"""
import glmnetPredict
importlib.reload(glmnetPredict)

nz = glmnetPredict.glmnetPredict(fit, scipy.empty([0]), scipy.empty([0]), 'nonzero')

