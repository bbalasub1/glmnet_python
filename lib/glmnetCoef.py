# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 11:14:27 2016

@author: bbalasub
"""

import scipy
from glmnetPredict import glmnetPredict

def glmnetCoef(obj, s = None, exact = False):
    
    if s is None:
        s = obj['lambdau']
    
    if exact and len(s) > 0:
        raise NotImplementedError('exact = True not implemented in glmnetCoef')
        
    result = glmnetPredict(obj, scipy.empty([0]), s, 'coefficients')    
    
    return(result)
    
    