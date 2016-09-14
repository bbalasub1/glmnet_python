# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 11:04:20 2016

@author: bbalasub
"""

import scipy
from glmnetCoef import glmnetCoef

def cvglmnetCoef(obj, s = None):
    
    if s is None or len(s) == 0:
        s = obj['lambda_1se']
        
    if isinstance(s, scipy.ndarray):
        lambdau = s
    elif isinstance(s, str):
        sbase = ['lambda_1se', 'lambda_min']
        indxtf = [x.startswith(s.lower()) for x in sbase] # find index of family in fambase
        sind= [i for i in range(len(indxtf)) if indxtf[i] == True]
        s = sbase[sind[0]]
        lambdau = obj[s]
    else:
        raise ValueError('Invalid form of s')
        
    result = glmnetCoef(obj['glmnet_fit'], lambdau)
    
    return(result)
    
    