# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 21:31:01 2016

@author: bbalasub
"""
from cvglmnetCoef import cvglmnetCoef
from glmnetPredict import glmnetPredict

def cvglmnetPredict(obj, newx = None, s = 'lambda_1se', **options):
    if newx is None:
        CVpred = cvglmnetCoef(obj)
        return(CVpred)
        
    if isinstance(s, (int, float)):
        lambdau = s
    else:
        if s in ['lambda_1se', 'lambda_min']:
            lambdau = obj[s]
        else:
            raise ValueError('Invalid form for s')
    
    CVpred = glmnetPredict(obj['glmnet_fit'], newx, lambdau, **options)
    
    return(CVpred)
    