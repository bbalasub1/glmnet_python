# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:42:37 2016

@author: bbalasub
"""
import joblib
import multiprocessing

def testParallel(parallel = True):
    
    inputs = range(0, 1000, 1)
    param = 1000
    if parallel == True:
    # parallel stuff
    # This is reference code for parallel implementation 
        inputs = range(10)
        num_cores = multiprocessing.cpu_count()
        results = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(childFunc)(i, param) for i in inputs)

    else:
        for i in inputs:
            childFunc(i)

    print(results)
    
def childFunc(i, param):
	    return i + param
