# -*- coding: utf-8 -*-
"""
def loadGlmLib():
=======================
INPUT ARGUMENTS:

                NONE

=======================
OUTPUT ARGUMENTS: 

glmlib          Returns a glmlib object with methods that are equivalent 
                to the fortran functions in GLMnet.f
=======================
"""
import ctypes
import os

def loadGlmLib():
    if os.name == 'posix':
        glmlib = ctypes.cdll.LoadLibrary('../glmnet_python/GLMnet.so') 
        return(glmlib)
    elif os.name == 'nt':
        # this does not currently work
        raise ValueError('loadGlmlib does not currently work for windows')
        glmlib = ctypes.windll.LoadLibrary('../glmnet_python/GLMnet.dll')
    else:
        raise ValueError('loadGlmLib not yet implemented for non-posix OS')
        
