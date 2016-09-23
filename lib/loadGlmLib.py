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
        glmlib = ctypes.cdll.LoadLibrary('../../lib/GLMnet.so') 
        return(glmlib)
    else:
        raise ValueError('loadGlmLib not yet implemented for non-posix OS')
        
