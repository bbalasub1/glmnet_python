# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:48:30 2016

@author: bbalasub
"""
import os
cwd = os.getcwd()
print(cwd)

from example_parallel import testParallel 

testParallel(parallel = True)

