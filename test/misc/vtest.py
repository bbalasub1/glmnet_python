# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 19:16:05 2016

@author: bbalasub
"""

def vtest(*, x, y, fittype = 'gaussian', **kwargs):
    x = kwargs.get('age')
    print(x)
    print('t=', fittype)
       