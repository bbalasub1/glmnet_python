# -*- coding: utf-8 -*-
"""
This is in pre-alpha stage... only for troubleshooting purposes.

@author: bbalasub
"""

def printDict(s):
    for keys in s.keys():
        try:
            print('%10s  %20s  %20s' % (keys, type(s[keys]), s[keys].shape))
        except:
            print('%10s  %20s' % (keys, type(s[keys])), '  ', s[keys])
            