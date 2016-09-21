# -*- coding: utf-8 -*-
"""
Prints some info about a dictionary object. Used for troubleshooting.

"""

def printDict(s):
    for keys in s.keys():
        try:
            print('%10s  %20s  %20s' % (keys, type(s[keys]), s[keys].shape))
        except:
            print('%10s  %20s' % (keys, type(s[keys])), '  ', s[keys])
            