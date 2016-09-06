# -*- coding: utf-8 -*-
"""

@author: bbalasub
"""

def glmnetPrint(fit):

    print('\t df \t %dev \t lambdau\n')
    N = fit['lambdau'].size
    for i in range(N):
        line_p = '%d \t %f \t %f \t %f' % (i, fit['df'][i], fit['dev'][i], fit['lambdau'][i])
        print(line_p)


