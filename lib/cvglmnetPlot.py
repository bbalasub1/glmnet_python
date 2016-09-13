# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 12:31:37 2016

@author: bbalasub
"""

import scipy
import matplotlib.pyplot as plt

def cvglmnetPlot(cvobject, sign_lambda = 1.0, **options):
    
    sloglam = sign_lambda*scipy.log(cvobject['lambdau'])

    fig, ax1 = plt.subplots()    
    
    plt.errorbar(sloglam, cvobject['cvm'], cvobject['cvsd'], \
                 ecolor = (0.5, 0.5, 0.5), \
                 **options
                 )
    plt.hold(True)         
    plt.plot(sloglam, cvobject['cvm'], linestyle = 'dashed',\
             marker = 'o', markerfacecolor = 'r')             
    
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    
    xval = sign_lambda*scipy.log(scipy.array([cvobject['lambda_min'], cvobject['lambda_min']]))
    plt.plot(xval, ylim1, color = 'b', linestyle = 'dashed', \
             linewidth = 1)
        
    if cvobject['lambda_min'] != cvobject['lambda_1se']:
        xval = sign_lambda*scipy.log([cvobject['lambda_1se'], cvobject['lambda_1se']])
        plt.plot(xval, ylim1, color = 'b', linestyle = 'dashed', \
             linewidth = 1)

    ax2 = ax1.twiny()
    ax2.xaxis.tick_top()

    atdf = ax1.get_xticks()
    indat = scipy.ones(atdf.shape, dtype = scipy.integer)
    if sloglam[-1] >= sloglam[1]:
        for j in range(len(sloglam)-1, -1, -1):
            indat[atdf <= sloglam[j]] = j
    else:
        for j in range(len(sloglam)):
            indat[atdf <= sloglam[j]] = j

    prettydf = cvobject['nzero'][indat]
    
    ax2.set(XLim=xlim1, XTicks = atdf, XTickLabels = prettydf)
    ax2.grid()
    ax1.yaxis.grid()
    
    ax2.set_xlabel('Degrees of Freedom')
    
  #  plt.plot(xlim1, [ylim1[1], ylim1[1]], 'b')
  #  plt.plot([xlim1[1], xlim1[1]], ylim1, 'b')
    
    if sign_lambda < 0:
        ax1.set_xlabel('-log(Lambda)')
    else:
        ax1.set_xlabel('log(Lambda)')
        
    ax1.set_ylabel(cvobject['name'])
    
    #plt.show()

    
    