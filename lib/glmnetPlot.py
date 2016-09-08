# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:40:57 2016

@author: bbalasub
"""
import matplotlib.pyplot as plt
import scipy

def glmnetPlot(x, xvar = 'norm', label = False, ptype = 'coef', **options):

    # process inputs
    xvar = getFromList(xvar, ['norm', 'lambda', 'dev'], 'xvar should be one of ''norm'', ''lambda'', ''dev'' ')    
    ptype = getFromList(ptype, ['coef', '2norm'], 'ptype should be one of ''coef'', ''2norm'' ')    

    if x['class'] in ['elnet', 'lognet', 'coxnet', 'fishnet']:
        print(x['class'])
        plotCoef(x['beta'], [], x['lambda'], x['df'], x['dev'], '', 'Coefficients', options)

    elif x['class'] in ['multnet', 'mrelnet']:
        beta = x['beta']
        if xvar == 'norm':
            norm = 0
            nzbeta = beta
            for i in range(len(beta)):
                which = nonzeroCoef(beta[i])
                nzbeta[i] = beta[i][which, :]
                norm = norm + scipy.sum(scipy.absolute(nzbeta[i]), axis = 0)
        else:
            norm = 0
        
        if ptype == 'coef':
            ncl = x['dfmat'].shape[0]
            if x['class'] == 'multnet':
                for i in range(ncl):
                    str = 'Coefficients: Class %d' % (i) 
                    print(options)
                    plotCoef(beta[i], norm, x['lambdau'], x['dfmat'][i,:], 
                             x['dev'], label, xvar, '', str, **options)
            else:
                    str = 'Coefficients: Response %d' % (i) 
                    plotCoef(beta[i], norm, x['lambdau'], x['dfmat'][i,:], 
                             x['dev'], label, xvar, '', str, **options)
        else:
            dfseq = scipy.round_(scipy.mean(x['dfmat'], axis = 0))
            coefnorm = beta[1]*0
            for i in range(beta):
                coefnorm = coefnorm + scipy.absolute(beta[i])**2
            coefnorm = scipy.sqrt(coefnorm)
            if x['class'] == 'multnet':
                str = 'Coefficient 2Norms'
                plotCoef(coefnorm, norm, x['lambdau'], dfseq, x['dev'],
                         label, xvar, '',str, **options);
            else:
                str = 'Coefficient 2Norms'
                plotCoef(coefnorm, norm, x['lambdau'], x['dfmat'][0,:], x['dev'],
                         label, xvar, '', str, **options);                





# =========================================
# helper functions
# =========================================
def getFromList(xvar, xvarbase, errMsg):
    indxtf = [x.startswith(xvar.lower()) for x in xvarbase] # find index 
    xvarind = [i for i in range(len(indxtf)) if indxtf[i] == True]
    if len(xvarind) == 0:
        raise ValueError(errMsg)
    else:
        xvar = xvarbase[xvarind[0]]
    return xvar    
# end of getFromList()
# =========================================
def nonzeroCoef(beta, bystep = False):
    result = scipy.absolute(beta) > 0
    if not bystep:
        result = scipy.any(result, axis = 1)
    
    return(result)
# end of nonzeroCoef()
# =========================================
def plotCoef(beta, norm, lambdau, df, dev, label, xvar, xlab, ylab, **options):
    which = nonzeroCoef(beta)
    idwhich = [i for i in range(len(which)) if which[i] == True]
    nwhich = len(idwhich)
    if nwhich == 0:
        raise ValueError('No plot produced since all coefficients are zero')
    elif nwhich == 1:
        raise ValueError('1 or less nonzero coefficients; glmnet plot is not meaningful')
    
    beta = beta[which, :]
    if xvar == 'norm':
        if len(norm) == 0:
            index = scipy.sum(scipy.absolute(beta), axis = 0)
        else:
            index = norm
        iname = 'L1 Norm'
    elif xvar == 'lambda':
        index = scipy.log(lambdau)
        iname = 'Log Lambda'
    elif xvar == 'dev':
        index = dev
        iname = 'Fraction Deviance Explained'
        
    if len(xlab) == 0:
        xlab = iname

    # prepare for figure    
    plt.figure()    
    # plot x vs y
    beta = scipy.transpose(beta)
    plt.plot(index, beta, **options)
    
    # TODO: draw lambdau and df axes on the figure
    
    
    # TODO: put label
    
    
    
# plotCoef
# =========================================
