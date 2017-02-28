# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------
 glmnetPlot.m: plot coefficients from a "glmnet" object
--------------------------------------------------------------------------

 DESCRIPTION:
    Produces a coefficient profile plot fo the coefficient paths for a
    fitted "glmnet" object.

 USAGE:
    glmnetPlot(fit);
    glmnetPlot(fit, xvar = 'norm');
    glmnetPlot(fit, xvar = 'norm', label = False);
    glmnetPlot(fit, xvar = 'norm', label = False, ptype = 'coef');
    glmnetPlot(fit, xvar = 'norm', label = False, ptype = 'coef', ...);

 INPUT ARGUMENTS:
 x           fitted "glmnet" model.
 xvar        What is on the X-axis. 'norm' plots against the L1-norm of
             the coefficients, 'lambda' against the log-lambda sequence,
             and 'dev' against the percent deviance explained.
 label       If true, label the curves with variable sequence numbers.
 type        If type='2norm' then a single curve per variable, else
             if type='coef', a coefficient plot per response.
 varargin    Other graphical parameters to plot.

 DETAILS:
    A coefficient profile plot is produced. If x is a multinomial model, a
    coefficient plot is produced for each class.

 LICENSE: 
    GPL-2

 AUTHORS:
    Algorithm was designed by Jerome Friedman, Trevor Hastie and Rob Tibshirani
    Fortran code was written by Jerome Friedman
    R wrapper (from which the MATLAB wrapper was adapted) was written by Trevor Hasite
    The original MATLAB wrapper was written by Hui Jiang,
    and is updated and maintained by Junyang Qian.
    This Python wrapper (adapted from the Matlab and R wrappers) is written by Balakumar B.J., 
    Department of Statistics, Stanford University, Stanford, California, USA.

 REFERENCES:
    Friedman, J., Hastie, T. and Tibshirani, R. (2008) Regularization Paths for Generalized Linear Models via Coordinate Descent, 
    http://www.jstatsoft.org/v33/i01/
    Journal of Statistical Software, Vol. 33(1), 1-22 Feb 2010
    
    Simon, N., Friedman, J., Hastie, T., Tibshirani, R. (2011) Regularization Paths for Cox's Proportional Hazards Model via Coordinate Descent,
    http://www.jstatsoft.org/v39/i05/
    Journal of Statistical Software, Vol. 39(5) 1-13

    Tibshirani, Robert., Bien, J., Friedman, J.,Hastie, T.,Simon, N.,Taylor, J. and Tibshirani, Ryan. (2010) Strong Rules for Discarding Predictors in Lasso-type Problems,
    http://www-stat.stanford.edu/~tibs/ftp/strong.pdf
    Stanford Statistics Technical Report

 SEE ALSO:
    glmnet, glmnetSet, glmnetPrint, glmnetPredict and glmnetCoef.

 EXAMPLES:
     import matplotlib.pyplot as plt
     scipy.random.seed(1)
     x=scipy.random.normal(size = (100,20))
     y=scipy.random.normal(size = (100,1))
     g4=scipy.random.choice(4,size = (100,1))*1.0
     fit1=glmnet(x = x.copy(),y = y.copy())
     glmnetPlot(fit1)
     plt.figure()
     glmnetPlot(fit1, 'lambda', True);
     fit3=glmnet(x = x.copy(),y = g4.copy(), family = 'multinomial')
     plt.figure()
     glmnetPlot(fit3)
"""
import matplotlib.pyplot as plt
import scipy

def glmnetPlot(x, xvar = 'norm', label = False, ptype = 'coef', **options):

    # process inputs
    xvar = getFromList(xvar, ['norm', 'lambda', 'dev'], 'xvar should be one of ''norm'', ''lambda'', ''dev'' ')    
    ptype = getFromList(ptype, ['coef', '2norm'], 'ptype should be one of ''coef'', ''2norm'' ')    

    if x['class'] in ['elnet', 'lognet', 'coxnet', 'fishnet']:
        handle = plotCoef(x['beta'], [], x['lambdau'], x['df'], x['dev'], 
        label, xvar, '', 'Coefficients', **options)

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
                    handle = plotCoef(beta[i], norm, x['lambdau'], x['dfmat'][i,:], 
                             x['dev'], label, xvar, '', str, **options)
                    if i < ncl - 1:         
                        plt.figure()             
            else:
                    str = 'Coefficients: Response %d' % (i) 
                    handle = plotCoef(beta[i], norm, x['lambdau'], x['dfmat'][i,:], 
                             x['dev'], label, xvar, '', str, **options)
        else:
            dfseq = scipy.round_(scipy.mean(x['dfmat'], axis = 0))
            coefnorm = beta[1]*0
            for i in range(len(beta)):
                coefnorm = coefnorm + scipy.absolute(beta[i])**2
            coefnorm = scipy.sqrt(coefnorm)
            if x['class'] == 'multnet':
                str = 'Coefficient 2Norms'
                handle = plotCoef(coefnorm, norm, x['lambdau'], dfseq, x['dev'],
                         label, xvar, '',str, **options);
                if i < ncl - 1:                         
                    plt.figure()         
            else:
                str = 'Coefficient 2Norms'
                handle = plotCoef(coefnorm, norm, x['lambdau'], x['dfmat'][0,:], x['dev'],
                         label, xvar, '', str, **options);                

    return(handle)
# end of glmnetplot
# =========================================
#
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
    if len(result.shape) == 1:
        result = scipy.reshape(result, [result.shape[0], 1])
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

    # draw the figures
    #fig, ax1 = plt.subplots()    
    fig = plt.gcf()
    ax1 = plt.gca()
    
    # plot x vs y
    beta = scipy.transpose(beta)
    ax1.plot(index, beta, **options)
    
    ax2 = ax1.twiny()
    ax2.xaxis.tick_top()
    
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    
    atdf = ax1.get_xticks()
    indat = scipy.ones(atdf.shape, dtype = scipy.integer)
    if index[-1] >= index[1]:
        for j in range(len(index)-1, -1, -1):
            indat[atdf <= index[j]] = j
    else:
        for j in range(len(index)):
            indat[atdf <= index[j]] = j
    prettydf = df[indat]
    prettydf[-1] = df[-1]        
    
    ax2.set(XLim=[min(index), max(index)], XTicks = atdf, XTickLabels = prettydf)
    ax2.grid()
    ax1.yaxis.grid()

    ax2.set_xlabel('Degrees of Freedom')    
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    
    # put the labels
    if label:
        xpos = max(index)
        adjpos = 1
        if xvar == 'lambda':
            xpos = min(index)
            adjpos = 0
        bsize = beta.shape
        for i in range(beta.shape[1]):
            str = '%d' % idwhich[i]
            ax1.text(1/2*xpos + 1/2*xlim1[adjpos], beta[bsize[0]-1, i], str)
    
    plt.show()
    
    handle = dict()
    handle['fig'] = fig
    handle['ax1'] = ax1
    handle['ax2'] = ax2
    return(handle)

# end of plotCoef
# =========================================
