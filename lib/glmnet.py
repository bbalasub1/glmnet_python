# -*- coding: utf-8 -*-
"""
glmnet:

Pre:
    x <numpy array of nobs x nvars, required>       : regression x variable 
    y <numpy array of nobs x 1 or more, required>   : regression y variable 
    family <string, optional>                       : family to fit (gaussian, binomail, poisson, multinomial, cox, mgaussian)
    options <dict, optional>                        : fit parameters
    
Post: 
    

@author: bbalasub
"""
from glmnetSet import glmnetSet
from glmnetControl import glmnetControl
import numpy as np

def glmnet(x, y, family = 'gaussian', options = None):
    
    ## set input default options
    # set options to default if nothing is passed in    
    if options is None:
        options = glmnetSet();
    
    ## match the family, abbreviation allowed
    fambase = ['gaussian','binomial','poisson','multinomial','cox','mgaussian'];
    # find index of family in fambase
    indxtf = [x.startswith(family) for x in fambase] # find index of family in fambase
    famind = [i for i in range(len(indxtf)) if indxtf[i] == True]
    if len(famind) == 0:
        raise ValueError('Family should be one of ''gaussian'', ''binomial'', ''poisson'', ''multinomial'', ''cox'', ''mgaussian''')
    elif len(famind) > 1:
        raise ValueError('Family could not be uniquely determined : Use a longer description of the family string.')        
    else:
        family = fambase[famind[0]] 
    
    ## prepare options
    options = glmnetSet(options)

    ## error check options parameters
    if options['alpha'] > 1.0 :
        print('Warning: alpha > 1.0; setting to 1.0')
        options['alpha'] = 1.0
 
    if options['alpha'] < 0.0 :
        print('Warning: alpha < 0.0; setting to 0.0')
        options['alpha'] = 0.0

    ## 
    parm  = options['alpha']
    nlam  = options['nlambda']
    nobs  = x.shape[0]
    nvars = x.shape[1]
    
    # check weights length
    weights = options['weights']
    if len(weights) == 0:
        weights = np.ones([nobs, 1], dtype = np.double)
    elif len(weights) != nobs:
        raise ValueError('Error: Number of elements in ''weights'' not equal to number of rows of ''x''')
    
    # check y length
    nrowy = y.shape[0]
    if nrowy != nobs:
        raise ValueError('Error: Number of elements in ''y'' not equal to number of rows of ''x''')
    
    # check ne   
    ne = options['dfmax']
    if len(ne) == 0:
        ne = nvars + 1

    # check nx
    nx = options['pmax']
    if len(nx) == 0:
        nx = min(ne*2 + 20, nvars)

    # check jd
    exclude = options['exclude']
    print('WARNING!! check Exclude implementation for type !!!')
    if len(exclude) == 0:
        exclude = np.unique(exclude)
        if np.any(exclude > 0 & exclude < nvars):
            raise ValueError('Error: Some excluded variables are out of range')
        jd = np.append(len(exclude), exclude)
    else:
        jd = 0
        
    # check vp    
    vp = options['penalty_factor']
    if len(vp) == 0:
        vp = np.ones([1, nvars])
    
    # inparms
    inparms = glmnetControl()
    
    # cl
    cl = options['cl']
    if any(cl[0,:] > 0):
        raise ValueError('Error: The lower bound on cl must be non-positive')

    if any(cl[1,:] < 0):
        raise ValueError('Error: The lower bound on cl must be non-negative')
        
    cl[0, cl[0, :] == np.double('-inf')] = -1.0*inparms['big']    
    cl[1, cl[1, :] == np.double('inf')]  =  1.0*inparms['big']    
    
    if cl.shape[1] < nvars:
        if cl.shape[1] == 1:
            cl = cl*np.ones([1, nvars])
        else:
            raise ValueError('ERROR: Require length 1 or nvars lower and upper limits')
    else:
        cl = cl[:, 0:nvars-1]
        
        
    exit_rec = 0
    if any(cl.reshape([1,cl.size]) == 0):
        fdev = inparms['fdev']
        if fdev != 0:
            optset = dict()
            optset['fdev'] = 0
            glmnetControl(optset)
            exit_rec = 1
            
    # 
    isd  = np.double(options['standardize'])
    intr = np.double(options['intr'])
    if (intr == True) & (family == 'cox'):
        print('Warning: Cox model has no intercept!')
        
    jsd        = options['standardize_resp']
    thresh     = options['thresh']    
    lambdau    = options['lambdau']
    lambda_min = options['lambda_min']
    
    if len(lambda_min) == 0:
        if nobs < nvars:
            lambda_min = 0.01
        else:
            lambda_min = 1e-4
    
    lempty = (len(lambdau) == 0)
    if lempty:
        if (lambda_min >= 1):
            raise ValueError('ERROR: lambda_min should be less than 1')
        flmin = lambda_min
        ulam  = 0.0
    else:
        flmin = 1.0
        if any(lambdau < 0):
            raise ValueError('ERROR: lambdas should be non-negative')
        
        ulam = -np.sort(-lambdau)    # reverse sort
        nlam = lambdau.size
    #
    maxit = options['maxit']
    gtype = options['gtype']
    if len(gtype) == 0:
        if (nvars < 500):
            gtype = 'covariance'
        else:
            gtype = 'naive'
    
    # ltype
    ltype = options['ltype']
    ltypelist = ['Newton', 'modified.Newton']
    indxtf    = [x.startswith(ltype) for x in ltypelist]
    indl      = [i for i in range(len(indxtf)) if indxtf[i] == True]
    if len(indl) != 1:
        raise ValueError('ERROR: ltype should be one of ''Newton'' or ''modified.Newton''')
    else:
        kopt = indl
    
    
    ## finally call the appropriate fit code
    if family == 'gaussian':
        # call elnet
        pass
    elif family == 'binomial':
        # call lognet
        pass
    elif family == 'multinomial':
        # call lognet
        pass
    elif family == 'cox':
        # call coxnet
        pass
    elif family == 'mgaussian':
        # call mrelnet
        pass
    elif family == 'poisson':
        # call fishnet
        pass
    else:
        raise ValueError('calling a family of fits that has not been implemented yet')
            
    # post process, package and return data
    return -1

#----------------------------------------- 
# end of method glmnet   
#----------------------------------------- 
