# -*- coding: utf-8 -*-
"""

@author: bbalasub
"""

def lognet(x, is_sparse, irs, pcs, y, weights, offset, parm, 
          nobs, nvars, jd, vp, cl, ne, nx, nlam, flmin, ulam, 
          thresh, isd, intr, maxit, kopt, family):

    # import packages/methods
    import scipy
    import ctypes
    
    # load shared fortran library
    # this is a bit of a pain. 
    # unless a new python console is started
    # the shared library will persist in memory
    glmlib = ctypes.cdll.LoadLibrary('./GLMnet.so') 
    
    # 
    noo = y.shape[0]
    nc = y.shape[1]
    if not (noo == nobs):
        raise ValueError('x and y have different number of rows in call to glmnet')
    if nc == 1:
        classes, sy = scipy.unique(y, return_inverse = True)
        nc = len(classes)
        indexes = scipy.eye(nc, nc)
        y = indexes[sy, :]
    else:
        classes = scipy.arange(nc) + 1 # 1:nc
    #
    if family == 'binomial':
        if nc > 2:
            raise ValueError('More than two classes in y. use multinomial family instead')
        else:
            nc = 1
            y = y[:, [2, 1]]
    #
    if (len(weights) != 0): 
        t = weights > 0
        if scipy.any(t):
            y = y[t, :]
            x = x[t, :]
            weights = weights[t]
            nobs = scipy.sum(t)
        else:
            t = scipy.empty([1], type = scipy.integer)
        #
        if len(y.shape) == 1:
            mv = len(y)
            ny = 1
        else:    
            mv, ny = y.shape 
            
        y = y*scipy.tile(weights, 1, ny)
    
    #
    if len(offset) == 0:
        offset = y*0
        is_offset = False
    else:
        if len(t) != 0:
            offset = offset[t, :]
        do = offset.shape
        if do[0] != nobs:
            raise ValueError('offset should have the same number of values as observations in binominal/multinomial call to glmnet')
        if nc == 1:
            if do[1] == 1:
                offset = scipy.column_stack((offset, -offset), 1)
            if do[1] > 2:
                raise ValueError('offset should have 1 or 2 columns in binomial call to glmnet')
        if (family == 'multinomial') and (do[1] != nc):
            raise ValueError('offset should have same shape as y in multinomial call to glmnet')
        if_offset = True

    #
    if is_sparse:
        pass
    else:
        pass
    
            
    # pre-process data     
    ybar = scipy.dot(y, weights)
    ybar = ybar/sum(weights)
    nulldev = (y - ybar)**2 * weights
    # ka
    lst = ['covariance', 'naive']
    ka = [i for i in range(len(lst)) if lst[i] == gtype]
    if len(ka) == 0:
        raise ValueError('unrecognized type for ka');
    else:
        ka = ka[0] + 1 # convert from 0-based to 1-based index for fortran
    # offset
    if len(offset) == 0:
        offset = y*0
        is_offset = False
    else:
        is_offset = True

    # remove offset from y
    y = y - offset
    
    # now convert types and allocate memort before calling 
    # glmnet fortran library
    ######################################
    # --------- PROCESS INPUTS -----------
    ######################################
    # force inputs into fortran order and scipy float64
    copyFlag = False
    x = x.astype(dtype = scipy.float64, order = 'F', copy = copyFlag) 
    irs = irs.astype(dtype = scipy.integer, order = 'F', copy = copyFlag)
    pcs = pcs.astype(dtype = scipy.integer, order = 'F', copy = copyFlag)    
    y = y.astype(dtype = scipy.float64, order = 'F', copy = copyFlag)    
    weights = weights.astype(dtype = scipy.float64, order = 'F', copy = copyFlag)    
    jd = jd.astype(dtype = scipy.int32, order = 'F', copy = copyFlag)        
    vp = vp.astype(dtype = scipy.float64, order = 'F', copy = copyFlag)    
    cl = cl.astype(dtype = scipy.float64, order = 'F', copy = copyFlag)    
    ulam   = ulam.astype(dtype = scipy.float64, order = 'F', copy = copyFlag)    

    ######################################
    # --------- ALLOCATE OUTPUTS ---------
    ######################################
    # lmu
    lmu = -1
    lmu_r = ctypes.c_int(lmu)
    # a0
    a0   = scipy.zeros([nlam], dtype = scipy.float64)
    a0   = a0.astype(dtype = scipy.float64, order = 'F', copy = False)    
    a0_r = a0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # ca
    ca   = scipy.zeros([nx, nlam], dtype = scipy.float64)
    ca   = ca.astype(dtype = scipy.float64, order = 'F', copy = False)    
    ca_r = ca.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # ia
    ia   = -1*scipy.ones([nx], dtype = scipy.int32)
    ia   = ia.astype(dtype = scipy.int32, order = 'F', copy = False)    
    ia_r = ia.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    # nin
    nin   = -1*scipy.ones([nlam], dtype = scipy.int32)
    nin   = nin.astype(dtype = scipy.int32, order = 'F', copy = False)    
    nin_r = nin.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    # rsq
    rsq   = -1*scipy.ones([nlam], dtype = scipy.float64)
    rsq   = rsq.astype(dtype = scipy.float64, order = 'F', copy = False)    
    rsq_r = rsq.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # alm
    alm   = -1*scipy.ones([nlam], dtype = scipy.float64)
    alm   = alm.astype(dtype = scipy.float64, order = 'F', copy = False)    
    alm_r = alm.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # nlp
    nlp = -1
    nlp_r = ctypes.c_int(nlp)
    # jerr
    jerr = -1
    jerr_r = ctypes.c_int(jerr)


    #  ###################################
    #   main glmnet fortran caller
    #  ###################################  
    if is_sparse:
        # sparse elnet
              glmlib.spelnet_( 
              ctypes.byref(ctypes.c_int(ka)),
              ctypes.byref(ctypes.c_double(parm)), 
              ctypes.byref(ctypes.c_int(len(weights))), 
              ctypes.byref(ctypes.c_int(nvars)),
              x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
              irs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 
              pcs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 
              y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
              weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
              jd.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 
              vp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
              cl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
              ctypes.byref(ctypes.c_int(ne)), 
              ctypes.byref(ctypes.c_int(nx)), 
              ctypes.byref(ctypes.c_int(nlam)), 
              ctypes.byref(ctypes.c_double(flmin)), 
              ulam.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
              ctypes.byref(ctypes.c_double(thresh)), 
              ctypes.byref(ctypes.c_int(isd)), 
              ctypes.byref(ctypes.c_int(intr)), 
              ctypes.byref(ctypes.c_int(maxit)), 
              ctypes.byref(lmu_r),
              a0_r, 
              ca_r, 
              ia_r, 
              nin_r, 
              rsq_r, 
              alm_r, 
              ctypes.byref(nlp_r), 
              ctypes.byref(jerr_r)
              )
    else:
        # call fortran routines
        glmlib.elnet_( 
              ctypes.byref(ctypes.c_int(ka)),
              ctypes.byref(ctypes.c_double(parm)), 
              ctypes.byref(ctypes.c_int(len(weights))), 
              ctypes.byref(ctypes.c_int(nvars)),
              x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
              y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
              weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
              jd.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 
              vp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
              cl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
              ctypes.byref(ctypes.c_int(ne)), 
              ctypes.byref(ctypes.c_int(nx)), 
              ctypes.byref(ctypes.c_int(nlam)), 
              ctypes.byref(ctypes.c_double(flmin)), 
              ulam.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
              ctypes.byref(ctypes.c_double(thresh)), 
              ctypes.byref(ctypes.c_int(isd)), 
              ctypes.byref(ctypes.c_int(intr)), 
              ctypes.byref(ctypes.c_int(maxit)), 
              ctypes.byref(lmu_r),
              a0_r, 
              ca_r, 
              ia_r, 
              nin_r, 
              rsq_r, 
              alm_r, 
              ctypes.byref(nlp_r), 
              ctypes.byref(jerr_r)
              )
   
    #  ###################################
    #   post process results
    #  ###################################  
     
    # check for error
    if (jerr_r.value > 0):
        raise ValueError("Fatal glmnet error in library call : error code = ", jerr_r.value)
    elif (jerr_r.value < 0):
        print("Warning: Non-fatal error in glmnet library call: error code = ", jerr_r.value)
        print("Check results for accuracy. Partial or no results returned.")
    
    # clip output to correct sizes
    lmu = lmu_r.value
    a0 = a0[0:lmu]
    ca = ca[0:nx, 0:lmu]    
    ia = ia[0:nx]
    nin = nin[0:lmu]
    rsq = rsq[0:lmu]
    alm = alm[0:lmu]
    
    # ninmax
    ninmax = max(nin)
    # fix first value of alm (from inf to correct value)
    if lempty:
        t1 = scipy.log(alm[1])
        t2 = scipy.log(alm[2])
        alm[0] = scipy.exp(2*t1 - t2)        
    # create return fit dictionary
    if ninmax > 0:
        ca = ca[0:ninmax, :]
        df = scipy.sum(scipy.absolute(ca) > 0, axis=0)
        ja = ia[0:ninmax] - 1    # ia is 1-indexed in fortran
        oja = scipy.argsort(ja)
        ja1 = ja[oja]
        beta = scipy.zeros([nvars, lmu], dtype = scipy.float64)
        beta[ja1, :] = ca[oja, :]
    else:
        beta = scipy.zeros([nvars, lmu], dtype = scipy.float64)
        df = scipy.zeros([1, lmu], dtype = scipy.float64)
    
    fit = dict()
    fit['a0'] = a0
    fit['beta'] = beta
    fit['dev'] = rsq
    fit['nulldev'] = nulldev
    fit['df']= df
    fit['lambdau'] = alm
    fit['npasses'] = nlp_r.value
    fit['jerr'] = jerr_r.value
    fit['dim'] = scipy.array([nvars, lmu], dtype = scipy.integer)
    fit['offset'] = is_offset
    fit['class'] = 'elnet'    
 
    #  ###################################
    #   return to caller
    #  ###################################  

    return fit
#----------------------------------------- 
# end of method elmnet
#----------------------------------------- 
    