# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 21:40:50 2016

@author: bbalasub
"""

def elnet(x, is_sparse, irs, pcs, y, weights, offset, gtype, parm, lempty, 
          nvars, jd, vp, cl, ne, nx, nlam, flmin, ulam, thresh, isd, intr, 
          maxit, family):

    # import packages/methods
    import scipy
    import ctypes
    
    # load shared fortran library
    # this is a bit of a pain. 
    # unless a new python console is started
    # the shared library will persist in memory
    glmlib = ctypes.cdll.LoadLibrary('./GLMnet.so') 
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

    # now convert types and allocate memort before calling 
    # glmnet fortran library
    ######################################
    # --------- INPUTS -------------------
    ######################################
    x = x.astype(dtype = scipy.float64, order = 'F', copy = True)    
    y = y.astype(dtype = scipy.float64, order = 'F', copy = True)    
    weights = weights.astype(dtype = scipy.float64, order = 'F', copy = True)    
    jd = jd.astype(dtype = scipy.int32, order = 'F', copy = True)        
    vp = vp.astype(dtype = scipy.float64, order = 'F', copy = True)    
    cl = cl.astype(dtype = scipy.float64, order = 'F', copy = True)    
    ulam   = ulam.astype(dtype = scipy.float64, order = 'F', copy = True)    

    ######################################
    # --------- OUTPUTS -------------------
    ######################################
    # lmu
    lmu = -1
    lmu_r = ctypes.c_int(lmu)
    # a0
    a0   = scipy.zeros([nlam], dtype = scipy.float64)
    a0   = a0.astype(dtype = scipy.float64, order = 'F', copy = True)    
    a0_r = a0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # ca
    ca   = scipy.zeros([nx, nlam], dtype = scipy.float64)
    ca   = ca.astype(dtype = scipy.float64, order = 'F', copy = True)    
    ca_r = ca.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # ia
    ia   = -1*scipy.ones([nx], dtype = scipy.int32)
    ia   = ia.astype(dtype = scipy.int32, order = 'F', copy = True)    
    ia_r = ia.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    # nin
    nin   = -1*scipy.ones([nlam], dtype = scipy.int32)
    nin   = nin.astype(dtype = scipy.int32, order = 'F', copy = True)    
    nin_r = nin.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    # rsq
    rsq   = -1*scipy.ones([nlam], dtype = scipy.float64)
    rsq   = rsq.astype(dtype = scipy.float64, order = 'F', copy = True)    
    rsq_r = rsq.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # alm
    alm   = -1*scipy.ones([nlam], dtype = scipy.float64)
    alm   = alm.astype(dtype = scipy.float64, order = 'F', copy = True)    
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
        # call glmnetProcessor
        print('is_sparse not implemented')
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
   
    print('lmu='); print(lmu_r.value)
    print('a0='); print(a0)
    print('alm='); print(alm)
    print('len alm = ', len(alm))
           
    # return fit as a dict          
    fit = dict()          
    return fit
#----------------------------------------- 
# end of method elmnet
#----------------------------------------- 
    