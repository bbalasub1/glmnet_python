# -*- coding: utf-8 -*-
"""
Internal function called by glmnet. See also glmnet, cvglmnet

"""
# import packages/methods
import scipy
import ctypes
from wtmean import wtmean
from loadGlmLib import loadGlmLib

def mrelnet(x, is_sparse, irs, pcs, y, weights, offset, parm, 
          nobs, nvars, jd, vp, cl, ne, nx, nlam, flmin, ulam, 
          thresh, isd, jsd, intr, maxit, family):

    # load shared fortran library
    glmlib = loadGlmLib() 
    
    # 
    nr = y.shape[1]
    wym = wtmean(y, weights)
    wym = scipy.reshape(wym, (1, wym.size))
    yt2 = (y - scipy.tile(wym, (y.shape[0], 1)))**2
    nulldev = scipy.sum(wtmean(yt2,weights)*scipy.sum(weights))

    if len(offset) == 0:
        offset = y*0
        is_offset = False
    else:
        if offset.shape != y.shape:
            raise ValueError('Offset must match dimension of y')
        is_offset = True
    #
    y = y - offset
    # now convert types and allocate memory before calling 
    # glmnet fortran library
    ######################################
    # --------- PROCESS INPUTS -----------
    ######################################
    # force inputs into fortran order and scipy float64
    copyFlag = False
    x = x.astype(dtype = scipy.float64, order = 'F', copy = copyFlag) 
    irs = irs.astype(dtype = scipy.int32, order = 'F', copy = copyFlag)
    pcs = pcs.astype(dtype = scipy.int32, order = 'F', copy = copyFlag)    
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
    a0   = scipy.zeros([nr, nlam], dtype = scipy.float64)
    a0   = a0.astype(dtype = scipy.float64, order = 'F', copy = False)    
    a0_r = a0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # ca
    ca   = scipy.zeros([nx, nr, nlam], dtype = scipy.float64)
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
        # sparse multnet
        glmlib.multspelnet_( 
              ctypes.byref(ctypes.c_double(parm)), 
              ctypes.byref(ctypes.c_int(nobs)),
              ctypes.byref(ctypes.c_int(nvars)),
              ctypes.byref(ctypes.c_int(nr)),
              x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
              pcs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 
              irs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 
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
              ctypes.byref(ctypes.c_int(jsd)),
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
        # call fortran multnet routine
        glmlib.multelnet_( 
              ctypes.byref(ctypes.c_double(parm)), 
              ctypes.byref(ctypes.c_int(nobs)),
              ctypes.byref(ctypes.c_int(nvars)),
              ctypes.byref(ctypes.c_int(nr)),
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
              ctypes.byref(ctypes.c_int(jsd)),
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
    a0 = a0[0:nr, 0:lmu]
    ca = ca[0:nx, 0:nr, 0:lmu]    
    ia = ia[0:nx]
    nin = nin[0:lmu]
    rsq = rsq[0:lmu]
    alm = alm[0:lmu]
    
    # ninmax
    ninmax = max(nin)
    # fix first value of alm (from inf to correct value)
    if ulam[0] == 0.0:
        t1 = scipy.log(alm[1])
        t2 = scipy.log(alm[2])
        alm[0] = scipy.exp(2*t1 - t2)        
    # create return fit dictionary
    if nr > 1:
        dfmat = a0.copy()
        dd = scipy.array([nvars, lmu], dtype = scipy.integer)
        beta_list = list()
        if ninmax > 0:
            # TODO: is the reshape here done right?
            ca = scipy.reshape(ca, (nx, nr, lmu))
            ca = ca[0:ninmax, :, :]
            ja = ia[0:ninmax] - 1    # ia is 1-indexed in fortran
            oja = scipy.argsort(ja)
            ja1 = ja[oja]
            df = scipy.any(scipy.absolute(ca) > 0, axis=1)
            df = scipy.sum(df, axis = 0)
            df = scipy.reshape(df, (1, df.size))
            for k in range(0, nr):
                ca1 = scipy.reshape(ca[:,k,:], (ninmax, lmu))
                cak = ca1[oja,:]
                dfmat[k, :] = scipy.sum(scipy.absolute(cak) > 0, axis = 0)
                beta = scipy.zeros([nvars, lmu], dtype = scipy.float64)
                beta[ja1, :] = cak
                beta_list.append(beta)
        else:
            for k in range(0, nr):
                dfmat[k, :] = scipy.zeros([1, lmu], dtype = scipy.float64)
                beta_list.append(scipy.zeros([nvars, lmu], dtype = scipy.float64))
            #
            df = scipy.zeros([1, lmu], dtype = scipy.float64)
        #        
        fit = dict()
        fit['beta'] = beta_list
        fit['dfmat']= dfmat
    else:
        dd = scipy.array([nvars, lmu], dtype = scipy.integer)
        if ninmax > 0:
            ca = ca[0:ninmax,:];
            df = scipy.sum(scipy.absolute(ca) > 0, axis = 0);
            ja = ia[0:ninmax] - 1; # ia is 1-indexes in fortran
            oja = scipy.argsort(ja)
            ja1 = ja[oja]
            beta = scipy.zeros([nvars, lmu], dtype = scipy.float64);
            beta[ja1, :] = ca[oja, :];
        else:
            beta = scipy.zeros([nvars,lmu], dtype = scipy.float64);
            df = scipy.zeros([1,lmu], dtype = scipy.float64);
            fit['beta'] = beta
            
    fit['a0'] = a0
    fit['dev'] = rsq
    fit['nulldev'] = nulldev
    fit['df'] = df
    fit['lambdau'] = alm
    fit['npasses'] = nlp_r.value
    fit['jerr'] = jerr_r.value
    fit['dim'] = dd
    fit['offset'] = is_offset
    fit['class'] = 'mrelnet'  
    
    #  ###################################
    #   return to caller
    #  ###################################  

    return fit
#----------------------------------------- 
# end of method mrelnet
#----------------------------------------- 
                
