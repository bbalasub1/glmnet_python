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
    
    # load shared fortran library
    glmlib = ctypes.cdll.LoadLibrary('./GLMnet.so') # this is a bit of a pain. 
                                                # unless a new python console is started
                                                # the shared library will persist in memory
    # pre-process data     
    ybar = scipy.dot(y, weights)
    ybar = ybar/sum(weights)
    nulldev = (y - ybar)**2 * weights
    # ka
    lst = ['covariance', 'naive']
    ka = [i for i in range(len(lst)) if lst[i] == str] + 1
    if len(ka) == 0:
        raise ValueError('unrecognized type for ka');
    # offset
    if len(offset) == 0:
        offset = y*0
        is_offset = False
    else:
        is_offset = True

   # now convert and allocate before calling glmnet fortran library



    #  ###################################
    #   main glmnet fortran caller
    #  ###################################  
    if is_sparse:
        # call glmnetProcessor
        fit = glmnetFortranCaller(task,parm,x,y-offset,jd,vp,ne,nx,nlam,flmin,ulam,
                       thresh,isd,weights,ka,cl,intr,maxit,irs,pcs);
    else:
        # call fortran routines
        glmlib.elnet_( ctypes.byref(ka_r),
              ctypes.byref(parm_r), 
              ctypes.byref(no_r), 
              ctypes.byref(ni_r),
              x_r, 
              y_r, 
              w_r, 
              jd_r, 
              vp_r, 
              cl_r, 
              ctypes.byref(ne_r), 
              ctypes.byref(nx_r), 
              ctypes.byref(nlam_r), 
              ctypes.byref(flmin_r), 
              ulam_r, 
              ctypes.byref(thr_r), 
              ctypes.byref(isd_r), 
              ctypes.byref(intr_r), 
              ctypes.byref(maxit_r), 
              ctypes.byref(lmu_r),
              a0_r, 
              ca_r, 
              ia_r, 
              nin_r, 
              rsq_r, 
              alm_r, 
              ctypes.byref(nlp_r), 
              ctypes.byref(jerr_r))
              
    # return fit as a dict          
    fit = dict()          
    return fit
#----------------------------------------- 
# end of method elmnet
#----------------------------------------- 
    