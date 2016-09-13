##########################################
# Sample caller code for elnet
# 

import scipy
import ctypes 
from glmnet import glmnet
from glmnetControl import glmnetControl
from glmnetSet import glmnetSet

glmlib = ctypes.cdll.LoadLibrary('../../lib/GLMnet.so') # this is a bit of a pain. 
                                                # unless a new python console is started
                                                # the shared library will persist in memory
# load data (identical to QuickStartExample.RData)
# glmnet=function(x,y,family=c("gaussian","binomial","poisson","multinomial","cox","mgaussian"),
# weights,offset=NULL,alpha=1.0,nlambda=100,
# lambda.min.ratio=ifelse(nobs<nvars,1e-2,1e-4),lambda=NULL,standardize=TRUE,
# intercept=TRUE,thresh=1e-7,dfmax=nvars+1,pmax=min(dfmax*2+20,nvars),
# exclude,penalty.factor=rep(1,nvars),lower.limits=-Inf,upper.limits=Inf,maxit=100000,
# type.gaussian=ifelse(nvars<500,"covariance","naive"),
# type.logistic=c("Newton","modified.Newton"),standardize.response=FALSE,
# type.multinomial=c("ungrouped","grouped")){

baseDataDir= '/home/bbalasub/Desktop/Summer2016/glmnet/glmnet_R/'
y = scipy.loadtxt(baseDataDir + 'QuickStartExampleY.dat', dtype = scipy.float64)
x = scipy.loadtxt(baseDataDir + 'QuickStartExampleX.dat', dtype = scipy.float64)

# convert x and y to 'F' (fortran) order and scipy float64
y = y.astype(dtype = scipy.float64, order = 'C', copy = True)
x = x.astype(dtype = scipy.float64, order = 'C', copy = True)

# call elnet directly
#      subroutine elnet  (ka,parm,no,ni,x,y,w,jd,vp,cl,ne,nx,nlam,flmin,u    787 
#     *lam,thr,isd,intr,maxit,  lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr)
######################################
# --------- INPUTS -------------------
######################################
# ka
ka_r = ctypes.c_int(1) 
# parm
parm_r = ctypes.c_double(1.0)
# no
no = len(y)
no_r = ctypes.c_int(no)
# ni
ni = x.shape[1]
ni_r = ctypes.c_int(ni)
# x
x_r = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
# y
y_r = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
# w
w = scipy.ones([no], dtype = scipy.float64)
w = w.astype(dtype = scipy.float64, order = 'F', copy = True)    
w_r = w.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
# jd
jd = scipy.ones([1], dtype = scipy.int32)
jd = jd.astype(dtype = scipy.int32, order = 'F', copy = True)    
jd_r = jd.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
# vp
vp = scipy.ones([ni], dtype = scipy.float64)
vp = vp.astype(dtype = scipy.float64, order = 'F', copy = True)    
vp_r = vp.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
# cl
options = glmnetSet()
inparms = glmnetControl()
cl = options['cl']
cl[0, cl[0, :] == scipy.double('-inf')] = -1.0*inparms['big']    
cl[1, cl[1, :] == scipy.double('inf')]  =  1.0*inparms['big']   
if cl.shape[1] < ni:
    if cl.shape[1] == 1:
        cl = cl*scipy.ones([1, ni], dtype = scipy.float64)
    else:
        raise ValueError('ERROR: Require length 1 or nvars lower and upper limits')
else:
    cl = cl[:, 0:ni-1]
cl = cl.astype(dtype = scipy.float64, order = 'F', copy = True)    
cl_r = cl.ctypes.data_as(ctypes.POINTER(ctypes.c_double))    
# ne
ne = ni + 1    
ne_r = ctypes.c_int(ne)
# nx
nx = ni
nx_r = ctypes.c_int(nx)
# nlam
nlam = 100
nlam_r = ctypes.c_int(nlam)
# flmin
flmin = 1.0e-4
flmin_r = ctypes.c_double(flmin)
# ulam
ulam   = scipy.zeros([1], dtype = scipy.float64)
ulam   = ulam.astype(dtype = scipy.float64, order = 'F', copy = True)    
ulam_r = ulam.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
# thr
thr = 1.0e-7
thr_r = ctypes.c_double(thr)
# isd
isd = 1
isd_r = ctypes.c_int(isd)
# intr
intr = 1
intr_r = ctypes.c_int(intr)
# maxit
maxit = 100000
maxit_r = ctypes.c_int(maxit)
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
# elnet
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

print('lmu='); print(lmu_r.value)
print('a0='); print(a0)
print('alm='); print(alm)
