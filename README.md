# glmnet python 

## Introduction

This is a python version of the glmnet library in R. The underlying fortran codes are the same as the R version, and uses a pathwise coordinate descent algorithm. 

This python version is a front-end that moulds the input and output data to be in a format compatible with the fortran requirements. 

During installation, the fortran code is compiled in the local machine using gfortran, and is called by the python code. 

## Installation

Unzip the package into a suitable location.

Recompile the GLMnet.so shared library (located in ./lib) using:

      ```gfortran GLMnet.f -fPIC -fdefault-real-8 -shared -o GLMnet.so```
 
Currently, the checked-in version of GLMnet.so is compiled for:

Linux version 2.6.32-573.26.1.el6.x86_64 (gcc version 4.4.7 20120313 (Red Hat 4.4.7-16) (GCC) ) for CentOS 6.7 (Final) running on a 8-core Intel(R) Core(TM) i7-2630QM CPU machine. The gfortran version used was 4.4.7 20120313 (Red Hat 4.4.7-17) (GCC).
 
## Authors:

Algorithm was designed by Jerome Friedman, Trevor Hastie and Rob Tibshirani. Fortran code was written by Jerome Friedman. R wrapper (from which the MATLAB wrapper was adapted) was written by Trevor Hastie.

The original MATLAB wrapper was written by Hui Jiang (14 Jul 2009), and was updated and is maintained by Junyang Qian (30 Aug 2013).

This python wrapper (which was adapted from the MATLAB and R wrappers) was written by B. J. Balakumar, bbalasub@stanford.edu (5 Sep 2016).

Department of Statistics, Stanford University, Stanford, California, USA. 

REFERENCES:
* Friedman, J., Hastie, T. and Tibshirani, R. (2008) Regularization Paths for Generalized Linear Models via Coordinate Descent, 
http://www.jstatsoft.org/v33/i01/
*Journal of Statistical Software, Vol. 33(1), 1-22 Feb 2010*
    
* Simon, N., Friedman, J., Hastie, T., Tibshirani, R. (2011) Regularization Paths for Cox's Proportional Hazards Model via Coordinate Descent,
http://www.jstatsoft.org/v39/i05/
*Journal of Statistical Software, Vol. 39(5) 1-13*

* Tibshirani, Robert., Bien, J., Friedman, J.,Hastie, T.,Simon, N.,Taylor, J. and Tibshirani, Ryan. (2010) Strong Rules for Discarding Predictors in Lasso-type Problems,
http://www-stat.stanford.edu/~tibs/ftp/strong.pdf
*Stanford Statistics Technical Report*

