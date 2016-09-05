# glmnet python 

## Introduction

This is a python version of the glmnet library in R. The underlying fortran codes are the same as the R version, and uses a pathwise coordinate descent algorithm. 

This python version is a front-end that moulds the input and output data to be in a format compatible with the fortran requirements. 

During installation, the fortran code is compiled in the local machine using gfortran, and is called by the python code. 

## Installation

Unzip the package into a suitable location.

Recompile the GLMnet.so shared library (located in ./lib) using:

      gfortran GLMnet.f -fPIC -fdefault-real-8 -shared -o GLMnet.so
 
Currently, the checked-in version of GLMnet.so is compiled using:

Linux version 2.6.32-573.26.1.el6.x86_64 (gcc version 4.4.7 20120313 (Red Hat 4.4.7-16) (GCC) ) for CentOS 6.7 (Final)
 

