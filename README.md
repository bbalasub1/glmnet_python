# glmnet_python

This is a python version of the glmnet library in R. The underlying fortran codes are the same as the R version, and uses a pathwise coordinate descent algorithm. 

This python version is a front-end that molds the input and output data to be in a format compatible with the fortran requirements.

During installation, the fortran code is compiled in the local machine using gfortran, and is called by the python code. 

It is currently a work in progress as of Aug 27, 2016.
