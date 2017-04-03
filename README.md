# Glmnet for python 

## Contact

B. J. Balakumar
bbalasub@stanford.edu 

Han Fang
hanfang.cshl@gmail.com

## Install

Using pip (recommended)
    
    pip install glmnet_py

Complied from source

    git clone https://github.com/hanfang/glmnet_python.git
    cd glmnet_python
    python setup.py install

Requirement: Python3, Linux

Currently, the checked-in version of GLMnet.so is compiled for the following config:

 **Linux:** Linux version 2.6.32-573.26.1.el6.x86_64 (gcc version 4.4.7 20120313 (Red Hat 4.4.7-16) (GCC) ) 
 **OS:** CentOS 6.7 (Final) 
 **Hardware:** 8-core Intel(R) Core(TM) i7-2630QM 
 **gfortran:** version 4.4.7 20120313 (Red Hat 4.4.7-17) (GCC)


## Usage
    import glmnet_python
    from glmnet import glmnet

For more examples, see https://github.com/hanfang/glmnet_python/tree/master/test

    
## Introduction

This is a python version of the popular `glmnet` library (beta release). Glmnet fits the entire lasso or elastic-net regularization path for `linear` regression, `logistic` and `multinomial` regression models, `poisson` regression and the `cox` model. 

The underlying fortran codes are the same as the `R` version, and uses a cyclical path-wise coordinate descent algorithm as described in the papers linked below. 

Currently, `glmnet` library methods for gaussian, multi-variate gaussian, binomial, multinomial, poisson and cox models are implemented for both normal and sparse matrices.

Additionally, cross-validation is also implemented for gaussian, multivariate gaussian, binomial, multinomial and poisson models. CV for cox models is yet to be implemented. 

CV can be done in both serial and parallel manner. Parallellization is done using `multiprocessing` and `joblib` libraries.

During installation, the fortran code is compiled in the local machine using `gfortran`, and is called by the python code. 

````diff
+Getting started:
````
*The best starting point to use this library is to start with the Jupyter notebooks in the `test` directory (glmnet_examples.ipynb). Detailed explanations of function calls and parameter values along with plenty of examples are provided there to get you started.*

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

