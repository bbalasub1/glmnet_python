# Glmnet for python 

[![PyPI version](https://badge.fury.io/py/glmnet-py.svg)](https://badge.fury.io/py/glmnet-py)
[![GPL Licence](https://badges.frapsoft.com/os/gpl/gpl.svg?v=103)](https://opensource.org/licenses/GPL-2.0/)
[![Documentation Status](https://readthedocs.org/projects/glmnet-python/badge/?version=latest)](http://glmnet-python.readthedocs.io/en/latest/?badge=latest)

## Install

Using pip (recommended)
    
    pip install glmnet_py

Complied from source

    git clone https://github.com/bbalasub1/glmnet_python.git
    cd glmnet_python
    python setup.py install
    (use python setup.py install --user if you get a permission denied message. This does a local install for the user)

Requirement: Python 3, Linux

Currently, the checked-in version of GLMnet.so is compiled for the following config:

 **Linux:** Linux version 2.6.32-573.26.1.el6.x86_64 (gcc version 4.4.7 20120313 (Red Hat 4.4.7-16) (GCC) ) 
 **OS:** CentOS 6.7 (Final) 
 **Hardware:** 8-core Intel(R) Core(TM) i7-2630QM 
 **gfortran:** version 4.4.7 20120313 (Red Hat 4.4.7-17) (GCC)

**For MacOS** installation, here are some solutions that have worked for others: https://github.com/bbalasub1/glmnet_python/issues/13#issuecomment-813000987 **

## Documentation
   Read the Docs: [![Documentation Status](https://readthedocs.org/projects/glmnet-python/badge/?version=latest)](http://glmnet-python.readthedocs.io/en/latest/?badge=latest) or click [me](http://glmnet-python.readthedocs.io/en/latest/glmnet_vignette.html)


## Usage
    import glmnet_python
    from glmnet import glmnet

For more examples, see [iPython notebook](https://github.com/bbalasub1/glmnet_python/blob/master/test/glmnet_examples.ipynb "iPython Notebook")


    
## Introduction

This is a python version of the popular `glmnet` library (beta release). Glmnet fits the entire lasso or elastic-net regularization path for `linear` regression, `logistic` and `multinomial` regression models, `poisson` regression and the `cox` model. 

The underlying fortran codes are the same as the `R` version, and uses a cyclical path-wise coordinate descent algorithm as described in the papers linked below. 

Currently, `glmnet` library methods for gaussian, multi-variate gaussian, binomial, multinomial, poisson and cox models are implemented for both normal and sparse matrices.

Additionally, cross-validation is also implemented for gaussian, multivariate gaussian, binomial, multinomial and poisson models. CV for cox models is yet to be implemented. 

CV can be done in both serial and parallel manner. Parallellization is done using `multiprocessing` and `joblib` libraries.

During installation, the fortran code is compiled in the local machine using `gfortran`, and is called by the python code. 

*The best starting point to use this library is to start with the Jupyter notebooks in the `test` directory ([iPython notebook](https://github.com/bbalasub1/glmnet_python/blob/master/test/glmnet_examples.ipynb "iPython Notebook")). Detailed explanations of function calls and parameter values along with plenty of examples are provided there to get you started.*

## Authors:

Algorithm was designed by Jerome Friedman, Trevor Hastie and Rob Tibshirani. Fortran code was written by Jerome Friedman. R wrapper (from which the MATLAB wrapper was adapted) was written by Trevor Hastie.

The original MATLAB wrapper was written by Hui Jiang (14 Jul 2009), and was updated and is maintained by Junyang Qian (30 Aug 2013).

This python wrapper (which was adapted from the MATLAB and R wrappers) was originally written by B. J. Balakumar (5 Sep 2016). 

List of other contributors along with a summary of their contributions is included in the contributors.dat file.

B. J. Balakumar, bbalasub@gmail.com (Sep 5, 2016). Department of Statistics, Stanford University, Stanford, CA

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

## License:

This software is released under GNU General Public License v3.0 or later. 
