# -*- coding: utf-8 -*-
"""
 DESCRIPTION:
    View and/or change the factory default parameters in glmnet. Currently, only
    fdev and big are used in the glmnet libraries. 

 USAGE:
   glmnetControl; (with no input or output arguments)
   displays all inner parameters and their current values.
   glmnetControl(pars);
   sets the internal parameters that appear in the fields of pars to the
   new values. 

 ARGUMENTS:
 pars is a structure with the following fields.
 fdev        minimum fractional change in deviance for stopping path;
             factory default = 1.0e-5.
 devmax      maximum fraction of explained deviance for stopping path;
             factory default = 0.999.
 eps         minimum value of lambda.min.ratio (see glmnet); factory
             default= 1.0e-6.
 big         large floating point number; factory default = 9.9e35. Inf in
             definition of upper.limit is set to big.
 mnlam       minimum number of path points (lambda values) allowed;
             factory default = 5.
 pmin        minimum null probability for any class. factory default =
             1.0e-5.
 exmx        maximum allowed exponent. factory default = 250.0.
 prec        convergence threshold for multi response bounds adjustment
             solution. factory default = 1.0e-10.
 mxit  	  maximum iterations for multiresponse bounds adjustment
             solution. factory default = 100.
 factory     If true, reset all the parameters to the factory default;
             default is false.

 DETAILS:
    If called with no arguments, glmnetControl() returns a structure with 
    the current settings of these parameters. Any arguments included in the
    fields of the input structure sets those parameters to the new values, 
    and then silently returns. The values set are persistent for the 
    duration of the Matlab session.

 LICENSE: GPL-2

 AUTHORS:
    Algorithm was designed by Jerome Friedman, Trevor Hastie and Rob Tibshirani
    Fortran code was written by Jerome Friedman
    R wrapper (from which the MATLAB wrapper was adapted) was written by Trevor Hasite
    The original MATLAB wrapper was written by Hui Jiang,
    and is updated and maintained by Junyang Qian.
    This Python wrapper (adapted from the Matlab and R wrappers) 
    is written by Balakumar B.J., bbalasub@stanford.edu 
    Department of Statistics, Stanford University, Stanford, California, USA.

 REFERENCES:
    Friedman, J., Hastie, T. and Tibshirani, R. (2008) Regularization Paths for Generalized Linear Models via Coordinate Descent, 
    http://www.jstatsoft.org/v33/i01/
    Journal of Statistical Software, Vol. 33(1), 1-22 Feb 2010
    
    Simon, N., Friedman, J., Hastie, T., Tibshirani, R. (2011) Regularization Paths for Cox's Proportional Hazards Model via Coordinate Descent,
    http://www.jstatsoft.org/v39/i05/
    Journal of Statistical Software, Vol. 39(5) 1-13

    Tibshirani, Robert., Bien, J., Friedman, J.,Hastie, T.,Simon, N.,Taylor, J. and Tibshirani, Ryan. (2010) Strong Rules for Discarding Predictors in Lasso-type Problems,
    http://www-stat.stanford.edu/~tibs/ftp/strong.pdf
    Stanford Statistics Technical Report

 SEE ALSO:
    glmnet.
"""

def glmnetControl(pars = None):
    import scipy
    
    # default options
    ivals = dict();
    ivals["fdev"]    = scipy.float64(1e-5)
    ivals["devmax"]  = scipy.float64(0.999)
    ivals["eps"]     = scipy.float64(1e-6)
    ivals["big"]     = scipy.float64(9.9e35)
    ivals["mnlam"]   = scipy.float64(5)
    ivals["pmin"]    = scipy.float64(1e-5)
    ivals["exmx"]    = scipy.float64(250)
    ivals["prec"]    = scipy.float64(1e-10)
    ivals["mxit"]    = scipy.float64(100)
    
    # quick return if no user opts
    if pars == None:
        return ivals
    
    # if options are passed in by user, update options with values from opts
    parsInIvals = set(pars.keys()) - set(ivals.keys());
    if len(parsInIvals) > 0:          # assert 'opts' keys are subsets of 'options' keys
        raise ValueError('attempting to set glmnet controls that are not known to glmnetControl')
    else:        
        ivals = merge_dicts(ivals, pars)
    
    return ivals

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

# end of glmnetControl()

        
        
