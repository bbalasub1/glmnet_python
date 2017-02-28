# -*- coding: utf-8 -*-
"""

Sets parameters for glmnet. Returns a default dictionary of parameters
if nothing is passed in. The user is allowed to pass a partial dictionary 
of parameters. Parameter values not in the user input are replaced by 
default values.

Note: The input 'opts' dictionary is expected to contain keys that are a subset
of the keys in the 'options' dictionary below. This check is enforced to make
sure that typos in the keynames do not modify behavior of code (as an example, a typo
that results in the passing of 'alpha' as 'alhpa' should not result in the 
default value of 'alpha' getting passed on silently)

INPUT ARGUMENTS:
-----------------
    opts <optional dict> : dictionary of parameters

OUTPUT ARGUMENTS:
----------------
    options <dict>       : dictionary of parameters

USAGE:
-----
    # return default values as a dict() in options
    options = glmnetSet()
    # set default values for all parameters except 
    #  for alpha, intr, maxit, offset parameters. Set
    #  given values for these parameters.
    options = glmnetSet( alpha = 0.1, \
                         intr = False, \
                         maxit = scipy.int32(1e6), \
                         offset = scipy.empty([0]) )
    # same as previous case, except we pass in a 
    #  dict() object instead
    opts = dict(); opts['alpha'] = 0.5; 
    options = glmnetSet(opts)

..................................................................
Parameter              Default value 
                       Description
..................................................................
alpha                <scipy.float64>
                     The elasticnet mixing parameter, with 0 < alpha <= 1.
                     The penalty is defined as
                           (1-alpha)/2(||beta||_2)^2+alpha||beta||_1.
                     Default is alpha = 1, which is the lasso penalty;
                     Currently alpha = 0 the ridge penalty.

nlambda              <scipy.int32>
                     The number of lambda values - default is 

lambdau              <scipy.ndarray: 1D array of M or 2D array of M x 1>
                     A user supplied lambda sequence. Typical usage is to
                     have the program compute its own lambda sequence
                     based on nlambda and lambda_min. Supplying a value of
                     lambda override this. WARNING: Use with care. Do not 
                     supply a single value for lambda (for predictions 
                     after CV use cvglmnetPredict() instead). Supply a 
                     decreasing sequence of lambda values. glmnet relies
                     on its warm starts for speed, and it's often faster
                     to fit a whole path than compute a single fit.

standardize          <boolean>
                     Logical flag for x variable standardization, prior to
                     fitting the model sequence. The coefficients are
                     always returned on the original scale. Default is
                     standardize = true. If variables are in the same
                     units already, you might not wish to standardize. See
                     details below for y standardization with
                     family='gaussian'.

weights              <scipy.ndarray: 1D array of nobs or 2D array of nobs x 1>
                     Observation weights. Can be total counts if responses
                     are proportion matrices. Default is 1 for each
                     observation.

intr                 <boolean>
                     Should intercept(s) be fitted (default=true) or set
                     to zero (false).

offset               <scipy.ndarray: 1D array of nobs or 2D array of 1 x nobs>
                     A vector of length nobs that is included in the
                     linear predictor (a nobs x nc matrix for the
                     "multinomial" family). Useful for the "poisson"
                     family (e.g. log of exposure time), or for refining a
                     model by starting at a current fit. Default is []. If
                     supplied, then values must also be supplied to the
                     predict function.

lambda_min           <scipy.ndarray: size 1 x 1 and dtype of scipy.float64> 
                     Smallest value for lambda, as a fraction of
                     lambda_max, the (data derived) entry value (i.e., the
                     smallest value for which all coefficients are zero).
                     The default depends on the sample size nobs relative
                     to the number of variables nvars. If nobs > nvars,
                     the default is 0.0001, close to zero. If nobs <
                     nvars, the defaults is 0.01. A very small value of
                     lambda_min will lead to a saturated fit. This is
                     undefined for "binomial" and "multinomial" models,
                     and glmnet will exit gracefully when the percentage
                     deviance explained is almost 1.

thresh               <scipy.float64>
                     Convergence threshold for coordinate descent. Each 
                     inner coordinate-descent loop continues until the 
                     maximum change in the objective after any coefficient 
                     update is less than thresh times the null deviance. 
                     Defaults value is 1E-4.

dfmax                <scipy.ndarray: size 1 x 1>
                     Limit the maximum number of variables in the model. 
                     Useful for very large nvars, if a partial path is
                     desired. Default is nvars + 1.

pmax                 <scipy.ndarray: size 1 x 1>
                     Limit the maximum number of variables ever to be
                     nonzero. Default is min(dfmax * 2 + 20, nvars).

exclude              <scipy.ndarray: 0-based 1D array of indices>
                     Indices of variables to be excluded from the model. 
                     Default is none. Equivalent to an infinite penalty
                     factor (next item).

penalty_factor       <scipy.ndarray: 1D array of size nvars; dtype scipy.float64>
                     Separate penalty factors can be applied to each
                     coefficient. This is a number that multiplies lambda
                     to allow differential shrinkage. Can be 0 for some
                     variables, which implies no shrinkage, and that
                     variable is always included in the model. Default is
                     1 for all variables (and implicitly infinity for
                     variables listed in exclude). Note: the penalty
                     factors are internally rescaled to sum to nvars, and
                     the lambda sequence will reflect this change.

maxit                <scipy.int32>
                     Maximum number of passes over the data for all lambda
                     values; default is 10^5.

cl                   <scipy.ndarray: 2D array of shape 2 x nvars; dtype scipy.float64>
                     Two-row matrix with the first row being the lower 
                     limits for each coefficient and the second the upper
                     limits. Can be presented as a single column (which
                     will then be replicated), else a matrix of nvars
                     columns. Default [-Inf;Inf].

gtype                <str>
                     Two algorithm types are supported for (only)
                     family = 'gaussian'. The default when nvar<500 is
                     options.gtype = 'covariance', and saves all
                     inner-products ever computed. This can be much faster
                     than options.gtype='naive', which loops through nobs
                     every time an inner-product is computed. The latter
                     can be far more efficient for nvar >> nobs
                     situations, or when nvar > 500.

ltype                <str>
                     If 'Newton' then the exact hessian is used (default),
                     while 'modified.Newton' uses an upper-bound on the
                     hessian, and can be faster.

standardize_resp     <boolean>
                     This is for the family='mgaussian' family, and allows
                     the user to standardize the response variables.

mtype                <str>
                     If 'grouped' then a grouped lasso penalty is used on
                     the multinomial coefficients for a variable. This
                     ensures they are all in our out together. The default
                     is 'ungrouped'.

LICENSE: 
-------
    GPL-2

AUTHORS:
-------
    Algorithm was designed by Jerome Friedman, Trevor Hastie and Rob Tibshirani
    Fortran code was written by Jerome Friedman
    R wrapper (from which the MATLAB wrapper was adapted) was written by Trevor Hasite
    The original MATLAB wrapper was written by Hui Jiang,
    and is updated and maintained by Junyang Qian.
    This Python wrapper (adapted from the Matlab and R wrappers) 
    is written by Balakumar B.J., bbalasub@stanford.edu 
    Department of Statistics, Stanford University, Stanford, California, USA.

"""

def glmnetSet(opts = None):
    import scipy
    
    # default options
    options = {
        "weights"             : scipy.empty([0]),
        "offset"              : scipy.empty([0]),
        "alpha"               : scipy.float64(1.0),
        "nlambda"             : scipy.int32(100),
        "lambda_min"          : scipy.empty([0]),
        "lambdau"             : scipy.empty([0]),
        "standardize"         : True,
        "intr"                : True,
        "thresh"              : scipy.float64(1e-7),
        "dfmax"               : scipy.empty([0]),
        "pmax"                : scipy.empty([0]),
        "exclude"             : scipy.empty([0], dtype = scipy.integer),
        "penalty_factor"      : scipy.empty([0]),
        "cl"                  : scipy.array([[scipy.float64(-scipy.inf)], [scipy.float64(scipy.inf)]]), 
        "maxit"               : scipy.int32(1e5),
        "gtype"               : [],
        "ltype"               : 'Newton',
        "standardize_resp"    : False,
        "mtype"               : 'ungrouped'
   }
    
    # quick return if no user opts
    if opts == None:
        print('pdco default options:')
        print(options)
        return options
    
    # if options are passed in by user, update options with values from opts
    optsInOptions = set(opts.keys()) - set(options.keys());
    if len(optsInOptions) > 0:          # assert 'opts' keys are subsets of 'options' keys
        print(optsInOptions, ' : unknown option for glmnetSet')
        raise ValueError('attempting to set glmnet options that are not known to glmnetSet')
    else:        
        options = merge_dicts(options, opts)
    
    return options

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result
