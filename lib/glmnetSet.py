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

Pre:
    opts <optional dict> : dictionary of parameters

Post:
    options <dict>       : dictionary of parameters

@author: bbalasub
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
        raise ValueError('attempting to set glmnet options that are not known to glmnetSet')
    else:        
        options = {**options, **opts}   # update values
    
    return options

