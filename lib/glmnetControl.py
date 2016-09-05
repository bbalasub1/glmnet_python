# -*- coding: utf-8 -*-
"""

Pre:
    pars <dictionary of parameters, optional>

Post:
    ivals <dictionary of parameters>

@author: bbalasub
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
        ivals = {**ivals, **pars}   # update values
    
    return ivals

# end of glmnetControl()

        
        