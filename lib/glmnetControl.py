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
    ivals["fdev"]    = 1e-5
    ivals["devmax"]  = 0.999
    ivals["eps"]     = 1e-6
    ivals["big"]     = 9.9e35
    ivals["mnlam"]   = 5
    ivals["pmin"]    = 1e-5
    ivals["exmx"]    = 250
    ivals["prec"]    = 1e-10
    ivals["mxit"]    = 100
    
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

        
        