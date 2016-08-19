# -*- coding: utf-8 -*-
"""

Pre:
    pars <dictionary of parameters, optional>

Post:
    ivals <dictionary of parameters>

@author: bbalasub
"""
def glmnetControl(pars = None):
    
    if pars == None:
        ivals = glmnetMex();
        
    if ('factor' in pars) & (pars['factory'] == True):
        ivals['fdev']       = 1e-5
        ivals['devmax']     = 0.999
        ivals['eps']        = 1e-6
        ivals['big']        = 9.9e35
        ivals['mnlam']      = 5
        ivals['pmin']       = 1e-5
        ivals['exmx']       = 250
        ivals['prec']       = 1e-10
        ivals['mxit']       = 100
        task = 0
        glmnetMex(task, ivals)
    else:
        ivals = glmnetMex();
        # if options are passed in by user, update options with values from opts
        optsInOptions = set(pars.keys()) - set(ivals.keys());
        if len(optsInOptions) > 0:          # assert 'opts' keys are subsets of 'options' keys
            raise ValueError('attempting to set glmnetControl parameters that are not known to glmnetControl')
        else:        
            ivals = {**ivals, **pars}   # update values
        print(ivals)
        task = 0
        glmnetMex(task, ivals)
        
    return ivals

# end of glmnetControl()

        
        