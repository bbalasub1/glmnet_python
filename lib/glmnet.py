# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 09:18:22 2016

@author: bbalasub
"""
from glmnetSet import glmnetSet

def glmnet(x, y, family = 'gaussian', options = None):
    
    ## set input default options
    # set options to default if nothing is passed in    
    if options is None:
        options = glmnetSet();
    
    ## match the family, abbreviation allowed
    fambase = ['gaussian','binomial','poisson','multinomial','cox','mgaussian'];
    # find index of family in fambase
    indxtf = [x.startswith(family) for x in fambase] # find index of family in fambase
    famind = [i for i in range(len(indxtf)) if indxtf[i] == True]
    if len(famind) == 0:
        raise ValueError('family should be one of ''gaussian'', ''binomial'', ''poisson'', ''multinomial'', ''cox'', ''mgaussian''')
    elif len(famind) > 1:
        raise ValueError('family could not be uniquely determined. use a longer description of the family string.')        
    else:
        family = fambase[famind[0]] 
    
    ## prepare options
    options = glmnetSet(options)

    ## error check options parameters
    if options['alpha'] > 1.0 :
        print('Warning: alpha > 1.0; setting to 1.0')
        options['alpha'] = 1.0
 
    if options['alpha'] < 0.0 :
        print('Warning: alpha < 0.0; setting to 0.0')
        options['alpha'] = 0.0

    ## TBD from here
    parm = options['alpha']
    nlam = options['nlambda']
    
        
    ## finally call the appropriate fit code
    if family == 'gaussian':
        # call elnet
        pass
    elif family == 'binomial':
        # call lognet
        pass
    elif family == 'multinomial':
        # call lognet
        pass
    elif family == 'cox':
        # call coxnet
        pass
    elif family == 'mgaussian':
        # call mrelnet
        pass
    elif family == 'poisson':
        # call fishnet
        pass
    else:
        raise ValueError('calling a family of fits that has not been implemented yet')
            
    # post process, package and return data
    return -1

#----------------------------------------- 
# end of method glmnet   
#----------------------------------------- 
