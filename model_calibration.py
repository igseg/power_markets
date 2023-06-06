# -*- coding: utf-8 -*-
"""
Created on Thu May 26 19:11:20 2022

@author: alberto
"""

import numpy as np
import warnings 

from numpy.random import default_rng
from scipy.optimize import minimize

from typing import Callable, Tuple, Union, Dict

from scipy.stats import norm, lognorm

from tools_qfb import compare_histogram_pdf


#%%

def minus_log_likelihood(parameters, model, X,  model_type='pdf'):
    """ Minus the log-likelihood of model pdf given the data."""
    
    if model_type == 'pdf':
        return - np.mean(np.log(model(X, *parameters)))
    elif model_type == 'logpdf':
        return - np.mean(model(X, *parameters))
    else:
        warnings.warn('model_type {}: unknown option'.format(model_type))
    
#%%

def fit_pdf_ML(X, model, parameters_seed, model_type='pdf'):
    """ Maximum likelihood estimation of the parameters of a model pdf.
    
    Example:

        >>> mu, sigma = 0.42, 2.78
        >>> n_sample = 100000
        >>> rng = default_rng()
        >>> X = mu + sigma * rng.standard_normal(n_sample)
        >>> parameters, _ = fit_pdf_ML(
        ...     X, norm.pdf , parameters_seed=(0.0, 1.0))
        >>> print(np.round(parameters, 1))
        [0.4 2.8]
    
    """

    info_optimization = minimize(
        lambda pars: minus_log_likelihood(pars, model, X, model_type),
        parameters_seed,  
        method='nelder-mead',
        options={'xatol': 1e-8, 'disp': False}
    )
    
    return info_optimization.x, info_optimization

#%%

# Run examples and test results

if __name__ == "__main__":
    import doctest
    doctest.testmod()

