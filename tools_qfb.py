# -*- coding: utf-8 -*-
"""
Utilities for the course on Numerical Computing (master QFB)

Created on Sun May  1 10:05:39 2022

@author: alberto.suarez@uam.es
"""


import numpy as np
import matplotlib.pyplot as plt

import numbers
import warnings

from typing import Callable
from matplotlib.axes import Axes
from scipy.integrate import quad


def scale_h_derivative(x_0: float, h:float) -> float:
    """Scale the increment for the numerical computation of derivatives"""
    
    h_scaled = np.asarray(h * x_0)

    indices = (h_scaled == 0.0)

    if isinstance(h, np.ndarray):
        h_scaled[indices] = h[indices]
    else:
        h_scaled[indices] = h
    
    return h_scaled

def numerical_derivative(
    f: Callable[[np.ndarray], np.ndarray],
    x_0: float,
    h: float = 1.0e-6,
    scale_h: bool = True,
) -> float:
    """ Estimate of the derivative by divided (central) differences.

    Args:
        f: Function whose derivative we wish to determine.
        x_0: Point at which the derivative is computed.
        h: Increment.

    Returns:
        A numerical estimate of the derivative.
        
    Examples:
       
        >>> 1.0 - numerical_derivative(np.exp, 0.0)
        2.6755486715046572e-11
        
        >>> numerical_derivative(np.exp, np.array([0.0, 1.0]))
        array([1.        , 2.71828183])

        >>> numerical_derivative(
        ...    np.exp, 
        ...    np.array([0.0, 0.0, 1.0, 1.0]), 
        ...    np.array(2*[1.0e-5, 1.0e-6]),
        ... )
        array([1.        , 1.        , 2.71828183, 2.71828183])
        
        
        >>> numerical_derivative(np.exp, 1.0, np.array([0.1, 1.0e-6, 1.0e-16]))
        array([2.72281456, 2.71828183, 0.        ])
    """
    
    if scale_h:
        h = scale_h_derivative(x_0, h) 
                        
    return (f(x_0 + h) - f(x_0 - h)) / (2.0 * h)

def numerical_second_derivative(
    f: Callable[[np.ndarray], np.ndarray],
    x_0: float,
    h: float = 1.0e-4,
    scale_h: bool = True,
) -> float:
    """ Estimate of the second derivative by divided differences.

    Args:
        f: Function whose derivative we wish to determine.
        x_0: Point at which the derivative is computed.
        h: Increment.

    Returns:
        A numerical estimate of the derivative.
        
    Examples:
       
        >>> 1.0 - numerical_second_derivative(np.exp, 0.0)
        -5.024759275329416e-09
        
        >>> numerical_second_derivative(np.exp, np.array([0.0, 1.0]))
        array([1.00000001, 2.71828187])

        >>> numerical_second_derivative(
        ...    np.exp, 
        ...    np.array([0.0, 0.0, 1.0, 1.0]), 
        ...    np.array(2*[1.0e-4, 1.0e-5]),
        ... )
        array([1.00000001, 0.99999897, 2.71828187, 2.71828782])
        
        
        >>> numerical_second_derivative(
        ...    np.exp, 1.0, np.array([0.1, 1.0e-4, 1.0e-8]))
        array([2.72054782, 2.71828187, 0.        ])
    """
    
    if scale_h:
        h = scale_h_derivative(x_0, h) 
                        
    return (f(x_0 + h) - 2.0 * f(x_0) + f(x_0 - h)) / h**2


def newton_raphson(
    f: Callable,
    df_dx: Callable,
    seed: float,
    tol_abs: float = 1.0e-6,
    max_iters: int = 50
) -> float:
    """ Zero of the function :math:`f` using the Newton-Raphson method.

    Args:
        f: Function whose zero we wish to determine.
        df_dx: Derivative of :math:`f`.
        seed: Initial estimate of the zero (close to an actual one).
        tol_abs: Absolute error estimate.
        max_iters: Maximum number of iteration.

    Returns:
        An approximation of a zero of :math:`f` and its estimated error.

    Examples:
        
     
        >>> f = lambda x: np.cos(x) - 0.1 * x
        >>> df_dx = lambda x: -np.sin(x) - 0.1
        >>> seed = 0.1
        >>> f_zero, error = newton_raphson(f, df_dx, seed, max_iters=20)
        >>> format_string = 'Zero of f = {:.4g} ({:.2g}),   '
        >>> format_string += 'f(f_zero) = {:.2e}.'
        >>> print(format_string.format(f_zero, np.abs(error), f(f_zero)))
        Zero of f = 5.267 (1.1e-09),   f(f_zero) = -2.22e-16.
        
        >>> print('f(f_zero) = {:.2e}.'.format(f(f_zero)))
        f(f_zero) = -2.22e-16.
        
        >>> f = lambda x: np.cos(x) - 0.1 * x
        >>> df_dx = lambda x: -np.sin(x) - 0.1
        >>> x_0 = np.pi
        >>> f_zero, error = newton_raphson(f, df_dx, x_0, max_iters=20)
        >>> format_string = 'Zero of f = {:.4g} ({:.2g}),   '
        >>> format_string += 'f(f_zero) = {:.2e}.'
        >>> print(format_string.format(f_zero, np.abs(error), f(f_zero)))
        Zero of f = -9.679 (2.8e-09),   f(f_zero) = 3.33e-16.
     
        
     
    """
    iter = 0
    delta_x = 2.0 * tol_abs    
    x = seed
    
    while np.abs(delta_x) > tol_abs and iter < max_iters:
        iter += 1
        delta_x = f(x) / df_dx(x)
        x -= delta_x

    if iter == max_iters: 
        warnings.warn('Maximum number of iterations reached.')
     
    return x, delta_x
    

def expected_value(
    f: Callable, 
    pdf: Callable, 
    x_inf: float = None,
    x_sup: float = None,
    cdf: Callable = None,
    cdf_inv: Callable = None,
    tol_abs: float = 1.49e-08  # scipy quad epsabs=1.49e-08
) -> float:
    """ Expected values of a random variable. 

    Args:
        f: Function whose average (expected value) we wish to compute.  
        pdf: Probability density function (pdf) of the random variable. 
        x_inf: lower bound of the support of the distribution.
        x_sup: upper bound of the support of the distribution.
        cdf: Cummulative distribution function (cdf) of the random variable.
        cdf_inv: inverse of the cdf of the random variable.
        tol_abs: target error in the quadrature.
        
    Returns:
        Expected value :math:`\mathbb{E}_X \left[f(X) \right]` 
        
    Examples:
        >>> from scipy.stats import norm 
        >>> mu, sigma = -2.0, 0.5
        >>> pdf = lambda x: norm.pdf(x, mu, sigma)
        >>> cdf_inv = lambda p: norm.ppf(p, mu, sigma)
        >>> mean = expected_value(lambda x: x, pdf, -np.inf, np.inf)
        >>> print(mean)
        -1.9999999999999996
        >>> variance = expected_value(
        ...    lambda x: (x - mean)**2, pdf, -np.inf, np.inf)
        >>> stdev = np.sqrt(variance)
        >>> asymmetry_coeff = expected_value(
        ...     lambda x: (x - mean)**3, pdf, cdf_inv=cdf_inv
        ... ) / stdev**3
        >>> kurtosis = expected_value(
        ...         lambda x: (x - mean)**4, pdf, cdf_inv=cdf_inv
        ...     ) / variance**2
        >>> # Subtract the kurtosis of a normal random variable.
        >>> kurtosis_excess = kurtosis - 3.0 
        >>> string_format = 'mean = {:.2g},  stdev = {:.2g}.' 
        >>> string_format.format(mean, stdev)
        'mean = -2,  stdev = 0.5.'
         
        >>> string_format = 'asymmetry coefficient = {:.2g},  '
        >>> string_format += 'kurtosis (excess) = {:.2g}.'
        >>> print(string_format.format(asymmetry_coeff, kurtosis_excess))
        asymmetry coefficient = -2.5e-15,  kurtosis (excess) = -1.1e-12.
              
    """
    
    if x_inf is None:
        x_inf = cdf_inv(np.finfo(float).eps / 2.0) # numerical minus infinity.
        
    if x_sup is None:
        x_sup = cdf_inv(1.0 - np.finfo(float).eps / 2.0) # numerical infinity.
    
    if cdf is None:
        normalization = quad(lambda x: pdf(x), x_inf, x_sup, epsabs=tol_abs)[0]
    else:
        normalization = cdf(x_sup) - cdf(x_inf)

    return (
        quad(lambda x: f(x) * pdf(x), x_inf, x_sup, epsabs=tol_abs )[0] 
        / normalization
    )


def compare_histogram_pdf(
    X: np.ndarray,
    pdf: Callable,
    n_bins: int = 30,
    ax: Axes = None
) -> Axes:
    """ Graphical comparison of a random sample with the corresponding pdf.

    Args:
        X: Random sample.
        pdf: Probability density function.
        n_bins: Number of bins for the histogram.
        ax: Axis in which the plot is made.

    Examples:
        
        >>> from numpy.random import randn
        >>> from scipy.stats import norm 
        >>> _ = compare_histogram_pdf(randn(1000), norm.pdf, n_bins=50)
        

    """

    if ax is None:
        ax = plt.gca()
    
    ax.hist(X, bins=50, density=True, label='histogram')

    interval = np.min(X), np.max(X)

    x_plot = np.linspace(*interval, num=1000) 
    y_plot = pdf(x_plot)

    ax.plot(
        x_plot, y_plot, 
        color='red', 
        linewidth=3,
        label= '$pdf(x)$'
    )
    ax.set_xlabel('$x$')
    ax.set_ylabel('$pdf(x)$')
    _ = ax.legend()
    
    return ax



def qqplot(x, y, quantiles=None, interpolation='nearest', ax=None, rug=False,
           rug_length=0.05, rug_kwargs=None, **kwargs):
    """Draw a quantile-quantile plot for `x` versus `y`.

    From https://stats.stackexchange.com/questions/403652/two-sample-quantile-quantile-plot-in-python
    Author: https://stats.stackexchange.com/users/97872/artem-mavrin

    Parameters
    ----------
    x, y : array-like
        One-dimensional numeric arrays.

    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If not provided, the current axes will be used.

    quantiles : int or array-like, optional
        Quantiles to include in the plot. This can be an array of quantiles, in
        which case only the specified quantiles of `x` and `y` will be plotted.
        If this is an int `n`, then the quantiles will be `n` evenly spaced
        points between 0 and 1. If this is None, then `min(len(x), len(y))`
        evenly spaced quantiles between 0 and 1 will be computed.

    interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
        Specify the interpolation method used to find quantiles when `quantiles`
        is an int or None. See the documentation for numpy.quantile().

    rug : bool, optional
        If True, draw a rug plot representing both samples on the horizontal and
        vertical axes. If False, no rug plot is drawn.

    rug_length : float in [0, 1], optional
        Specifies the length of the rug plot lines as a fraction of the total
        vertical or horizontal length.

    rug_kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.axvline() and
        matplotlib.axes.Axes.axhline() when drawing rug plots.

    kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.scatter() when drawing
        the q-q plot.
        
        
        
    Examples
    
        # Setup
        >>> rng = np.random.RandomState(0)  # Seed RNG for replicability

        # Example 1: samples of the same length
        >>> n = 100  # Number of samples to draw
        >>> x = rng.normal(size=n)  # Sample 1: X ~ N(0, 1)
        >>> y = rng.standard_t(df=5, size=n)  # Sample 2: Y ~ t(5)

        # Draw quantile-quantile plot
        >>> _ = plt.figure()
        >>> qqplot(x, y, c='b', alpha=0.5, edgecolor='k')
        >>> _ = plt.xlabel('X')
        >>> _ = plt.ylabel('Y')
        >>> _ = plt.title('Two samples of the same length')
        >>> plt.show()

        # Example 2: samples of different lengths
        >>> n_u = 50  # Number of U samples to draw
        >>> n_v = 100  # Number of V samples to draw
        >>> u = rng.normal(size=n_u)  # Sample 1: U ~ N(0, 1)
        >>> v = rng.standard_t(df=5, size=n_v)  # Sample 2: V ~ t(5)

        # Draw quantile-quantile plot
        >>> _ = plt.figure()
        >>> qqplot(u, v, c='b', alpha=0.5, edgecolor='k')
        >>> _ = plt.xlabel('U')
        >>> _ = plt.ylabel('V')
        >>> _ = plt.title('Two samples of different lengths')
        >>> plt.show()

        # Draw quantile-quantile plot with rug plot
        >>> _ = plt.figure()
        >>> qqplot(u, v, c='b', alpha=0.5, edgecolor='k', rug=True)
        >>> _ = plt.xlabel('U')
        >>> _ = plt.ylabel('V')
        >>> _ = plt.title('Two samples of different lengths, with rug plot')
        >>> plt.show()
        
    """
    
    # Get current axes if none are provided
    if ax is None:
        ax = plt.gca()

    if quantiles is None:
        quantiles = min(len(x), len(y))

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
    x_quantiles = np.quantile(x, quantiles, interpolation=interpolation)
    y_quantiles = np.quantile(y, quantiles, interpolation=interpolation)

    # Draw the rug plots if requested
    if rug:
        # Default rug plot settings
        rug_x_params = dict(ymin=0, ymax=rug_length, c='gray', alpha=0.5)
        rug_y_params = dict(xmin=0, xmax=rug_length, c='gray', alpha=0.5)

        # Override default setting by any user-specified settings
        if rug_kwargs is not None:
            rug_x_params.update(rug_kwargs)
            rug_y_params.update(rug_kwargs)

        # Draw the rug plots
        for point in x:
            ax.axvline(point, **rug_x_params)
        for point in y:
            ax.axhline(point, **rug_y_params)

    # Draw the q-q plot
    ax.scatter(x_quantiles, y_quantiles, **kwargs)
    x_plot_min = min(min(x_quantiles), min(y_quantiles))
    x_plot_max = max(max(x_quantiles), max(y_quantiles))
    _ = ax.plot(
        [x_plot_min, x_plot_max], [x_plot_min, x_plot_max]
        , c='r', linestyle="--"
    ) 
# Run examples and test results


if __name__ == "__main__":
    import doctest
    doctest.testmod()
