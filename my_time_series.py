# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import scipy

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from numpy.random import default_rng
from typing import Callable, Union
from pandas.plotting import autocorrelation_plot
from tools_qfb import compare_histogram_pdf, qqplot
from scipy.stats import norm, probplot, t
from scipy.optimize import least_squares, minimize


#%%

def simulate_AR(
    X_0: np.ndarray,
    phi_0: float,
    phi: np.ndarray,
    sigma: float,
    random_number_generator,
    n_trajectories: int,
    n_times: int,
) -> np.ndarray:
    """ Simulation of an AR process

        SDE: X_t = \phi_0
                   + \sum_{\tau=p} \phi_{\tau} X_{t-\tau}
                   +  sigma*epsilon_t

    Args:
        times: Integration (monitoring) grid (measurement times).
        X_0: Vector of initial values of the process.
        phi_0: Parameter of the AR process.
        phi: Remaining parameters of the AR process.
        sigma: Noise level.
        n_trajectories: Number of simulated trajectories.
        n_times: Number of values simulated.
        seed: Seed of the random number generator (for reproducibility).

    Returns:
        Simulation consisting of n_trajectories trajectories.
        Each trajectory is a row vector of the values of the simulated process.

    Example:

        >>> phi_0, phi = 0.3, [0.1, -0.8]
        >>> p = len(phi)
        >>> X_0 = np.ones(p) * phi_0 / (1.0 -np.sum(phi))
        >>> sigma = 0.1
        >>> random_number_generator = default_rng(seed=0).standard_normal
        >>> X, u_simulated = simulate_AR(X_0, phi_0, phi, sigma,
        ...    random_number_generator, n_trajectories=50, n_times=1000)
        >>> fig, ax = plt.subplots()
        >>> _ = ax.plot(X.T)
        >>> _ = ax.set_xlabel('t')
        >>> _ = ax.set_ylabel('$X_t$')
        >>> _ = ax.set_title('AR({})'.format(p))
        >>> t_stationary = int(10.0 * t_transient(phi))
        >>> fig, ax = plt.subplots()
        >>> _ = plot_acf(X[0, t_stationary:], lags=30, ax=ax)
        >>> u_computed = residuals_AR(X, phi_0, phi)
        >>> tests_gaussian_white_noise(u_computed[0, t_stationary:])
        >>> print('{:6f}'.format(
        ...     np.max(np.abs(u_simulated[:, p:] - u_computed))))
        0.000000


    """
    p = len(phi)
    X = np.empty((n_trajectories, n_times))
    X[:, :p] = X_0

    epsilon = random_number_generator((n_trajectories, n_times))

    u = sigma * epsilon

    for t in range(p, n_times):
        X[:, t] = (
            phi_0
            + X[:, t - np.arange(1, p + 1)] @ phi
            + u[:, t]
        )

    return X, u

# %%
def residuals_AR(X, phi_0, phi):
    """ Residuals of an AR model. """

    p = len(phi)

    vector = (X.ndim == 1)
    if vector:
        X = X[np.newaxis, :] # time series as a two dimensional array

    u = X[:, p:] - phi_0
    for tau in range(1, p + 1):
        u -=  X[:, (p - tau):-tau] * phi[tau - 1]

    if vector:
        return np.ravel(u)  # Return residuals as a one-dimensional array
    else:
        return u


#%%

def simulate_ARMA_GARCH(
    X_0: np.array,
    u_0: np.array,
    h_0: np.ndarray,
    phi_0: float,
    phi: np.ndarray,
    theta: np.ndarray,
    kappa: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    random_number_generator,
    n_trajectories: int,
    n_times: int,
) -> np.ndarray:
    """ Simulation of an ARMA(p, q) + GARCH (r,s) process.

        SDE: X_t = \phi_0
                   + \sum_{\tau=p} \phi_{\tau} X_{t-\tau}
                   + \sum_{\tau=p} \theta_{\tau} epsilon_{t-\tau}
                   u_t

             u_t = \sqrt(h_t) *epsilon_t
             h_t = \phi_0
                   + \sum_{\tau=1^r} \alpha_{\tau} X_{t-\tau}
                       + \sum_{\tau=p} \theta_{\tau} epsilon_{t-\tau}

    Args:
        times: Integration (monitoring) grid (measurement times).
        X_0: Vector of initial values of the process.
        u_0: Vector of initial values of the innovations.
        phi_0: constant term of the ARMA part of the model.
        phi: AR parameters.
        theta: MA parameters.
        kappa: Constant term of the GARCH part of the model.
        alpha: GARCH coefficients of the square of the delayed innovations.
        beta: GARCH coefficients of the delayed volatities term.
        n_trajectories: Number of simulated trajectories.
        n_times: Number of values simulated.
        seed: Seed of the random number generator (for reproducibility).

    Returns:
        Simulation consisting of n_trajectories trajectories.
        Each trajectory is a row vector of the values of the simulated process.

    Example:

        >>> phi_0, phi= 0.3, [0.1, 0.3]
        >>> p = len(phi) - 1
        >>> theta = [0.3, 0.2]
        >>> q = len(theta)
        >>> kappa = 0.1
        >>> alpha = [0.1, 0.15]
        >>> r = len(alpha)
        >>> beta = [0.1, 0.6]
        >>> s = len(beta)
        >>> random_number_generator = default_rng(seed=0).standard_normal
        >>> delay = max(p, q, r, s)
        >>> X_0 = np.ones(delay) * phi[0] / (1.0 -np.sum(phi[1:]))
        >>> u_0 = np.zeros(delay)
        >>> h_0 = np.ones(delay) * kappa / (1.0 + np.sum(alpha) + np.sum(beta))
        >>> X, u, h = simulate_ARMA_GARCH(X_0, u_0, h_0,
        ...    phi_0, phi, theta, kappa, alpha, beta,
        ...    random_number_generator,
        ...    n_trajectories=50, n_times=1000)
        >>> fig, ax = plt.subplots()
        >>> _ = ax.plot(X.T)
        >>> _ = ax.set_xlabel('t')
        >>> _ = ax.set_ylabel('$X_t$')
        >>> _ = ax.set_title('ARMA({},{}) + GARCH({},{})'.format(p, q, r, s))
        >>> t_stationary = 100
        >>> fig, ax = plt.subplots()
        >>> _ = plot_acf(X[0, t_stationary:], lags=30, ax=ax)
        >>> tests_gaussian_white_noise(u[0, t_stationary:])
        ... # u is non-Gaussian white noise with nonlinear dependencies.
        >>> epsilon = u[0,t_stationary:] / np.sqrt(h[0,t_stationary:])
        ... # Gaussian white noise (independent because of Gaussianity).
        >>> tests_gaussian_white_noise(epsilon)

    """
    p = len(phi)
    q = len(theta)
    r = len(alpha)
    s = len(beta)

    delay = max(p, q, r, s)

    X = np.empty((n_trajectories, n_times))
    X[:, :delay] = X_0

    u = np.empty((n_trajectories, n_times))
    u[:, :delay] = u_0

    h = np.empty((n_trajectories, n_times))
    h[:, :delay] = h_0

    epsilon = random_number_generator((n_trajectories, n_times))

    for t in range(delay, n_times):
        h[:, t] = (
            kappa
            + u[:, t - np.arange(1, r + 1)]**2 @ alpha
            + h[:, t - np.arange(1, s + 1)] @ beta
        )

        u[:, t] = np.sqrt(h[:, t]) * epsilon[:, t]
        X[:, t] = (
            phi_0
            + X[:, t - np.arange(1, p + 1)] @ phi
            + u[:, t - np.arange(1, q + 1)] @ theta
            + u[:, t]
        )

    return X, u, h

# %%
def residuals_ARMA_GARCH(X, phi_0, phi, theta, kappa, alpha, beta):
    """Residuals of an AR + GARCH model."""

    p = len(phi)
    q = len(theta)
    r = len(alpha)
    s = len(beta)
    delay = max([p, q, r, s])

    vector = (X.ndim == 1)
    if vector:
        X = X[np.newaxis, :] # time series as a two dimensional array

    n_times = np.shape(X)[1]

    u = np.empty_like(X)
    h = np.empty_like(X)

    # Assume that the initial values of the innovations
    # are errors of the prediction by the unconditional mean.`
    u[:, :delay] = X[:, :delay] - phi_0 / (1.0 - np.sum(phi))

    # Assume that the initial values of the volatility term is the
    # unconditional variance.
    h[:, :delay] = kappa / (1.0 - np.sum(alpha) - np.sum(beta))

    for t in range(delay, n_times):
        u[:, t] = (
            X[:, t] - (
                phi_0
                + X[:, t - np.arange(1, p + 1)] @ phi
                + u[:, t - np.arange(1, q + 1)] @ theta
            )
        )

        h[:, t] = (
            kappa
            + u[:, t - np.arange(1, r + 1)]**2 @ alpha
            + h[:, t - np.arange(1, s + 1)] @ beta
        )

    if vector:  # Return values as one-dimensional arrays.
        return np.ravel(u), np.ravel(h)
    else:
        return u, h


# %%

def fit_AR_LS(X, phi_0_seed, phi_seed):
    """ Least-squares error fit to AR process.

    Example:

        >>> phi_0, phi = 0.3, [0.1, -0.8]
        >>> p = len(phi)
        >>> X_0 = np.ones(p) * phi_0 / (1.0 -np.sum(phi))
        >>> sigma = 0.1
        >>> random_number_generator = default_rng(seed=0).standard_normal
        >>> X, u = simulate_AR(
        ...     X_0, phi_0, phi, sigma,
        ...     random_number_generator, n_trajectories=1, n_times=1000)
        >>> phi_0_LS, phi_LS, _ = fit_AR_LS(
        ...     np.ravel(X), phi_0_seed=0.0, phi_seed=np.zeros(p))
        >>> print(np.round(phi_0_LS, 1), np.round(phi_LS, 1))
        0.3 [ 0.1 -0.8]
    """

    def mean_squared_error(parameters):
        return np.mean(
            residuals_AR(X, phi_0=parameters[0], phi=parameters[1:])**2
        )

    parameters_seed = np.empty(1 + len(phi_seed))
    parameters_seed[0], parameters_seed[1:] = phi_0_seed, phi_seed


    info_optimization = least_squares(mean_squared_error, parameters_seed)

    parameters = info_optimization.x

    return parameters[0], parameters[1:], info_optimization

#%%

def fit_AR_ML_gaussian_noise(X, phi_0_seed, phi_seed, sigma_seed):
    """ Maximum likelihood estimation of an AR model with Gaussian noise."""

    from scipy.optimize import minimize

    def minus_log_likelihood_AR_gaussian_noise(parameters, X):
        """ Minus the log-likelihood of AR model given the data (Gaussian)."""
        u = residuals_AR(X, phi_0=parameters[0], phi=parameters[1:-1])
        return - np.mean(norm.logpdf(u, loc=0.0, scale=parameters[-1]))


    parameters_seed = np.zeros(len(phi_seed) + 2)
    parameters_seed[0] = phi_0_seed
    parameters_seed[1:-1] = phi_seed
    parameters_seed[-1] = sigma_seed

    info_optimization = minimize(
        lambda parameters: minus_log_likelihood_AR_gaussian_noise(
            parameters, X
        ),
        parameters_seed,
        method='nelder-mead',
        options={'xatol': 1e-8, 'disp': False}
    )

    parameters = info_optimization.x

    phi_0 = parameters[0]
    phi = parameters[1:-1]
    sigma = parameters[-1]
    return phi_0, phi, sigma, info_optimization
#%%
def fit_AR_ML_student_t_noise(X, phi_0_seed, phi_seed, sigma_seed, nu_seed):
    """ ML estimation of AR model with Student's t innovations."""
    from scipy.optimize import minimize
    def minus_log_likelihood_AR_student_t_noise(parameters, X):
        """ Minus the log-likelihood of AR(p) model given the data."""
        return None # TO BE IMPLEMENTED

    parameters_seed = np.zeros(len(phi_seed) + 3)
    parameters_seed[0] = phi_0_seed
    parameters_seed[1:-2] = phi_seed
    parameters_seed[-2] = sigma_seed
    parameters_seed[-1] = nu_seed

    info_optimization =  None # TO BE IMPLEMENTED

    parameters = info_optimization.x
    phi_0 = parameters[0]
    phi = parameters[1:-2]
    sigma = parameters[-2]
    nu = parameters[-1]

    return phi_0, phi, sigma, nu, info_optimization

#%%
def fit_AR_GARCH_ML_gaussian_noise(
    X,
    phi_0_seed,
    phi_seed,
    kappa_seed,
    alpha_seed,
    beta_seed,
):
    """ ML estimation of an AR+GARCH model with Gaussian noise."""
    from scipy.optimize import minimize

    def minus_log_likelihood_AR_GARCH_gaussian_noise(parameters, X):
        """ Minus log-likelihood of AR(p) model given the data (Gaussian)."""

        phi_0 = parameters[0]
        phi = parameters[1:p+1]
        kappa = parameters[p+1]
        alpha = parameters[p+2:p+r+2]
        beta = parameters[p+r+2:]

        u, h = residuals_ARMA_GARCH(X, phi_0, phi, [], kappa, alpha, beta)

        return - np.mean(
            norm.logpdf(u, loc=0.0, scale=np.sqrt(h))
        )


    p = len(phi_seed)
    r = len(alpha_seed)
    s = len(beta_seed)
    parameters_seed = np.zeros((1 + p) + (1 + r + s))
    parameters_seed[0] = phi_0_seed
    parameters_seed[1:p+1] = phi_seed
    parameters_seed[p+1] = kappa_seed
    parameters_seed[p+2:p+r+2] = alpha_seed
    parameters_seed[p+r+2:] = beta_seed

    info_optimization = minimize(
        lambda parameters: minus_log_likelihood_AR_GARCH_gaussian_noise(
            parameters, X),
        parameters_seed,
        method='nelder-mead',
        options={'xatol': 1e-8, 'disp': False}
    )
    parameters = info_optimization.x
    phi_0 = parameters[0]
    phi = parameters[1:p+1]
    kappa = parameters[p+1]
    alpha = parameters[p+2:p+r+2]
    beta = parameters[p+r+2:]
    return phi_0, phi, kappa, alpha, beta, info_optimization


#%%
def fit_AR_GARCH_ML_student_t_noise(
    X,
    phi_0_seed,
    phi_seed,
    kappa_seed,
    alpha_seed,
    beta_seed,
    nu_seed,
):
    """ ML estimation of an AR+GARCH model with Student's t noise."""
    from scipy.optimize import minimize

    def minus_log_likelihood_AR_GARCH_t_noise(parameters, X):
        """ Minus log-likelihood of AR(p) model given the data (t-student)."""

        phi_0 = parameters[0]
        phi = parameters[1:p+1]
        kappa = parameters[p+1]
        alpha = parameters[p+2:p+r+2]
        beta = parameters[p+r+2:p+r+s+2]
        df = parameters[-1]

        u, h = residuals_ARMA_GARCH(X, phi_0, phi, [], kappa, alpha, beta)
#         print(parameters)
#         print(np.min(h))
#         print('='*70)
        return - np.mean(
            t.logpdf(u, df = df, loc = 0.0, scale = np.sqrt(h))
        )


    p = len(phi_seed)
    r = len(alpha_seed)
    s = len(beta_seed)
    parameters_seed = np.zeros((1 + p) + (1 + r + s) + 1)
    parameters_seed[0] = phi_0_seed
    parameters_seed[1:p+1] = phi_seed
    parameters_seed[p+1] = kappa_seed
    parameters_seed[p+2:p+r+2] = alpha_seed
    parameters_seed[p+r+2:p+r+s+2] = beta_seed
    parameters_seed[-1] = nu_seed

    info_optimization = minimize(
         lambda parameters: minus_log_likelihood_AR_GARCH_t_noise(
            parameters, X),
        parameters_seed,
        method='nelder-mead',
#         method='Powell',
        options={'xatol': 1e-8, 'disp': False}
    )
    parameters = info_optimization.x
    phi_0 = parameters[0]
    phi = parameters[1:p+1]
    kappa = parameters[p+1]
    alpha = parameters[p+2:p+r+2]
    beta = parameters[p+r+2:p+r+s+2]
    nu = parameters[-1]
    return phi_0, phi, kappa, alpha, beta, nu, info_optimization


#%%
def t_transient(phi):
    """ Compute the length of the transient regime for an AR(p) process."""

    p = len(phi)
    phi_matrix = np.diag(np.ones(p - 1), -1)
    phi_matrix[0, :] = phi
    eigenvalue_max_abs_value = np.max(np.abs(np.linalg.eig(phi_matrix)[0]))

    return - 1.0 / np.log(eigenvalue_max_abs_value)


# %%

def tests_gaussian_white_noise(noise, figsize=(12, 4)):
    """ Tests to determine whether sample is Gaussian white noise.

    """

    fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True)

    mu, sigma = 0.0, np.std(noise)

    compare_histogram_pdf(
        noise,
        lambda x: norm.pdf(x, mu, sigma),
        ax=axs[0],
    )

    probplot(noise, sparams=(mu, sigma), dist='norm', plot=axs[1])

    fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    _ = plot_acf(noise, lags=30, ax=axs[0]) # linear autocorrelations
    _ = plot_acf(np.abs(noise), lags=30, ax=axs[1]) # non-linear dependencies
    _ = axs[0].set_title(
        'Autocorrelations of the time series.')
    _ = axs[0].set_xlabel(r'$\tau$')
    _ = axs[0].set_ylabel(r'$\rho(\tau)$')

    _ = axs[1].set_title(
        'Autocorrelations of the absolute values of the time series.')
    _ = axs[1].set_xlabel(r'$\tau$')
    _ = axs[1].set_ylabel(r'$\rho(\tau)$')

#%%

def test_t_student_noise(noise, nu, figsize=(12,4)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    compare_histogram_pdf(
        noise,
        lambda x: stats.t.pdf(x, nu, loc = 0, scale = 1),
        n_bins=50,
        ax=axs[0],
    )
    rng = default_rng(seed=2)
    reference_sample = (
        stats.t.rvs(nu, size=(len(noise)), random_state=rng)
    )

    qqplot(reference_sample, noise, ax=axs[1])
    _ = axs[1].set_xlabel('t sample quantiles')
    _ = axs[1].set_ylabel('log-returns quantiles')

    fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    _ = plot_acf(noise, lags=30, ax=axs[0]) # linear autocorrelations
    _ = plot_acf(np.abs(noise), lags=30, ax=axs[1]) # non-linear dependencies
    _ = axs[0].set_title(
        'Autocorrelations of the time series.')
    _ = axs[0].set_xlabel(r'$\tau$')
    _ = axs[0].set_ylabel(r'$\rho(\tau)$')

    _ = axs[1].set_title(
        'Autocorrelations of the absolute values of the time series.')
    _ = axs[1].set_xlabel(r'$\tau$')
    _ = axs[1].set_ylabel(r'$\rho(\tau)$')

def arma_garch_t_noise(prices, phi_0, phi, theta, kappa, alpha, beta, nu):

    log_returns = 100 * np.diff(np.log(prices))

    fig, axs = plt.subplots(1, 2, figsize=(16, 4))
    _ = axs[0].plot(prices)
    _ = axs[1].plot(log_returns)

    # Calibrating the AR + GARCH with t-student-distributed residuals.

#     p = len(phi) - 1
    p = len(phi)
    q = len(theta)
    r = len(alpha)
    s = len(beta)

    phi_0_ML, phi_ML, kappa_ML, alpha_ML, beta_ML, nu_ML, info_optimization = fit_AR_GARCH_ML_student_t_noise(
        log_returns,
        phi_0_seed=phi_0,
        phi_seed=phi,
        kappa_seed=kappa,
        alpha_seed=alpha,
        beta_seed=beta,
        nu_seed = nu
    )

    ## Printing results

    print('Maximum likelihood estimates:')
    print(
        np.round(phi_0_ML, 4),
        np.round(phi_ML, 4),
        np.round(kappa_ML, 4),
        np.round(alpha_ML, 4),
        np.round(beta_ML, 4),
        np.round(nu_ML, 4),
    )

    print()
#     print(info_optimization)

    ## Analysing residuals

    u, h = residuals_ARMA_GARCH(
        log_returns,
        phi_0=phi_0_ML,
        phi=phi_ML,
        theta=[],
        kappa=kappa_ML,
        alpha=alpha_ML,
        beta=beta_ML
    )

    test_t_student_noise(u/np.sqrt(h), nu_ML)

    params = [phi_0_ML, phi_ML, kappa_ML, alpha_ML, beta_ML, nu_ML]

    return params, u, h

def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x, ddof=1)/np.var(y, ddof=1) #calculate F test statistic
    dfn = x.size-1 #define degrees of freedom numerator
    dfd = y.size-1 #define degrees of freedom denominator
    p = 1-scipy.stats.f.cdf(f, dfn, dfd) #find p-value of F test statistic
    return f, p

def ls(weights, R):
    
    err = R[0,:]
    for i, weight in enumerate(weights):
        err = err - weight * R[i+1,:]
    return np.sum(err**2)

def synthetic_control_estimator(train_sample, test_sample,
                                loss_function = None,
                                constraints = ({'type': 'eq', 'fun' : lambda x: np.sum(x) - 1}),
                                returns = True,
                                test_mean = True,
                               ):
    """
    Estimating weights for the synthetic control
    
    Args:
    
        train_sample:  observations in the training period, no nans, one row per time-serie and
        one column per time step
        test_sample:   same as train but test
        loss_function: the loss function, the inputs are the weights
        constraints:   are the constraints
        returns:       indicate if the data are returns or not, if True it will plot levels
        test_mean:     True to show mean variation tests
        
    Returns:
        weights: the estimated weights for the replica portfolio
        control: Values of the synthetic control over the whole period (train and test)
    """
    
    if loss_function == None:
        loss_function = lambda weights: ls(weights, train_sample)
    
    parameters_seed = np.zeros(train_sample.shape[0]-1)
    parameters_seed[0] = 1
    
    info_optimization = minimize(
    loss_function, 
    parameters_seed,
    method='SLSQP',
    constraints = constraints
    )
    
    weights = info_optimization.x
    
    sample  = np.hstack((train_sample,test_sample))
    
    control = np.matmul(weights, sample[1:,:])
        
    if test_mean:
        
        idx = train_sample.shape[1]
        
        print('\n Train period:\n ')
        
        print(f' Mean value for the Real data: {train_sample[0,:].mean()}')
        print(f' Mean value for the Control data: {control[:idx].mean()}')
        print(f' p-value of the differences test: {stats.ttest_ind(train_sample[0,:], control[:idx])[1]}')

        print('=' * 30)
        
        print('\n Test period: \n')

        print(f' Mean value for the Real data: {test_sample[0,:].mean()}')
        print(f' Mean value for the Control data: {control[idx:].sum()}')
        print(f' p-value of the differences test: {stats.ttest_ind(test_sample[0,:], control[idx:])[1]}')

        print('=' * 30)
        
        print('\n Temporal differences: \n ')
        
        print(f' p-value of the differences test for the Real data: {stats.ttest_ind(train_sample[0,:], test_sample[0,:])[1]}')
        print(f' p-value of the differences test for the synthetic data: {stats.ttest_ind(control[:idx], control[idx:])[1]}')

    colors = ['#90353b', '#1d4971', 'green']
    
    if returns:
        plt.plot(sample[0,:].cumsum() * 100, label = 'Real data', color = colors[0])
        plt.plot(control.cumsum() * 100, label = 'Synthetic data', color = colors[1])
        plt.axvline(x = idx - 1, color = 'r', label = 'Threshold')
        plt.legend()
        plt.tight_layout()
        plt.ylabel('Comulative returns (%)')
        plt.show()
    
    else:
        plt.plot(sample[0,:].cumsum() * 100, label = 'Real data', color = colors[0])
        plt.plot(control.cumsum() * 100, label = 'Synthetic data', color = colors[1])
        plt.axvline(x = idx - 1, color = 'r', label = 'Threshold')
        plt.legend()
        plt.tight_layout()
        plt.ylabel('Comulative returns (%)')
        plt.show()
    
    return weights, control

# Run examples and test results


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    
