import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def minimgap(c, P, Q, alpha_0, options_fmin, supp_theta, sd_ests, mean_ests, vcv_ests):
    """
    Minmgap estimation of the G prior function
    """
    result = minimize(lambda x: likelihood(x, P, Q, c), alpha_0,
            method='CG', jac=True, options=options_fmin, tol=1e-10)
    alpha_hat = result.x
    logL, dlogL, g_theta = likelihood(alpha_hat, P, Q, c, optimization=False)

    # Report mean and std dev of estimated theta distribution
    mean_theta = np.sum(supp_theta * g_theta) / np.sum(g_theta)
    sd_theta = np.sqrt((np.sum((supp_theta ** 2) * g_theta) / np.sum(g_theta)) - mean_theta ** 2)
    difs = np.array([mean_theta - mean_ests, sd_ests - sd_theta])
    gap = difs.T @ np.linalg.inv(vcv_ests) @ difs

    return gap

def likelihood(alpha, P, Q, c, optimization = True):
    """
    Likelihood function to be maximized for the G estimation
    """

    # Calculate prior
    g = np.exp(Q @ alpha - logsumexp(Q @ alpha))

    # Calculate objective
    logL = -np.sum(np.log(P @ g)) + c * np.sqrt(alpha.T @ alpha)

    # Calculate gradient of objective
    T = len(alpha)
    M = len(P[0])
    F = len(P)
    dg = np.tile(g.T, (T, 1)) * (Q.T - np.repeat((Q.T @ g).reshape(-1, 1), repeats = M, axis = 1))
    dlogL = -(np.sum(np.repeat((1 / (P @ g)).reshape(-1, 1), repeats = T, axis = 1) * (P @ dg.T), axis=0) - (c * alpha / np.sqrt(alpha.T @ alpha)))
    if optimization:
        return logL, dlogL
    else:
        return logL, dlogL, g

def logsumexp(x, axis=None):
    # Compute log(sum(exp(x), axis)) while avoiding numerical underflow.
    # By default, axis = 0 (columns).
    # Written by Michael Chen (sth4nth@gmail.com) in MATLAB.
    if axis is None:
        # Determine which dimension sum will use
        axis = np.argmax(np.array(x.shape) != 1)
        if np.size(axis) == 0:
            axis = 0

    # Subtract the largest in each column/axis
    y = np.max(x, axis=axis)
    x = x - np.expand_dims(y, axis=axis)
    s = y + np.log(np.sum(np.exp(x), axis=axis))
    i = ~np.isfinite(y)
    if np.any(i):
        s[i] = y[i]

    return s