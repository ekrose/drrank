import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def minimgap(c, P, Q, alpha_0, options_fmin, supp_delta, sd_ests, mean_ests, vcv_ests):
    """
    Minmgap estimation of the G prior function
    """
    result = minimize(lambda x: likelihood(x, P, Q, c), alpha_0, options=options_fmin)
    alpha_hat = result.x
    logL, dlogL, g_delta = likelihood(alpha_hat, P, Q, c, optimization=False)

    # Report mean and std dev of estimated delta distribution
    mean_delta = np.sum(supp_delta * g_delta) / np.sum(g_delta)
    sd_delta = np.sqrt((np.sum((supp_delta ** 2) * g_delta) / np.sum(g_delta)) - mean_delta ** 2)
    difs = np.array([mean_delta - mean_ests, sd_ests - sd_delta])
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
        return logL
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