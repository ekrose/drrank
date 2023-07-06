import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def minimgap(c, P, Q, alpha_0, options_fmin, supp_delta, sd_ests, mean_ests, vcv_ests):
    """
    Minmgap estimation of the G prior function
    """
    result = minimize(lambda x: likelihood(x, P, Q, c), alpha_0, method='BFGS', options=options_fmin)
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


def pairprob(di, dj, si, sj, supp_i, supp_j, gprod):
    """
    Compute pairwise probabilities between observation i and j
    """
    xpart = (1 / si) * norm.pdf((di - supp_i) / si)
    ypart = (1 / sj) * norm.pdf((dj - supp_j) / sj)
    ll_prod = xpart[:, np.newaxis] * ypart[np.newaxis, :]

    # All cells where i >= j
    i_gt_j = np.greater.outer(supp_i, supp_j)
    i_le_j = np.less_equal.outer(supp_i, supp_j)

    lkly_ij = np.sum(i_gt_j * ll_prod * gprod)
    lkly_ji = np.sum(i_le_j * ll_prod * gprod)
    prob_ij = lkly_ij / (lkly_ji + lkly_ij)  # probability di > dj

    return prob_ij

def edif(di, dj, si, sj, supp_i, supp_j, gprod, p):
    xpart = (1 / si) * norm.pdf((di - supp_i) / si)
    ypart = (1 / sj) * norm.pdf((dj - supp_j) / sj)
    ll_prod = xpart[:, np.newaxis] * ypart[np.newaxis, :] * gprod
    ll_prod = ll_prod / np.sum(ll_prod)

    # All cells where i >= j
    i_ge_j = np.greater_equal.outer(supp_i, supp_j)

    gaps = np.subtract.outer(supp_i, supp_j) ** p
    exp_dif = np.sum(i_ge_j * ll_prod * gaps)

    return exp_dif


def compute_pairwise_probs(i, F, smin, smax, supp_points, deltas, s, gprod):
    """
    Compute pairwise probabilities between observation i and j
    """
    tmp = np.zeros((F, 3))
    supporti = np.linspace(smin, smax, num=supp_points)
    for j in range(F):
        supportj = np.linspace(smin, smax, num=supp_points)
        tmp[j, 0] = pairprob(deltas[i], deltas[j], s[i], s[j], supporti, supportj, gprod)
        tmp[j, 1] = edif(deltas[i], deltas[j], s[i], s[j], supporti, supportj, gprod, 1)
        tmp[j, 2] = edif(deltas[i], deltas[j], s[i], s[j], supporti, supportj, gprod, 2)
    tmp[i, 0] = 0.5
    return tmp