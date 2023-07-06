######################################################################
# Class to estimate the G distribution, then estimate the posterior  #
######################################################################

import numpy as np
import pandas as pd
from scipy.stats import norm
from patsy import dmatrix
from sklearn.preprocessing import scale
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.stats import norm
from drrank_prior import minimgap, likelihood, compute_pairwise_probs
import tqdm
from joblib import Parallel, delayed
import multiprocessing
import time

class estimate_distribution():
    """
    Class to estimate both prior and posterior distribution.
    Inputs should be a vector of estimates and standard errors previously calculated.
    Arguments:
        deltas: vector of estimates
        s: standard errors of the estimates deltas
        
    """

    def __init__(self, deltas, s):

        # Initialize our class and save our parameters and initial estimates
        
        # In both cases, transform them into arrays in case they are lists
        if isinstance(deltas, list):
            self.deltas = np.asarray([deltas]).T
        else:
             self.deltas = deltas
        if isinstance(s, list):
            self.s = np.asarray([s]).T 
        else:
             self.s = s

        # Get the number of options we are ranking (i.e. length of our estimates)
        self.F = len(deltas)

    def estimate_prior(self, supp_points = 5000, spline_order = 5, seed = None):
        """
        Estimate the prior distribution G
        Arguments:
            supp_points: support points of delta and z-score distributions (default = 5000)
            spline_order: spline order for our prior distribution estimate, 0.45 for cont, 0.49 for disc efron, 0.0116 for poisson, 0.0031 for balanced poisson (default = 5)
            seed: specify seed number
        """
        print("Estimating prior distribution")
        self.supp_points = supp_points
        self.spline_order = spline_order

        # Set up objects for estimation

        # Create transformation to deconvolve
        delta_hat = self.deltas.copy()
        z = self.deltas.copy()
        s_tilde = self.s.copy()

        # Calculate support of z
        supp_delta_min = np.min(self.deltas)
        supp_delta_max = np.max(self.deltas)
        supp_z = np.linspace(supp_delta_min, supp_delta_max, self.supp_points)
        M = len(supp_z)
        self.M = M

        # Save the support information
        self.supp_delta_min = supp_delta_min
        self.supp_delta_max = supp_delta_max


        # Calculate P matrix
        # Continuous normal density
        s_tilde_big = np.tile(s_tilde, (1, M))
        z_big = np.tile(z, (1, M))
        supp_z_big = np.tile(supp_z.reshape(1, M), (self.F, 1))
        P = (1 / s_tilde_big) * norm.pdf((z_big - supp_z_big) / s_tilde_big)
        self.P = P

        # Estimate Model
        # Report mean and std dev of estimated z distribution
        sd_ests = np.sqrt(np.var(delta_hat, ddof = 1) - np.mean(self.s**2))

        # estimate the variance-covariance matrix of mean and sd. estimates
        mean_ests = np.mean(delta_hat) # Calculate the mean of delta_hat
        var_mean_ests = np.sum(self.s**2) / self.F**2 # Calculate the variance of the mean estimate
        var_sd_ests = (2 * np.sum(self.s**4) + 4 * np.sum((delta_hat - mean_ests)**2 - self.s**2)) / self.F**2 # Calculate the variance of the standard deviation estimate
        covar_mean_sd_ests = -2 / self.F**2 * np.sum(((delta_hat - mean_ests)**2) * self.s**2) # Calculate the covariance between mean and standard deviation estimates
        vcv_ests = np.array([[var_mean_ests, covar_mean_sd_ests], [covar_mean_sd_ests, var_sd_ests]]) # Create a covariance matrix using the mean and standard deviation estimates
        # Save it
        self.vcv_ests = vcv_ests

        # And estimate the splines
        X = supp_z.reshape(-1, 1)
        std_spline = 1

        # We either try to use R functions, otherwise if the user does not have R functions, we proceed with patsy BSpline implementation
        try:
            # import rpy2's package module
            import rpy2.robjects.packages as rpackages
            import rpy2.robjects as robjects
            import rpy2.robjects.numpy2ri
            rpy2.robjects.numpy2ri.activate()
            # import R's utility package
            splines = rpackages.importr('splines')
            # Get the ns function from R
            ns = robjects.r['ns']

            # Convert our X matrix to an R object
            nr,nc = X.shape
            Xr = robjects.r.matrix(X, nrow=nr, ncol=nc)
            Q = ns(Xr, df = self.spline_order)
        except:
            Q = dmatrix("bs(x,df = T, degree = 3)-1", {"x": X, 'T': self.spline_order}, return_type='dataframe')
        
        if std_spline > 0:
            # Standardize
            Q = scale(Q)
            Q = np.apply_along_axis(lambda w: w / np.sqrt(np.sum(w * w)), 1, Q)

        # Tune G to match mean and SD
        supp_delta = supp_z
        rng = np.random.default_rng(seed=seed)
        alpha_0 = rng.standard_normal((self.spline_order, 1)) 

        # Setup the solver options
        options_fmin = {
            'disp': False,
            'maxiter': 500,
            'gtol': 1e-6
        }

        print("Optimizing likelihood...")
        # Minimize and get the correct c
        result = minimize_scalar(lambda x: minimgap(x, P, Q, alpha_0, options_fmin, 
                                                    supp_delta, sd_ests, mean_ests, 
                                                    vcv_ests),
                                bounds=(0, 0.1))
        c = result.x
        fval = result.fun
        exitflag = result.success
        # Check if the minimization was successful
        if exitflag == False:
            raise AssertionError("Optimization was unsuccessful")
        
        # minimize and solve our likelihood function
        result = minimize(lambda x: likelihood(x, P, Q, c), alpha_0, method='BFGS',
                           options=options_fmin, tol = 1e-6)
        alpha_hat = result.x
        # Get the estimated g_deltas
        logL, dlogL, g_delta = likelihood(alpha_hat, P, Q, c, optimization = False)

        print("...done!")
        print(f"Using T {self.spline_order} and penalty {c:.5f}")

        # Report mean and std dev of estimated delta distribution
        mean_delta = np.sum(supp_delta * g_delta) / np.sum(g_delta)
        sd_delta = np.sqrt((np.sum((supp_delta ** 2) * g_delta) / np.sum(g_delta)) - mean_delta ** 2)

        # Print results
        print(f"Prior mean: {mean_delta:4f}")
        print(f"Prior standard deviation: {sd_delta:4f}")


        # Rescale g
        g_delta = g_delta / np.sum(g_delta)

        # Save the estimates
        self.prior_g = {'mean_delta': mean_delta, 'sd_delta': sd_delta, 'g_delta': g_delta}


    def compute_posteriors(self, g_delta = None, ncores = -1):
        """
        Estimate the posterior distribution
        Arguments:
            g_delta: if a different estimation of the prior has been used, supply it through g_delta, default is None, i.e. use the estimates from estimates_prior()
            ncores: how many CPUs for parallel processing, default is all CPUs available (ncores = -1)
        Access the estimated posteriors calling self.posterior_features()
        """

        # First check if we have computed the prior

        if g_delta == None:
            try:
                g_delta = self.prior_g['g_delta']
            except:
                raise ValueError("Prior distribution was not estimated, cannot proceed to compute the posteriors")
        else:
            print("Using user-estimated prior distribution G")

        def eb_posteriors(est, se, supp_delta, g_delta):
            """
            Compute the posterior from an individual estimate
            """
            post_dist = (1 / se) * norm.pdf((est - supp_delta) / se) * g_delta
            post_dist = post_dist / np.sum(post_dist)

            trans_supp = np.sin(supp_delta) ** 2
            pmean = np.sum(post_dist * supp_delta)
            pmean_trans = np.sum(post_dist * trans_supp)
            pmean_sq = np.sum(post_dist * (supp_delta ** 2))
            pmean_sq_p = np.sum(post_dist * (np.sin(supp_delta) ** 4))
            lci = np.max(trans_supp[np.cumsum(post_dist) <= 0.025])
            uci = np.max(trans_supp[np.cumsum(post_dist) <= 0.975])

            argmax = np.argmax(post_dist)
            pmode = supp_delta[argmax]

            return pmean, lci, uci, pmode, pmean_trans, pmean_sq, pmean_sq_p
    
        def eb_wrapper(n):
            return eb_posteriors(self.deltas[n], self.s[n], support, g_delta)

        supp_delta_min = np.min(self.deltas)
        supp_delta_max = np.max(self.deltas)
        support = np.linspace(supp_delta_min, supp_delta_max, num=self.supp_points)

        
        # Parallelize the loop
        # If ncores is -1, use all CPUs
        if ncores == -1:
            ncores = multiprocessing.cpu_count() 

        print(f"Calculating Posteriors, using {ncores} cores...")
        start = time.time()
        # n_jobs is the number of parallel jobs
        posterior_features = Parallel(n_jobs=ncores)(delayed(eb_wrapper)(n) for n in range(self.F))
        end = time.time()
        print('... done (t = {:.4f}s)'.format(end-start))

        # save the posterior features
        self.posterior_features = np.asarray(posterior_features)

    def compute_pis(self, g_delta = None, ncores = -1):
        """
        Estimate the pairwise ordering probabilities
        Arguments:
            g_delta: if a different estimation of the prior has been used, supply it through g_delta, default is None, i.e. use the estimates from estimates_prior()
            ncores: how many CPUs for parallel processing, default is all CPUs available (ncores = -1)
        Access the estimated pairwise ordering probabilities calling self.pis()
        """

        if g_delta == None:
            try:
                g_delta = self.prior_g['g_delta']
            except: 
                raise ValueError("Prior distribution was not estimated, cannot proceed to compute the posteriors")
        else:
            print("Using user-estimated prior distribution G")

        gprod = g_delta@g_delta.T
    
        
        def pairprob_wrapper(j):
            """
            Wrapper to parallelize computation of pairwise probabilities
            """
            tmp = np.zeros((1,3))
            """
            Compute pairwise probabilities between observation i and j
            """
            xpart = (1 /  self.s[i]) * norm.pdf((self.deltas[i] - supporti) /  self.s[i])
            ypart = (1 / self.s[j]) * norm.pdf((self.deltas[j] - supportj) / self.s[j])
            ll_prod = xpart[:, np.newaxis] * ypart[np.newaxis, :]

            # All cells where i >= j
            i_gt_j = np.greater.outer(supporti, supportj)
            i_le_j = np.less_equal.outer(supporti, supportj)

            lkly_ij = np.sum(i_gt_j * ll_prod * gprod)
            lkly_ji = np.sum(i_le_j * ll_prod * gprod)
            prob_ij = lkly_ij / (lkly_ji + lkly_ij)  # probability di > dj
            tmp[0,0] = prob_ij

            """
            Compute exponential difference
            """
            ll_prod = ll_prod * gprod
            ll_prod = ll_prod / np.sum(ll_prod)

            # All cells where i >= j
            i_ge_j = np.greater_equal.outer(supporti, supportj)

            # Do it for both p = 1 and p = 2
            gaps = np.subtract.outer(supporti, supportj) ** 1
            exp_dif = np.sum(i_ge_j * ll_prod * gaps)
            tmp[0, 1] = exp_dif

            gaps = np.subtract.outer(supporti, supportj) ** 2
            exp_dif = np.sum(i_ge_j * ll_prod * gaps)
            tmp[0, 2] = exp_dif

            return tmp

        # Parallelize the loop
        # If ncores is -1, use all CPUs
        if ncores == -1:
            ncores = multiprocessing.cpu_count() 
        
        smin = self.supp_delta_min
        smax = self.supp_delta_max
        supporti = np.linspace(smin, smax, num=self.supp_points)
        supportj = np.linspace(smin, smax, num=self.supp_points)
        
        
        print(f"Calculating pairwise probabilities, using {ncores} cores...")
        store = []
        for i in tqdm.tqdm(range(self.F)):
            """
            Compute pairwise probabilities between observation i and j
            """
            start = time.time()
            # n_jobs is the number of parallel jobs
            res = Parallel(n_jobs=ncores)(delayed(pairprob_wrapper)(j) for j in range(self.F))
            end = time.time()
            print('{:.4f} s'.format(end-start))

            res = np.vstack(res)
            res[i, 0] = 0.5
            store.append(res)
        
        print("...done!")

        self.pis = store.copy()