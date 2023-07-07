######################################################################
# Class to estimate the mixing distribution, then estimate posterior #
######################################################################

import numpy as np
import pandas as pd
from scipy.stats import norm
from patsy import dmatrix
from sklearn.preprocessing import scale
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.stats import norm
from src.drrank_prior import minimgap, likelihood
import tqdm
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns

class prior_estimate():
    """
    Class to estimate both prior and posterior distribution.
    Inputs should be a vector of estimates and standard errors previously calculated.
    Arguments:
        deltas: vector of estimates
        s: standard errors of the estimates deltas
        
    """
    def __init__(self, deltas, s, transform=lambda x: x):

        # Transform estimates to lists
        if isinstance(deltas, list):
            self.deltas = np.asarray([deltas]).T
        else:
             self.deltas = deltas
        if isinstance(s, list):
            self.s = np.asarray([s]).T 
        else:
             self.s = s

        # Get the number of units we are ranking
        self.F = len(deltas)

        # Inverse of any transform used to variance stabilize estimates
        self.inv_transform = transform

    def estimate_prior(self, supp_points=5000, spline_order=5, seed=None):
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

        # Calculate support of prior
        supp_delta_min = np.min(self.deltas)
        supp_delta_max = np.max(self.deltas)
        supp_delta = np.linspace(supp_delta_min, supp_delta_max, self.supp_points)
        M = len(supp_delta)
        self.M = M

        # Save the support information
        self.supp_delta_min = supp_delta_min
        self.supp_delta_max = supp_delta_max
        self.supp_delta = supp_delta

        # Calculate P matrix
        s_tilde_big = np.tile(self.s.reshape(-1,1), (1, M))
        deltas_big = np.tile(self.deltas.reshape(-1,1), (1, M))
        supp_delta_big = np.tile(supp_delta.reshape(1, M), (self.F, 1))
        P = (1 / s_tilde_big) * norm.pdf((deltas_big - supp_delta_big) / s_tilde_big)
        self.P = P

        # Report mean and std dev of prior
        mean_ests = np.mean(self.deltas) # Calculate the mean of delta_hat
        sd_ests = np.sqrt(np.var(self.deltas, ddof=1) - np.mean(self.s**2))

        # Save them
        self.mean_ests = mean_ests
        self.sd_ests = sd_ests

        # Estimate the variance-covariance matrix of mean and sd. estimates
        var_mean_ests = np.sum(self.s**2) / self.F**2 
        var_sd_ests = (2 * np.sum(self.s**4) + 4 * np.sum((self.deltas - mean_ests)**2 - self.s**2)) / self.F**2 
        covar_mean_sd_ests = -2 / self.F**2 * np.sum(((self.deltas - mean_ests)**2) * self.s**2) 
        vcv_ests = np.array([[var_mean_ests, covar_mean_sd_ests], [covar_mean_sd_ests, var_sd_ests]]) 
        
        # Save it
        self.vcv_ests = vcv_ests

        # Comput the spline bais
        X = supp_delta.reshape(-1, 1)
        std_spline = 1

        # Use R if possible for consistency with Efron, otherwise use Patsy BSpline
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
            nr, nc = X.shape
            Xr = robjects.r.matrix(X, nrow=nr, ncol=nc)
            Q = ns(Xr, df = self.spline_order)
        except Exception as e:
            print("Cannot use R to create spline basis, switching to python")
            Q = dmatrix("bs(x, df = T, degree = 3)-1", {"x": X, 'T': self.spline_order}, return_type='dataframe')
        
        if std_spline > 0:
            # Standardize
            Q = scale(Q)
            Q = np.apply_along_axis(lambda w: w / np.sqrt(np.sum(w * w)), 0, Q)

        # Tune G to match mean and SD
        rng = np.random.default_rng(seed=seed)
        alpha_0 = rng.standard_normal((self.spline_order, 1)) 
        # alpha_0 = rng.random((self.spline_order, 1)) 

        # Setup the solver options
        options_fmin = {
            'disp': False,
            'maxiter': 3000,
            'gtol': 1e-8
        }

        print("\nPicking penalzation parameter...")

        # Minimize and get the correct c
        result = minimize_scalar(lambda x: minimgap(x, P, Q, alpha_0, options_fmin, 
                                                    supp_delta, sd_ests, mean_ests, 
                                                    vcv_ests),
                                bounds=(0, 0.3))
        c = result.x

        # Check if the minimization was successful
        if result.success != True:
            raise AssertionError("Optimization was unsuccessful")

        print(f"Using T {self.spline_order} and penalty {c:.5f}")
        
        # minimize and solve our likelihood function
        print("\nOptimizing likelihood...")
        result = minimize(lambda x: likelihood(x, P, Q, c), alpha_0,
                          options=options_fmin, tol = 1e-6)
        alpha_hat = result.x
        if result.success != True:
            print("Warning: likelihood may not have converged")
            print("Jacobian squared norm: {}".format(np.sum(np.power(result.jac,2))))

        # Get the estimated g_deltas
        logL, dlogL, g_delta = likelihood(alpha_hat, P, Q, c, optimization = False)

        # Report mean and std dev of estimated delta distribution
        mean_delta = np.sum(supp_delta * g_delta) / np.sum(g_delta)
        sd_delta = np.sqrt((np.sum((supp_delta ** 2) * g_delta) / np.sum(g_delta)) - mean_delta ** 2)

        # Print results
        print(f"\nEstimated mean: {mean_ests:4f}")
        print(f"Estimated standard deviation: {sd_ests:4f}")
        print(f"Prior mean: {mean_delta:4f}")
        print(f"Prior standard deviation: {sd_delta:4f}")

        # Rescale g to sum to one
        g_delta = g_delta / np.sum(g_delta)

        # Save the estimates
        self.prior_g = {'mean_delta': mean_delta, 'sd_delta': sd_delta, 'g_delta': g_delta}

    def plot_estimates(self, g_delta=None, show_plot = True, save_path = None):
        """
        Function to plot an histogram of the estimates
        Arguments:
            g_delta: provide your own estimates of the prior distribution
            show_plot: set to True to display the plot
            save_path: specify the path to save the plot, set to None to avoid saving it
        """

        if g_delta == None:
            try:
                g_delta = self.prior_g['g_delta']
            except:
                raise ValueError("Prior distribution was not estimated, cannot proceed to compute the posteriors")
        else:
            print("Using user-supplied prior distribution")
                

        # Get the estimated prior_g
        g_delta = self.prior_g['g_delta']
        # Transform it to get the density
        g_delta = g_delta/max(g_delta)/3


        # Calculate the limits of our plot
        x_min = min(self.supp_delta)*0.99
        x_max = max(self.supp_delta)*1.01
        y_lim = min([round(max(g_delta),1)+0.2, 1])

        # Plot the distribution
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.histplot(data = self.deltas, 
                    line_kws = {'alpha': 0.6}, kde = False,
                    binwidth = 0.0025
                    , stat = 'probability', fill = True,
                alpha = 0.3, common_norm = False, ax = ax)
        sns.lineplot(x = self.supp_delta, y = g_delta, color = 'red')
        sns.despine()
        plt.ylabel('Scaled density / mass', fontsize = 25)
        plt.xlabel('Deltas', fontsize = 25)
        plt.ylim(0,y_lim)
        plt.xlim(x_min, x_max)
        plt.yticks(fontsize = 20)
        plt.xticks(fontsize = 20)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', alpha = 0.2)
        ax.text(x_max, y_lim*0.84, "Bias-corrected SD: {:4.4f}\nDecon. implied SD: {:4.4f}\nMean: {:4.4f}\nDecon. implied mean: {:4.4f}".format(
            self.sd_ests, self.prior_g['sd_delta'], self.mean_ests, self.prior_g['mean_delta']), fontsize = 20, horizontalalignment = 'right')

        if save_path != None:
            plt.tight_layout()
            plt.savefig(save_path, format='pdf', dpi=300)

        if show_plot:
            plt.show()

    def compute_posterior_distributions(self, g_delta=None):
        # Decide which prior density to use
        if g_delta == None:
            try:
                g_delta = self.prior_g['g_delta']
            except:
                raise ValueError("Prior distribution was not estimated, cannot proceed to compute the posteriors")
        else:
            print("Using user-supplied prior distribution")

        # Compute posterior distribution for each estimate
        post_dist = ((1 / self.s) * norm.pdf((self.deltas - self.supp_delta[:,np.newaxis]) / self.s)) * g_delta[:,np.newaxis]
        post_dist = post_dist / np.sum(post_dist,0)

        self.post_dist = post_dist

    def posterior_features(self, g_delta, alpha=.05):
        # Estimate posterior if necessary
        if self.post_dist is None:
            self.compute_posterior_distributions(g_delta)

        # Compute posterior features
        self.pmean = self.supp_delta.dot(self.post_dist)
        self.pmean_trans = self.inv_transform(self.supp_delta).dot(self.post_dist)

        self.lci = self.supp_delta[np.argmin(np.cumsum(self.post_dist,0) <= alpha/2,0)]
        self.uci = self.supp_delta[np.argmin(np.cumsum(self.post_dist,0) <= 1-alpha/2,0)]

        self.lci_trans = self.inv_transform(self.lci)
        self.uci_trans = self.inv_transform(self.uci)
    
    def compute_posteriors(self, alpha=.05, g_delta=None):
        # Get posterior distributions
        self.compute_posterior_distributions(g_delta)

        # Get posterior features
        self.posterior_features(g_delta, alpha)

    def compute_pis(self, g_delta=None, ncores=-1, power=0):
        """
        Estimate pairwise loss compoments. When power=0, these are
        simply Prob(unit_i > unit_j | Y, G). When power > 0, these are
        E[max(unit_i - unit_j,0)^power | Y, G], enabling extensions to loss
        functions that weight ranking loss by powers of the cardinal
        difference between i and j.
        Arguments:
            g_delta: if a different estimation of the prior has been used, supply it through g_delta, default is None, i.e. use the estimates from estimates_prior()
            ncores: how many CPUs for parallel processing, default is all CPUs available (ncores = -1)
            power: cardinal weights. Set to zero (default) for pairwise posterior ordering probabilities
                Set to 1 for absolute error weighted loss. Set to 2 for square-weighted loss.
        
        Access the estimated pairwise components calling self.pis()
        """
        # Estimate posterior if necessary
        if self.post_dist is None:
            self.compute_posterior_distributions(g_delta)

        # Integrand
        gaps = (np.greater.outer(self.supp_delta, self.supp_delta) 
                * (np.subtract.outer(
                        self.inv_transform(self.supp_delta),
                        self.inv_transform(self.supp_delta)) ** power))

        # If ncores is -1, use all CPUs
        if ncores == -1:
            ncores = multiprocessing.cpu_count() 
                
        print(f"\nCalculating pi matrix, using {ncores} cores...")
        
        # Pi calculation wrapper
        def pair_dif(post_dist, gaps, i,j):
            post_prod = post_dist[:,i][:,np.newaxis] * post_dist[:,j][np.newaxis,:]
            return np.sum(gaps * post_prod) / np.sum(post_prod)

        pis = np.zeros((self.F,self.F))
        for i in tqdm.tqdm(range(self.F)):
            def par_wrapper(j):
                # if power == 0, then p_ij = 1 - p_ji
                if power == 0 and i > j and 0 == 1:
                    return 1 - pis[j,i]
                else:
                    return pair_dif(self.post_dist, gaps, i, j)
            pis[i,:] = Parallel(n_jobs=ncores)(delayed(par_wrapper)(j) for j in range(self.F))

        print("...done!")

        self.pis = pis.copy()
        return pis