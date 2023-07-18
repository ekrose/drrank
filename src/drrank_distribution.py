######################################################################
# Class to estimate the mixing distribution, then estimate posterior #
######################################################################

import numpy as np
import pandas as pd
from scipy.stats import norm
from patsy import dmatrix
from sklearn.preprocessing import scale
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm
from drrank_prior import minimgap, likelihood
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
        thetas: vector of estimates
        s: standard errors of the estimates thetas
        
    """
    def __init__(self, thetas, s, transform=lambda x: x):

        # Transform estimates to lists
        if isinstance(thetas, list):
            self.thetas = np.asarray([thetas]).T
        else:
             self.thetas = thetas
        if isinstance(s, list):
            self.s = np.asarray([s]).T 
        else:
             self.s = s

        # Get the number of units we are ranking
        self.F = len(thetas)

        # Inverse of any transform used to variance stabilize estimates
        self.inv_transform = transform

    def estimate_prior(self, supp_points=5000, spline_order=5, seed=None):
        """
        Estimate the prior distribution G
        Arguments:
            supp_points: support points of theta and z-score distributions (default = 5000)
            spline_order: spline order for our prior distribution estimate, 0.45 for cont, 0.49 for disc efron, 0.0116 for poisson, 0.0031 for balanced poisson (default = 5)
            seed: specify seed number
        """
        print("Estimating prior distribution")
        self.supp_points = supp_points
        self.spline_order = spline_order

        # Calculate support of prior
        supp_theta_min = np.min(self.thetas)
        supp_theta_max = np.max(self.thetas)
        supp_theta = np.linspace(supp_theta_min, supp_theta_max, self.supp_points)
        M = len(supp_theta)
        self.M = M

        # Save the support information
        self.supp_theta_min = supp_theta_min
        self.supp_theta_max = supp_theta_max
        self.supp_theta = supp_theta

        # Calculate P matrix
        s_tilde_big = np.tile(self.s.reshape(-1,1), (1, M))
        thetas_big = np.tile(self.thetas.reshape(-1,1), (1, M))
        supp_theta_big = np.tile(supp_theta.reshape(1, M), (self.F, 1))
        P = (1 / s_tilde_big) * norm.pdf((thetas_big - supp_theta_big) / s_tilde_big)
        self.P = P

        # Report mean and std dev of prior
        mean_ests = np.mean(self.thetas) # Calculate the mean of theta_hat
        sd_ests = np.sqrt(np.var(self.thetas, ddof=1) - np.mean(self.s**2))

        # Save them
        self.mean_ests = mean_ests
        self.sd_ests = sd_ests

        # Estimate the variance-covariance matrix of mean and sd. estimates
        var_mean_ests = np.sum(self.s**2) / self.F**2 
        var_sd_ests = (2 * np.sum(self.s**4) + 4 * np.sum((self.thetas - mean_ests)**2 - self.s**2)) / self.F**2 
        covar_mean_sd_ests = -2 / self.F**2 * np.sum(((self.thetas - mean_ests)**2) * self.s**2) 
        vcv_ests = np.array([[var_mean_ests, covar_mean_sd_ests], [covar_mean_sd_ests, var_sd_ests]]) 
        
        # Save it
        self.vcv_ests = vcv_ests

        # Comput the spline bais
        X = supp_theta.reshape(-1, 1)
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
            print("Cannot use R to create cubic spline basis, switching to python")
            Q = dmatrix("cr(x, df = T)-1", {"x": X, 'T': self.spline_order}, return_type='dataframe')

        if std_spline > 0:
            # Standardize
            Q = scale(Q)
            Q = np.apply_along_axis(lambda w: w / np.sqrt(np.sum(w * w)), 0, Q)

        # Tune G to match mean and SD
        rng = np.random.default_rng(seed=seed)
        alpha_0 = rng.standard_normal((self.spline_order, 1)) 

        # Setup the solver options
        options_fmin = {
            'disp': False,
            'maxiter':10000,
        }

        print("\nPicking penalzation parameter...")

        # Minimize and get the correct c
        result = minimize_scalar(lambda x: minimgap(x, P, Q, alpha_0, options_fmin, 
                                                    supp_theta, sd_ests, mean_ests, 
                                                    vcv_ests),
                                bounds=(0, 0.1),
                                method='bounded',
                                options={'maxiter':500,
                                         'xatol':1e-4})
        c = result.x

        # Check if the minimization was successful
        if result.success != True:
            raise AssertionError("Optimization was unsuccessful")

        print(f"Using df {self.spline_order} and penalty {c:.8f}")
        
        # minimize and solve our likelihood function
        print("\nOptimizing likelihood...")
        result = minimize(lambda x: likelihood(x, P, Q, c), alpha_0,
                        method='CG', jac=True, 
                        options=options_fmin, tol=1e-12)
        alpha_hat = result.x
        print("Likelihood: {:5.4f}".format(result.fun))
        if result.success != True:
            print("Warning: likelihood may not have converged")
            print("Jacobian squared norm: {}".format(np.sum(np.power(result.jac,2))))

        # Get the estimated g_thetas
        logL, dlogL, g_theta = likelihood(alpha_hat, P, Q, c, optimization = False)

        # Report mean and std dev of estimated theta distribution
        mean_theta = np.sum(supp_theta * g_theta) / np.sum(g_theta)
        sd_theta = np.sqrt((np.sum((supp_theta ** 2) * g_theta) / np.sum(g_theta)) - mean_theta ** 2)

        # Print results
        print(f"\nEstimated mean: {mean_ests:4f}")
        print(f"Estimated standard deviation: {sd_ests:4f}")
        print(f"Prior mean: {mean_theta:4f}")
        print(f"Prior standard deviation: {sd_theta:4f}")

        # Rescale g to sum to one
        g_theta = g_theta / np.sum(g_theta)

        # Save the estimates
        self.prior_g = {'mean_theta': mean_theta, 'sd_theta': sd_theta, 'g_theta': g_theta}

    def plot_estimates(self, g_theta=None, show_plot=True, save_path=None):
        """
        Function to plot an histogram of the estimates
        Arguments:
            g_theta: provide your own estimates of the prior distribution
            show_plot: set to True to display the plot
            save_path: specify the path to save the plot, set to None to avoid saving it
        """

        if g_theta == None:
            try:
                g_theta = self.prior_g['g_theta']
            except:
                raise ValueError("Prior distribution was not estimated, cannot proceed to compute the posteriors")
        else:
            print("Using user-supplied prior distribution")
                

        # Get the estimated prior_g
        g_theta = self.prior_g['g_theta']

        # Scale to match histogram
        g_theta = g_theta/max(g_theta)/3
        support = self.supp_theta.copy()
        thetas = self.thetas.copy()

        # Calculate the limits of our plot
        x_min = min(support)*0.99
        x_max = max(support)*1.01
        y_lim = min([round(max(g_theta),1)+0.2, 1])

        # Plot the distribution
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.histplot(data = thetas, 
                    line_kws = {'alpha': 0.6}, kde = False,
                    binwidth = 0.0025,
                    stat = 'probability',
                    fill = True,
                    alpha = 0.3, common_norm = False, ax = ax)
        sns.lineplot(x = support, y = g_theta, color = 'red')
        sns.despine()
        plt.ylabel('Scaled density / mass', fontsize = 25)
        plt.xlabel(r'$\theta$', fontsize = 25)
        plt.ylim(0,y_lim)
        plt.xlim(x_min, x_max)
        plt.yticks(fontsize = 20)
        plt.xticks(fontsize = 20)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', alpha = 0.2)
        ax.text(x_max, y_lim*0.84, "Bias-corrected SD: {:4.4f}\nDecon. implied SD: {:4.4f}\nMean: {:4.4f}\nDecon. implied mean: {:4.4f}".format(
            self.sd_ests, self.prior_g['sd_theta'], self.mean_ests, self.prior_g['mean_theta']), fontsize = 20, horizontalalignment = 'right')

        if save_path != None:
            plt.tight_layout()
            plt.savefig(save_path, format='jpg', dpi=300)

        if show_plot:
            plt.show()

    def compute_posterior_distributions(self, g_theta=None):
        # Decide which prior density to use
        if g_theta == None:
            try:
                g_theta = self.prior_g['g_theta']
            except:
                raise ValueError("Prior distribution was not estimated, cannot proceed to compute the posteriors")
        else:
            print("Using user-supplied prior distribution")

        # Compute posterior distribution for each estimate
        post_dist = ((1 / self.s) * norm.pdf((self.thetas - self.supp_theta[:,np.newaxis]) / self.s)) * g_theta[:,np.newaxis]
        post_dist = post_dist / np.sum(post_dist,0)

        self.post_dist = post_dist

    def posterior_features(self, g_theta, alpha=.05):
        # Estimate posterior if necessary
        if self.post_dist is None:
            self.compute_posterior_distributions(g_theta)

        # Compute posterior features
        self.pmean = self.supp_theta.dot(self.post_dist)
        self.pmean_trans = self.inv_transform(self.supp_theta).dot(self.post_dist)

        self.lci = self.supp_theta[np.argmin(np.cumsum(self.post_dist,0) <= alpha/2,0)]
        self.uci = self.supp_theta[np.argmin(np.cumsum(self.post_dist,0) <= 1-alpha/2,0)]

        self.lci_trans = self.inv_transform(self.lci)
        self.uci_trans = self.inv_transform(self.uci)

        self.posterior_df = pd.DataFrame({
            'pmean': self.pmean,
            'pmean_trans': self.pmean_trans,
            'lci': self.lci,
            'uci': self.uci,
            'lci_trans': self.lci_trans,
            'uci_trans': self.uci_trans
        })
    
    def compute_posteriors(self, alpha=.05, g_theta=None):
        # Get posterior distributions
        self.compute_posterior_distributions(g_theta)

        # Get posterior features
        self.posterior_features(g_theta, alpha)

    def compute_pis(self, g_theta=None, ncores=-1, power=0):
        """
        Estimate pairwise loss compoments. When power=0, these are
        simply Prob(unit_i > unit_j | Y, G). When power > 0, these are
        E[max(unit_i - unit_j,0)^power | Y, G], enabling extensions to loss
        functions that weight ranking loss by powers of the cardinal
        difference between i and j.
        Arguments:
            g_theta: if a different estimation of the prior has been used, supply it through g_theta, default is None, i.e. use the estimates from estimates_prior()
            ncores: how many CPUs for parallel processing, default is all CPUs available (ncores = -1)
            power: cardinal weights. Set to zero (default) for pairwise posterior ordering probabilities
                Set to 1 for absolute error weighted loss. Set to 2 for square-weighted loss.
        
        Access the estimated pairwise components calling self.pis()
        """
        if power < 0:
            raise ValueError("Power option must be weakly positive")

        # Estimate posterior if necessary
        if self.post_dist is None:
            self.compute_posterior_distributions(g_theta)

        # Integrand
        gaps = (np.greater.outer(self.supp_theta, self.supp_theta) 
                * (np.subtract.outer(
                        self.inv_transform(self.supp_theta),
                        self.inv_transform(self.supp_theta)) ** power))

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