# SETUP ---------------------------------------------------------------------------

import pandas as pd
import os
import numpy as np

# 0. Import the data --------------------------------------------------------------

# Read in the data
data = pd.read_csv(os.getcwd() + '/example/name_example.csv')
data.head()

# 1. Estimate Prior ---------------------------------------------------------------

from drrank_distribution import prior_estimate
# thetas: set of estimates
# s: set of standard errors
thetas = data.thetas.values
s = data.s.values

# Initialize the estimator object
G = prior_estimate(thetas, s, lambda x: np.power(np.sin(x),2))

# Estimate the prior distribution G.
G.estimate_prior(supp_points=5000, spline_order=5, seed = 123)

# Dictionary with the prior distribution
G.prior_g

# Keys:
# mean_theta: mean of the prior
G.prior_g['mean_theta']
# sd_theta: std. of the prior
G.prior_g['sd_theta']
# g_theta: array of the actual prior G
G.prior_g['g_theta']

# Plot the estimated prior distribution
G.plot_estimates(save_path = "example/prior_distribution.jpg", 
                    binwidth = 0.0030,
                    line_kws = {'alpha': 0.6},
                    fill = True,
                    alpha = 0.3)

# 2. Posterior features and Pairwise probabilities --------------------------------

# Compute the posterior features
G.compute_posteriors(alpha=.05, g_theta=None)
G.pmean # posterior means
G.pmean_trans # inverse transformed posterior means
G.lci # lower limit of 1-alpha credible interval
G.uci # upper limit of 1-alpha credible interval
G.lci_trans # lower limit of inverse transformed 1-alpha credible interval
G.uci_trans # upper limit of inverse transformed 1-alpha credible interval

G.posterior_df.head() # Access everything as a Dataframe

# Compute the pairwise ordering probabilities
pis = G.compute_pis(g_theta=None, ncores=-1, power=0)

# 3. DRRank Rankings ------------------------------------------------------------

from drrank import fit

# Fit the report card function
results = fit(pis, lamb = 0.25, DR = None)

import numpy as np
from drrank import fit_multiple

# Try different values of Lambda
results_l = fit_multiple(pis, np.arange(0, 0.9, 0.01))

# Fit the report card function
results_dr = fit(pis, lamb = None, DR = 0.05)

# Plot the results
from drrank import fig_ranks

# Merge the results with the identity of our observations
results['firstname'] = data.firstname
fig_ranks(ranking = results, posterior_features = G.posterior_df, ylabels = 'firstname', ylabel_fontsize = 8, save_path = 'example/name_ranking.jpg')