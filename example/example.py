# SETUP ---------------------------------------------------------------------------

import pandas as pd
import os
import numpy as np

# 0. Import the data --------------------------------------------------------------


# Read in the data
data = pd.read_csv(os.getcwd() + '/example/theta_names_estimates.csv')
data.head()

# 1. Estimate Prior ---------------------------------------------------------------

from drrank_distribution import prior_estimate
# deltas: set of estimates
# s: set of standard errors
deltas = data.deltas.values
s = data.s.values

# Initialize the estimator object
G = prior_estimate(deltas, s, lambda x: np.power(np.sin(x),2))

# Estimate the prior distribution G.
G.estimate_prior(support_points=5000, spline_order=5)

# Dictionary with the prior distribution
G.prior_g

# Keys:
# mean_delta: mean of the prior
G.prior_g['mean_delta']
# sd_delta: std. of the prior
G.prior_g['sd_delta']
# g_delta: array of the actual prior G
G.prior_g['g_delta']

# Plot the estimated prior distribution
G.plot_estimates(save_path = "example/prior_distribution.jpg")

# 2. Posterior features and Pairwise probabilities --------------------------------

# Compute the posterior features
G.compute_posteriors(alpha=.05, g_delta=None)
G.pmean # posterior means
G.pmean_trans # inverse transformed posterior means
G.lci # lower limit of 1-alpha credible interval
G.uci # upper limit of 1-alpha credible interval
G.lci_trans # lower limit of inverse transformed 1-alpha credible interval
G.uci_trans # upper limit of inverse transformed 1-alpha credible interval

G.posterior_df.head() # Access everything as a Dataframe

# Compute the pairwise ordering probabilities
pis = G.compute_pis(g_delta=None, ncores=-1, power=0)

# 3. DRRank Rankings ------------------------------------------------------------

from drrank import fit
from simul_pij import simul_data

# Simulate data
p_ij = simul_data(size = 25)

# Fit the report card function
results = fit(p_ij, lamb = 0.25, DR = None)

import numpy as np
from drrank import fit_multiple

# Try different values of Lambda
results_l = fit_multiple(p_ij, np.arange(0, 0.9, 0.01))

# Fit the report card function
results_dr = fit(p_ij, lamb = None, DR = 0.05)

# Plot the results

from drrank import fig_ranks

# Merge the results with the identity of our observations
results['firstname'] = data.firstname
fig_ranks(ranking = results, posterior_features = G.posterior_df, ylabels = 'firstname', save_path = 'example/name_ranking.jpg')