# DRrank

DRrank is a Python library to implement the Empirical Bayes ranking scheme developed in [Kline, Rose, and Walters (2023)](https://arxiv.org/abs/2306.13005). This code was originally developed by [Hadar Avivi](https://avivihadar.github.io/).

## Installation:

The package uses the Gurobi optimizer. To use **DRrank** you must first install Gurobi and acquire a license. More guidance is available from Gurobi [here](https://www.gurobi.com/documentation/9.5/quickstart_windows/cs_python_installation_opt.html). Gurobi offers a variety of free licenses for academic use. For more information, see the following [page](https://www.gurobi.com/academia/academic-program-and-licenses/).


After having successfully set up Gurobipy, install  **DRrank** via pip:

```bash
pip install drrank
```

## Usage

### 1. Estimating the prior

 **DRrank** provides functionality to estimate a prior distribution with a variation on Efron (2016)'s [log-spline deconvolution](https://academic.oup.com/biomet/article-abstract/103/1/1/2390141?redirectedFrom=fulltext) approach, which uses flexible exponential family mixing distribution model with density parameterized by a flexible B-th order spline. 

 To estimate the prior, generate an instance of the `prior_estimate` class with each unit's estimated latent paramater, $\hat{\theta}_i$, and its associated standard errors. You also have the option of supplying an inverse transform, in case $\hat{\theta}_i$ have been transformed to stabilizes variances. When ranking name-specific contact rates in Kline, Rose, and Walters (2023), for example, we apply the transform $\hat{\theta}_i = sin^{-1} \sqrt{p_i}$, where $p_i$ is share of applications with name $i$ that received a call back.


```python
from drrank_distribution import prior_estimate
# deltas: set of estimates
# s: set of standard errors

# Initialize the estimator object
G = prior_estimate(deltas, s, lambda x: np.power(np.sin(x),2))

# Estimate the prior distribution G.
G.estimate_prior(support_points=5000, spline_order=5)
```

Use the `support_points` option (default=5000) to pick the number of points of support over which evaluate the prior density. The minimum and maximum of the support are the min and max of `deltas`. 

Use the `spline_order` option (default=5) to adjust the degrees of freedom of the spline that parameterizes the mixing distribution.

You can then graph the results by calling the following function:

```python
G.plot_estimates()
```

### 2. Estimation of posterior features and $P$ matrix

Once the prior distribution $G$ has been estimated, you can estimate posterior means and credible intervals, as well as the matrix pairwise ordering probabilities $\pi_{ij}$.

```python
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
```

In both functions, it is possible to provide your own prior distribution G by feeding an array as the `g_delta` argument. This density must take support on the values determined by `G.supp_delta`.

`compute_pis()` also provides the option to compute $\pi_{ij}$ as the posterior expectation of $max(\theta_i - \theta_j,0)^power$, providing an extension to weighted ranking exercises. The default, $power=0$, will produce $\pi_{ij}$ that are posterior ordering probabilities discussed in the section and implies that ranking mistakes a considered equally costly regardless of the cardinal difference between $\theta_i$ and $\theta_j$.

### 3. Estimate rankings

To compute rankings, use the **fit** function with a matrix $P$. In the unweighted case, $P$ reflects the posterior probabilities that observation i's latent measure $\theta_i$ exceeds unit j's. That is, each element of this matrix takes the form:

$\pi_{ij} = Pr(\theta_i > \theta_j | Y_i = y_i, Y_j = y_j)$

**DRrank** expects these probabilities to satisfy $\pi_{ij} = 1-\pi_{ji}$.

In the weighted case, $P$ captures expected differences between i and j's value of $\theta$, as discussed above. 

There are two ways to use **DRrank**.

First, one can supply a parameter $\lambda \in [0,1]$, which corresponds to the user's value of correctly ranking pairs of units relative to the costs of misclassifying them. $\lambda=1$ implies correct and incorrect rankings are valued equally, while $\lambda=0$ implies correct rankings are not valued at all. In the unweighted case, it is optimal to assign unit $i$ a higher grade than unit $j$ when $\pi_{ij} > 1/(1+\lambda)$, which implies $\lambda$ also determines the minimum level of posterior certainty required to rank units pairwise.

```python
from drrank import fit
from simul_pij import simul_data

# Simulate data
p_ij = simul_data(size = 25)

# Fit the report card function
results = fit(p_ij, lamb = 0.25, DR = None)
```

The results ojbect contains the row index of $P$, the assigned grades, and the Condorcet rank (i.e., grade under $\lambda=1$). 

|   obs_idx |   grades_lamb0.25 |   condorcet_rank |
|----------:|------------------:|-----------------:|
|         1 |                 1 |                5 |
|         2 |                 1 |                2 |
|         3 |                 1 |               13 |
|         4 |                 1 |                4 |
|         5 |                 1 |                9 |

We also provide functionality to compute results for a list of values of $\lambda$ in parallel:

```python
import numpy as np
from drrank import fit_multiple

# Try different values of Lambda
results_l = fit_multiple(p_ij, np.arange(0, 0.9, 0.01))
```

Second, one can ask **DRrank** to compute grades that maximize Kendall (1938)'s $\tau$, a measure of the rank correlation between units' latent rankings and assigned grades, subject to a constraint on the expected share of pairwise units incorrectly classified, which we refer to as the Discordance Rate (DR).

```python

# Fit the report card function
results = fit(p_ij, lamb = None, DR = 0.05)
```


Finally, we provide a functionality to plot the results of both the posterior features estimates and the rankings:

```python
from drrank import fig_ranks

fig_ranks(results, G.posterior_df)
```