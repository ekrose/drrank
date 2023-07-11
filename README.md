# DRrank

DRrank is a Python library to implement the Empirical Bayes ranking scheme developed in [Kline, Rose, and Walters (2023)](https://arxiv.org/abs/2306.13005). This code was originally developed by [Hadar Avivi](https://avivihadar.github.io/).

## Installation:

The package uses the Gurobi optimizer. To use **DRrank** you must first install Gurobi and acquire a license. More guidance is available from Gurobi [here](https://www.gurobi.com/documentation/9.5/quickstart_windows/cs_python_installation_opt.html). Gurobi offers a variety of free licenses for academic use. For more information, see the following [page](https://www.gurobi.com/academia/academic-program-and-licenses/).


After having successfully set up Gurobipy, install  **DRrank** via pip:

```bash
pip install drrank
```

## Usage

### 0. Load the name example

Within the folder *example*, we provide the file *name_example.csv*, which contains the estimated thetas and their relative standard errors of the Name Ranking problem in Section 4 of the paper. Check also the file *example.py* for all the code displayed in this README.

```python
import pandas as pd

# Read in the data
data = pd.read_csv(os.getcwd() + '/example/theta_names_estimates.csv')
data.head()
```

|   name_id |   deltas |         s | firstname   |
|----------:|---------:|----------:|:------------|
|         1 | 0.53788  | 0.0137214 | Adam        |
|         2 | 0.518394 | 0.0173332 | Aisha       |
|         3 | 0.53767  | 0.0174933 | Allison     |
|         4 | 0.532129 | 0.0138126 | Amanda      |
|         5 | 0.534998 | 0.0136503 | Amy         |


### 1. Estimating the prior

 **DRrank** provides functionality to estimate a prior distribution with a variation on Efron (2016)'s [log-spline deconvolution](https://academic.oup.com/biomet/article-abstract/103/1/1/2390141?redirectedFrom=fulltext) approach, which uses flexible exponential family mixing distribution model with density parameterized by a flexible B-th order natural spline. 

 To estimate the prior, generate an instance of the `prior_estimate` class with each unit's estimated latent paramater, $\hat{\theta}_i$, and its associated standard errors. You also have the option of supplying an inverse transform, in case the $\hat{\theta}_i$ have been transformed to stabilizes variances. When ranking name-specific contact rates in Kline, Rose, and Walters (2023), for example, we apply the transform $\hat{\theta}_i = sin^{-1} \sqrt{\hat{p}_i}$, where $\hat{p}_i$ is share of applications with name $i$ that received a callback.

To estimate the prior, feed `prior_estimate` an array of $\hat{\theta}_i$ and standard errors.

```python
from drrank_distribution import prior_estimate
# deltas: set of estimates
# s: set of standard errors
deltas = data.deltas.values
s = data.s.values

# Initialize the estimator object
G = prior_estimate(deltas, s, transform=lambda x: np.power(np.sin(x),2))

# Estimate the prior distribution G 
G.estimate_prior(support_points=5000, spline_order=5)
```

Use the `support_points` option (default=5000) to pick the number of points of support over which to evaluate the prior density. The minimum and maximum of the support are the minimum and maximum of `deltas`. 

Use the `spline_order` option (default=5) to adjust the degrees of freedom of the spline that parameterizes the mixing distribution.

The estimated prior distribution will be saved as a dictionary, you can access it with the following code:

```python

# Dictionary with the prior distribution
G.prior_g

# Keys:
# mean_delta: mean of the prior
G.prior_g['mean_delta']
# sd_delta: std. of the prior
G.prior_g['sd_delta']
# g_delta: array of the actual prior G
G.prior_g['g_delta']
```


You can then graph the results by calling the following function:

```python

# Plot the estimated prior distribution
G.plot_estimates(save_path = "example/prior_distribution.pdf")
```

![prior_distribution](example/prior_distribution.pdf)

Within the function you can specify the following arguments:
- *g_delta*: provide your own prior distribution G, None implies the function will utilize the estimated G from the *estimate_prior()* method (default = None)
- *show_plot*: whether to show the plot or not (default = True)
- *save_path*: path to where the plot will be saved, None implies the graph will not be saved (default = None) 

### 2. Estimation of posterior features and $P$ matrix

Once the prior distribution $G$ has been estimated, you can estimate posterior means and credible intervals, as well as the matrix of pairwise ordering probabilities $\pi_{ij}$.

```python
# Compute the posterior features
G.compute_posteriors(alpha=.05, g_delta=None)
G.pmean # posterior means
G.pmean_trans # inverse transformed posterior means
G.lci # lower limit of 1-alpha credible interval
G.uci # upper limit of 1-alpha credible interval
G.lci_trans # lower limit of inverse transformed 1-alpha credible interval
G.uci_trans # upper limit of inverse transformed 1-alpha credible interval

G.posterior_df.head() # Dataframe of posterior features
```

|    |    pmean |   pmean_trans |      lci |      uci |   lci_trans |   uci_trans |
|---:|---------:|--------------:|---------:|---------:|------------:|------------:|
|  0 | 0.522254 |      0.248849 | 0.511952 | 0.529469 |    0.239983 |    0.255101 |
|  1 | 0.515227 |      0.242848 | 0.494757 | 0.527798 |    0.225452 |    0.253645 |
|  2 | 0.520531 |      0.247375 | 0.496991 | 0.529019 |    0.227322 |    0.254709 |
|  3 | 0.521341 |      0.248065 | 0.498887 | 0.529003 |    0.228913 |    0.254695 |
|  4 | 0.521891 |      0.248537 | 0.502037 | 0.529244 |    0.231565 |    0.254905 |

Then compute the pairwise ordering probabilities $\pi_{ij}$:

```python
# Compute the pairwise ordering probabilities
pis = G.compute_pis(g_delta=None, ncores=-1, power=0)
```

In both functions, it is possible to provide your own prior distribution G by feeding an array as the `g_delta` argument. This density must take support on the values determined by `G.supp_delta`.

`compute_pis` also provides the option to compute $\pi_{ij}$ as the posterior expectation of $max(\theta_i - \theta_j,0)^power$, providing an extension to weighted ranking exercises. The default, $power=0$, will produce $\pi_{ij}$ that are posterior ordering probabilities discussed in the next section and implies that ranking mistakes a considered equally costly regardless of the cardinal difference between $\theta_i$ and $\theta_j$.

### 3. Estimate rankings

To compute rankings, use the **fit** function with a matrix $P$. In the unweighted case, $P$ reflects the posterior probabilities that observation i's latent measure $\theta_i$ exceeds unit j's. That is, each element of this matrix takes the form:

$\pi_{ij} = Pr(\theta_i > \theta_j | Y_i = y_i, Y_j = y_j)$

**DRrank** expects these probabilities to satisfy $\pi_{ij} = 1-\pi_{ji}$ and will report a warning if that does not appear to be the case.

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
results_dr = fit(p_ij, lamb = None, DR = 0.05)
```

Finally, we provide functionality to plot grades along with posterior means and credible intervals:

```python
from drrank import fig_ranks

# Merge the results with the identity of our observations
results['firstname'] = data.firstname
fig_ranks(ranking = results, posterior_features = G.posterior_df, ylabels = 'firstname', save_path = 'example/name_ranking.pdf')
```

![name_ranking](example/name_ranking.pdf)

Within the function you can specify the following arguments:
- *results*: ranking results from *drrank.fit*
- *posterior_features*: posterior features computed through *G.compute_posteriors()*
- *ylabels*: optional, specify the column in *results* where we have stored the labels of each observation (default = None)
- *show_plot*: whether to show the plot or not (default = True)
- *save_path*: path to where the plot will be saved, None implies the graph will not be saved (default = None) 