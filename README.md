# DRrank

DRrank is a Python library to implement the Empirical Bayes ranking scheme developed in [Kline, Rose, and Walters (2023)](https://arxiv.org/abs/2306.13005).

## Installation:

The package uses the Gurobi optimizer. To use **DRrank** you must first install Gurobi and acquire a license. More guidance is available from Gurobi [here](https://www.gurobi.com/documentation/9.5/quickstart_windows/cs_python_installation_opt.html)). Gurobi offers a variety of free licenses for academic use. For more information, see the following [page](https://www.gurobi.com/academia/academic-program-and-licenses/).


After having successfully set up Gurobipy, install  **DRrank** via pip:

```bash
pip install drrank
```

## Usage

To compute rankings, provide the **fit** function with a matrix $P$ of posterior estimates of the probability observation i's latent measure (e.g., bias, quality, etc.) exceeds unit j's. That is, each element of this matrix takes the form:

$\pi_{ij} = Pr(\theta_i > \theta_j | Y_i = y_i, Y_j = y_j)$

**DRrank** expects these probabilities to satisfy $\pi_{ij} = 1-\pi_{ji}$. 


There are two ways to use **DRrank**.

First, one can supply a parameter $\lambda \in [0,1]$, which corresponds to the user's value of correctly ranking pairs of units relative to the costs of misclassifying them. $\lambda=1$ implies correct and incorrect rankings are valued equally, while $\lambda=0$ implies correct rankings are not valued at all. In pairwise comparisons between units, it is optimal to assign unit $i$ a higher grade than unit $j$ when $\pi_{ij} > 1/(1+\lambda)$, which implies $\lambda$ also determines the minimum level of posterior certainty required to rank units pairwise.

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

We also provide functionality to compute results for a list of values $\lambda$ in parallel:

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
