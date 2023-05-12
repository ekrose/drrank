#############################################################################
#### Simple function to create a random Pij matrix to test the functions ####
#############################################################################
import numpy as np

def simul_data(size = 25, rng = None):
    """
    Specify the size of our dataset, then simulate a matrix of Pijs randomly drawing from a uniform distribution
    size: size of the dataset (default = 25)    
    rng: random number generator from Numpy, use it to specify a seed
    """
    sim = np.empty(shape = (size, size))
    if rng == None:
        rng = np.random.default_rng(seed = None)

    bias = rng.uniform(size = size)

    # Fill each value
    for i in range(0,len(sim)):
        for j in range(0,len(sim)):
            sim[i,j] = 0.5*(1 + (bias[i] - bias[j])/(bias[i]+bias[j]))

    # Fill the diagonal with 0s
    np.fill_diagonal(sim, 0)

    # Get the probabilities
    # Then set the bottom-part p_ji to 1-p_ij (i.e. Prob of ranking i > j = 1 - Prob of ranking j > i)
    sim.T[np.triu_indices(sim.shape[0],1)] = 1 - sim[np.triu_indices(sim.shape[0],1)]
    return sim