##########################################################
#### Loss functions and estimation functions          ####
##########################################################
import gurobipy as gp

def tau(i_j, Pij, Dij):
    """
    Computed expected tau
    Parameters:
    i_j: Coordinates of the Pij
    Pij: Posterior estimates of the probability of observation i being more biased than observation j
    Dij: Pairwise indicators
    Eij: Pairwise indicators
    """
    return gp.quicksum(
                Pij[a]*Dij[a] + Pij[(a[1],a[0])]*Dij[(a[1],a[0])]
                - Pij[a]*Dij[(a[1],a[0])] - Pij[(a[1],a[0])]*Dij[a]
                            for a in i_j)


def dp(i_j, Pij, Dij):
    """
    Computed expected discordance proportion
    Parameters:
    i_j: Coordinates of the Pij
    Pij: Posterior estimates of the probability of observation i being more biased than observation j
    Dij: Pairwise indicators
    Eij: Pairwise indicators
    """
    # Set objective function
    return gp.quicksum(  
                        Pij[a]*Dij[(a[1],a[0])] + Pij[(a[1],a[0])]*Dij[a]
                                for a in i_j)


