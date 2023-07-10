##########################################################
#### Report card estiamtes functions                  ####
##########################################################
import pandas as pd
import numpy as np
import os
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse.csgraph import connected_components
import multiprocessing
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
from drrank_loss import dp, tau

## function to clean the Pij matrix ##
def clean_data(pi):
    """
    Function to prepare the P_ij data to be fitted with the report card model
    p_ij: a numpy matrix of (n_obs,n_obs) dimension containing the posterior estimates of the probabilities
          P_ij of observation i being more biased than observation j - P_ij = Pr(Theta_i > Theta_j | Y_i, Y_j)
    """
    # Create a copy to not modify data in place
    p_ij = pi.copy()

    # Requires np.array as inputs
    if type(p_ij) is not np.ndarray:
        raise ValueError("Pi matrix must be numpy array.")

    # Fill the diagonal with 0, so no gain / cost to ranking i tied with i
    np.fill_diagonal(p_ij, 0)

    # Check if p_ij = 1-p_ij.T
    if not np.allclose(p_ij + np.eye(p_ij.shape[0]), (1-p_ij).T):
        print("Warrning: Pi matrix does not appear equal to (1 - Pi.T)")
        print("Max deviation {:7.6f}".format(np.max(np.abs(p_ij +  np.eye(p_ij.shape[0]) -  (1-p_ij).T))))

    # Get coordinate representation
    N = p_ij.shape[0]
    i = np.repeat(np.arange(N),N) + 1
    j = np.tile(np.arange(N),N) + 1
    v = np.ravel(p_ij)

    # Make a dataframe out of the coordinate representation of the matrix
    df_p_ij = pd.DataFrame({'i':i,'j':j,'P_ij':v})

    # Get the matrix coordinates
    df_p_ij['idx'] = list(zip(df_p_ij.i, df_p_ij.j))

    # Make a dictionary with keys the matrix coordinates, values the P_ij
    p_ij_dict = {x:v for x, v in zip(df_p_ij['idx'], df_p_ij['P_ij'])}

    # Split a single dictionary into multiple dictionaries
    i_j, Pij = gp.multidict(p_ij_dict)

    return i_j, Pij

## main function ##
def report_cards(i_j, Pij, lamb = None, DR = None, loss = 'binary', save_controls = False, save_dir = "dump", save_name = '_debug', FeasibilityTol = 1e-9, IntFeasTol = 1e-9, OptimalityTol = 1e-9):
    """
    Compute the report cards via Gurobi optimization
    Parameters:
    i_j: Coordinates of the Pij
    Pij: Posterior estimates of the probability of observation i being more biased than observation j
    lamb:  Tuning parameter trading off the gains of correctly ranking pairs of observations against the cost of misclassifying them - either lamb or DR must be specified
    DR: Discordance rate (i.e. shares of observation pairs misranked according to their grades) - either lamb or DR must be specified
    save_controls: if True, saves the estimates for debugging purposes (default = False)
    save_dir: if save_controls == True, name for the directory in which the estimates will be saved, if the default directory is used, it creates a folder named "dump" (default = "dump")
    save_name: if save_controls == True, name for the file in which the estimates will be saved (default = "_debug")
    FeasibilityTol: Feasibility Tollerance, Gurobi parameter (default = 1e-9)
    IntFeasTol: Integer feasibility Tollerance, Gurobi parameter (default = 1e-9)
    OptimalityTol: Optimality Tollerance, Gurobi parameter (default = 1e-9)
    """
    # Check that lambda and DR are correctly specified
    if (lamb is not None) & (DR is not None) :
        raise AssertionError("Must supply either lambda or DR, but not both.")
    if (lamb != None):
        if (lamb > 1) or (lamb < 0):
            raise AssertionError("Lambda must be within [0,1].")
    if (DR != None):
        if (DR > 1) or (DR < 0):
            raise AssertionError("DR must be within [0,1].")

    # the model
    with gp.Env() as env, gp.Model(env=env) as model:
        n_obs = max(i_j)[0] + 1

        # Tolearances
        model.Params.FeasibilityTol = FeasibilityTol
        model.Params.IntFeasTol = IntFeasTol
        model.Params.OptimalityTol = OptimalityTol
        
        # control variables
        Dij = model.addVars(i_j, vtype=GRB.BINARY, name="Dij")
        Eij = model.addVars(i_j, vtype=GRB.BINARY, name="Eij")

        if lamb is not None:
            loss = (1-lamb)*dp(i_j, Pij, Dij, Eij) - lamb*tau(i_j, Pij, Dij, Eij)
        elif DR is not None:
            loss = -tau(i_j, Pij, Dij, Eij)
        else:
            raise AssertionError("Must supply either lambda or DR.")

        model.setObjective(loss, GRB.MINIMIZE)

        # Add constraints to force transitivity
        model.addConstrs(
            (Dij[(i, j)] + Dij[(j, k)] - Dij[(i, k)] <= 1 for k in range(1, n_obs) for j in range(1, n_obs) for i in
             range(1, n_obs)
             if ((i, j) in i_j) & ((j, k) in i_j) & ((i, k) in i_j) & (i != j) & (i != k) & (j != k)),
            name="SST1")
        model.update()

        model.addConstrs(
            (Dij[(i, k)] + (1 - Dij[(j, k)]) - Dij[(i, j)] <= 1 for k in range(1, n_obs) for j in range(1, n_obs) for i in
             range(1, n_obs) if ((i, j) in i_j) & ((j, k) in i_j) & ((i, k) in i_j) & (i != j) & (i != k) & (j != k)),
            name="SST2")
        model.update()

        model.addConstrs(
            (Eij[(i, j)] + Eij[(j, k)] - Eij[(i, k)] <= 1 for k in range(1, n_obs) for j in range(1, n_obs) for i in
             range(1, n_obs)
             if ((i, j) in i_j) & ((j, k) in i_j) & ((i, k) in i_j) & (i != j) & (i != k) & (j != k)),
            name="SST3")
        model.update()

        model.addConstrs((Eij[(i,j)] + Dij[(i,j)] + Dij[(j,i)] == 1 for j in range(1, n_obs) for i in range(1, n_obs)),
                         name="logic1")
        model.update()
        model.addConstrs((Eij[(i, i)] == 1 for i in range(1, n_obs)),
                         name="logic2")
        model.update()

        # DR constraint, if necessary
        if DR is not None:
            npairs = np.sum(Pij.values())
            model.addConstr(dp(i_j, Pij, Dij, Eij) <= DR*npairs)
        model.update()

        # First optimize() if call fails - need to set NonConvex to 2
        try:
            print("Optimizing model...")
            model.optimize()
        except gp.GurobiError:
            print("...ptimize failed due to non-convexity")
            print("...optimizing with non-convexity")
            # Solve bilinear model
            model.Params.NonConvex = 2
            model.optimize()
        print("...optimized model successfully")

        ## print results
        print("%%%%%%%%%%%%%%%%%%%%%%%%%")
        if lamb is not None:
            print('lambda: %g' % lamb)
        elif DR is not None:
            print('DR: %g' % DR)
        print('Obj: %g' % model.ObjVal)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%")

        ## get results
        D_ij_hat = model.getAttr('x', Dij)
        E_ij_ij_hat = model.getAttr('x', Eij)

        data_items = D_ij_hat.items()
        data_list = list(data_items)
        df = pd.DataFrame(data_list, columns=['i_j', 'D_ij'])

        data_items = E_ij_ij_hat.items()
        data_list = list(data_items)
        df2 = pd.DataFrame(data_list, columns=['i_j', 'E_ij'])

        df = df.merge(df2, on='i_j', how='outer', validate="1:1", indicator=True)
        df.drop(columns=['_merge'], inplace=True)

        ## check violations
        df['D_ij'] = df['D_ij'].round(2)
        df['E_ij'] = df['E_ij'].round(2)
        df['sum'] = df['D_ij'] + df['E_ij']

        if df['sum'].max()>1.000000000002:
            raise AssertionError("Violation of transitivity constraints")

        df['D_ji'] = ((df['D_ij'] == 0) & (df['E_ij'] == 0)).astype(int)
        df['lambda'] = lamb

        df[['i', 'j']] = pd.DataFrame(df2['i_j'].tolist(), index=df.index)

        # For debugging save for later
        if save_controls==True:
            if save_dir == "dump":
                # Create a folder if the user has not specified any directory
                os.makedirs("dump/", exist_ok=True)
            print("Saving estimates in {}/df_aux_{}_{}.csv for debugging purposes".format(save_dir,lamb, save_name))
            df.to_csv("{}/df_aux_{}_{}.csv".format(save_dir,lamb, save_name), index = False)

        ## get the implied groups!
        print("Getting the implied groups...")
        df_groups = get_groups(df)
        if lamb is not None:
            df_groups.rename(columns={'groups': 'grades_lamb{}'.format(lamb)}, inplace=True)
        elif DR is not None:
            df_groups.rename(columns={'groups': 'grades_DR{}'.format(DR)}, inplace=True)

        return df_groups
    

## Function to get groups from ranking output 
def get_groups(df):
    """
    From the estimated ranks from the function "report_cards", get the observation groups
    df: dataframe with the Gurobi estimates
    Return observation groups
    """

    ## get groups
    df_aux = df[df['E_ij'] == 1]

    # generate adjacency matrix
    sim_mat = pd.crosstab(df_aux.i, df_aux.j)

    sim_mat2 = sim_mat.to_numpy()
    n_components, labels = connected_components(sim_mat2, directed=False)

    # mapping from observations to groups
    obs_groups = pd.DataFrame({"obs_idx": sim_mat.keys(),
                                "groups": labels})

    # test for violation of transitivity
    print(">> Test for violation of transitivity")
    if len(labels)==1:
        assert len(df)==len(df_aux)
    else:
        for l in np.unique(labels):
            f_in_gr = obs_groups["obs_idx"][obs_groups['groups'] == l].to_numpy()
            df_l = df[(df['i'].isin(f_in_gr)) & (df['j'].isin(f_in_gr))]

            #print(len(df_l))
            assert len(df_l) == len(df_l[df_l['E_ij']==1])

    print("%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Number of groups: {}".format(len(np.unique(labels))), flush=True)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%")

    ## bubble sort
    sorted_labels = np.unique(labels)

    swapped = True
    while swapped:
        swapped = False
        for i in range(len(sorted_labels) - 1):
            i_lab = sorted_labels[i]
            i_1_lab = sorted_labels[i + 1]

            # who is bigger?
            f_in_gr = obs_groups["obs_idx"][obs_groups['groups'] == i_lab].to_numpy()
            f_in_gr_1 = obs_groups["obs_idx"][obs_groups['groups'] == i_1_lab].to_numpy()

            df_aux = df[(df['j'].isin(f_in_gr)) & (df['i'].isin(f_in_gr_1))]
            dij = df_aux['D_ij'].max()
            dji = df_aux['D_ji'].max()

            if len(df_aux) == 0:
                df_aux = df[(df['i'].isin(f_in_gr)) & (df['j'].isin(f_in_gr_1))]
                dij = df_aux['D_ij'].max()
                dji = df_aux['D_ji'].max()

                condition = dij < dji
                # check we don't have violation of transitivity
                if dij == dji:
                    raise AssertionError("Violation of transitivity")
            else:
                condition = dji < dij
                # check we don't have violation of transitivity
                if dij == dji:
                    raise AssertionError("Violation of transitivity")

            if condition:
                sorted_labels[i], sorted_labels[i + 1] = sorted_labels[i + 1], sorted_labels[i]
                swapped = True

    print(">> Finished bubble sort!")
    new_vals = {sorted_labels[i]: i for i in range(len(sorted_labels))}
    obs_groups['groups'] = obs_groups['groups'].replace(new_vals) + 1

    return obs_groups


## function to fit the ranking model ##
def fit(Pij, lamb, DR = None, save_controls = False, save_dir = "dump", save_name = '_debug', FeasibilityTol = 1e-9, IntFeasTol = 1e-9, OptimalityTol = 1e-9):
    """
    Function to fit the report card model on a Pij matrix of posterior estimates of bias
    Parameters:
    i_j: Coordinates of the Pij
    Pij: Posterior estimates of the probability of observation i being more biased than observation j
    lamb:  Tuning parameter trading off the gains of correctly ranking pairs of observations against the cost of misclassifying them
    DR: Discordance proportion (i.e. shares of observation pairs misranked according to their grades) - either lamb or DR must be specified
    save_controls: if True, saves the estimates for debugging purposes (default = False)
    save_dir: if save_controls == True, name for the directory in which the estimates will be saved, if the default directory is used, it creates a folder named "dump" (default = "dump")
    save_name: if save_controls == True, name for the file in which the estimates will be saved (default = "_debug")
    FeasibilityTol: Feasibility Tollerance, Gurobi parameter (default = 1e-9)
    IntFeasTol: Integer feasibility Tollerance, Gurobi parameter (default = 1e-9)
    OptimalityTol: Optimality Tollerance, Gurobi parameter (default = 1e-9)
    """    
    # Clean the Pij and retrieve the sparse representation
    i_j, Pij = clean_data(Pij)

    # Produce the report cards
    res_df = report_cards(i_j, Pij, lamb, DR, save_controls = save_controls, save_dir = save_dir, save_name = save_name, FeasibilityTol = FeasibilityTol, IntFeasTol = IntFeasTol, OptimalityTol = OptimalityTol)

    # Compute condorcet ranks
    condorcet = report_cards(i_j, Pij, 1, None, save_controls = False, save_dir = save_dir, save_name = save_name, FeasibilityTol = FeasibilityTol, IntFeasTol = IntFeasTol, OptimalityTol = OptimalityTol)
    res_df['condorcet_rank'] = condorcet['grades_lamb1']

    return res_df

## function to get different ranking estimations based on a list of lambdas ##
def fit_multiple(Pij, lamb_list, ncores = 1, save_controls = False, save_dir = "dump", save_name = '_debug', FeasibilityTol = 1e-9, IntFeasTol = 1e-9, OptimalityTol = 1e-9):
    """
    Function to fit the report card model on a Pij matrix of posterior estimates of bias
    -> gives back a dataframe with results for various lambda parameters
    Parameters:
    i_j: Coordinates of the Pij
    Pij: Posterior estimates of the probability of observation i being more biased than observation j
    lamb_list:  List of lambdas to be used (lambda: Tuning parameter trading off the gains of correctly ranking pairs of observations against the cost of misclassifying them)
    ncores: Number of cores, if set to -1 uses all the cores available (default = 1)
    save_controls: if True, saves the estimates for debugging purposes (default = False)
    save_dir: if save_controls == True, name for the directory in which the estimates will be saved, if the default directory is used, it creates a folder named "dump" (default = "dump")
    save_name: if save_controls == True, name for the file in which the estimates will be saved (default = "_debug")
    FeasibilityTol: Feasibility Tollerance, Gurobi parameter (default = 1e-9)
    IntFeasTol: Integer feasibility Tollerance, Gurobi parameter (default = 1e-9)
    OptimalityTol: Optimality Tollerance, Gurobi parameter (default = 1e-9)
    """
    # If ncores is -1, use all CPUs
    if ncores == -1:
        ncores = multiprocessing.cpu_count() 

    # Clean the Pij and retrieve the sparse representation
    i_j, Pij = clean_data(Pij)

    # final dataframe to store all results
    n_obs = max(i_j)[0] + 1
    final_df = pd.DataFrame({'obs_idx': range(1, n_obs)})

    def report_cards_helper(x):
        """
        Helper function to run report_cards within a multiprocessing tool
        x: lambda value over which we iterate
        """
        return report_cards(i_j, Pij, x, DR = None, save_controls = save_controls, save_dir = save_dir, save_name = save_name, FeasibilityTol = FeasibilityTol, IntFeasTol = IntFeasTol, OptimalityTol = OptimalityTol)

    # Compute using multiprocessing to speed up everything
    print("Computing grades with {} cores".format(ncores), flush=True)
    with ThreadPool(ncores) as p:
        dfs = p.map(report_cards_helper, lamb_list)

    # Get the final result dataset
    for d in dfs:
        final_df = final_df.merge(d, on='obs_idx', validate="1:1")

    return final_df


def fig_ranks(ranking, posterior_features, gradecol=None, ylabels=None, show_plot=True, save_path=None):
    """
    Function to plot an histogram of the estimates
    Arguments:
        ranking: dataframe with ranking results from drrank.fit()
        posterior_features: posterior features computed from the drrank.prior_estimate() class
        gradecol: column name with the ranking grades, useful when fitting multiple lambdas. If None, looks for the right columns and pick the first one
        ylabels: provide the names of the observations, if None simply shows the observation number
        show_plot: set to True to display the plot
        save_path: specify the path to save the plot, set to None to avoid saving it
    """
    # Find smallest lambda that delivers num_groups
    if gradecol is not None:
        group = [x for x in ranking.keys() if gradecol in x][0]
    else:
        group = [x for x in ranking.keys() if 'grades_' in x][0]
    print("Using group {}".format(group))

    # First step - get posterior features:
    # Merge posterior features with the group rankings
    df_plot = pd.concat([ranking, posterior_features], axis = 1)

    # Sort everything
    df_plot = df_plot.sort_values([group], ascending=False)

    # Get upper and lower bounds
    df_plot["lb"] = df_plot['pmean'] - df_plot['lci']
    df_plot["ub"] = df_plot['uci'] - df_plot['pmean']

    # Construct figure
    fig, ax1 = plt.subplots(dpi=400, figsize=(6,7))
    y_range = np.arange(len(df_plot))
    groups = df_plot[group].unique()
    # Prepare labels for each group
    if len(groups) > 6:
        labels = [r'${} \bigstar$'.format(c) for c in range(len(groups),0,-1)]
    else:
        labels = [r'$\bigstar$'*c for c in range(len(groups),0,-1)]

    # Iterate and produce a plot for each group
    mean_lines = []
    for i, gr in enumerate(np.flip(groups)):
        idx = df_plot[group] == gr
        mean_lines += [ax1.errorbar(df_plot['pmean'][idx], y_range[idx],
                        xerr=[df_plot['lb'][idx], df_plot['ub'][idx]],
                        alpha=0.6, ms=1.5, elinewidth=0.6,
                    fmt='o', capsize=1.5, label=labels[i]
                        )]

    plt.grid(axis='y', alpha=0.35, linewidth=0.3)
    plt.legend(mean_lines,labels, loc='upper left', markerscale=1, bbox_to_anchor=(0,1))
    if ylabels != None:
        plt.yticks(y_range, ylabels, fontsize=4)    

    ax1.set_xlabel('Posterior means')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    if save_path != None:
        plt.tight_layout()
        plt.savefig(save_path, format='pdf', dpi=300)
        print("Figure saved!")

    if show_plot:
        plt.show()

    plt.clf()
    plt.close('all')