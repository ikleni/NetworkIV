import matplotlib.pylab as plt
import math
import numpy as np
import networkx as nx
import snap
import scipy.special
from tqdm.auto import tqdm
from scipy import sparse

from scipy import spatial, sparse
from scipy.stats import norm

### Network DGP ###

def gen_RGG(positions, r):
    """ 
    Returns an RGG from a given N-vector of dx1 positions (Nxd matrix). 
    
    positions = vector of node positions.
    r = linking threshold.
    """
    kdtree = spatial.KDTree(positions)
    pairs = kdtree.query_pairs(r) # default is Euclidean norm
    RGG = snap.GenRndGnm(snap.PUNGraph, len(positions), 0)
    for edge in (i for i in list(pairs)):
        RGG.AddEdge(int(edge[0]),int(edge[1]))
    return RGG

def ball_vol(d,r):
    """
    Returns the volume of a d-dimensional ball of radius r.
    """
    return math.pi**(d/2) * float(r)**d / scipy.special.gamma(d/2+1)


def gen_kappa(theta, d):
    """
    Returns kappa. Recall r_n = (kappa/n)^{1/d}.
    theta = true parameter.
    d = dimension of node positions.
    n = mean number of nodes.
    """
    vol = ball_vol(d,1)

    Phi2 = norm.cdf(-(theta[0] + 2*theta[1])/theta[3]) - norm.cdf(-(theta[0] + 2*theta[1] + theta[2])/theta[3])
    Phi1 = norm.cdf(-(theta[0] + theta[1])/theta[3]) - norm.cdf(-(theta[0] + theta[1] + theta[2])/theta[3])
    Phi0 = norm.cdf(-theta[0]/theta[3]) - norm.cdf(-(theta[0] + theta[2])/theta[3])
    gamma = math.sqrt( Phi2**2*0.25 + Phi1**2*0.5 + Phi0**2*0.25)
    
    return 1/(vol*gamma) - 0.3


def gen_V_exo(Z, eps, theta):
    """ 
    Returns 'exogenous' part of joint surplus function for each pair of nodes as a
    sparse upper triangular matrix.
    NB: This function is specific to the joint surplus function used in our
    simulations. 
    eps = sparse NxN upper triangular matrix. 
    Z = N-vector of binary attributes. 
    """
    N = Z.shape[0]
    sparse_ones = sparse.triu(np.ones((N,N)),1)
    Z_sum = sparse.triu(np.tile(Z, (N,1)) + np.tile(Z[:,np.newaxis], N),1)
    U = theta[0] * sparse_ones + theta[1] * Z_sum + eps
    return U

def gen_D(Pi, V_exo, theta2):
    """
    Returns a triplet of three snap graphs:
    D = opportunity graph with robust links removed.
    Pi_minus = subgraph of Pi without robustly absent potential links.
    Pi_exo = subgraph of Pi with only robust links.
    NB: This function is specific to the joint surplus used in our simulations.
    Pi = opportunity graph (in our case, the output of gen_RGG).
    V_exo = 'exogenous' part of joint surplus (output of gen_V_exo).
    theta2 = transitivity parameter (theta[2]).
    """
    N = V_exo.shape[0]
    D = snap.ConvertGraph(snap.PUNGraph, Pi)
    Pi_minus = snap.ConvertGraph(snap.PUNGraph, Pi)
    Pi_exo = snap.GenRndGnm(snap.PUNGraph, N, 0) 

    for edge in Pi.Edges():
        i = min(edge.GetSrcNId(), edge.GetDstNId())
        j = max(edge.GetSrcNId(), edge.GetDstNId())
        if V_exo[i,j] + min(theta2,0) > 0:
            D.DelEdge(i,j) 
            Pi_exo.AddEdge(i,j)
        if V_exo[i,j] + max(theta2,0) <= 0:
            D.DelEdge(i,j)
            Pi_minus.DelEdge(i,j)
 
    return (D, Pi_minus, Pi_exo)

def gen_G_subgraph(component, D, Pi_minus, Pi_exo, V_exo, theta2):
    """ 
    Returns a pairwise-stable network for nodes in component, via myopic best-
    response dynamics. This subnetwork is pairwise-stable taking as given the
    links in the rest of the network. Initial network for best-response dynamics
    is the opportunity graph. 
    NB: This function is specific to the joint surplus used in our simulations.
    
    component = component of D for which we want a pairwise-stable subnetwork.
    D, Pi_minus, Pi_exo = outputs of gen_D().
    V_exo = 'exogenous' part of joint surplus (output of gen_V_exo).
    theta2 = transitivity parameter (theta[2]).
    """
    stable = False
    meetings_without_deviations = 0

    D_subgraph = snap.GetSubGraph(D, component)

    # Start initial network on Pi, without robustly absent potential links.
    G = snap.GetSubGraph(Pi_minus, component)
    
    # For each node pair (i,j) linked in Pi_exo (i.e. their links are robust),
    # with either i or j in component, add their link to G.  Result is the
    # subgraph of Pi_minus on an augmented component of D.
    for i in component:
        for j in Pi_exo.GetNI(i).GetOutEdges():
            if not G.IsNode(j): G.AddNode(j)
            G.AddEdge(i,j)

    while not stable:
        # Need only iterate through links of D, since all other links are
        # robust.
        for edge in D_subgraph.Edges():
            # Iterate deterministically through default edge order order. Add or
            # remove link in G according to myopic best-respnose dynamics. If we
            # cycle back to any edge with no changes to the network, conclude
            # it's pairwise stable.
            i = min(edge.GetSrcNId(), edge.GetDstNId())
            j = max(edge.GetSrcNId(), edge.GetDstNId())
            cfriend = snap.GetCmnNbrs(G, i, j) > 0
            if V_exo[i,j] + theta2*cfriend > 0: # specific to model of V
                if G.IsEdge(i,j):
                    meetings_without_deviations += 1
                else:
                    G.AddEdge(i,j)
                    meetings_without_deviations = 0
            else:
                if G.IsEdge(i,j):
                    G.DelEdge(i,j)
                    meetings_without_deviations = 0
                else:
                    meetings_without_deviations += 1

        if meetings_without_deviations > D_subgraph.GetEdges():
            stable = True

    return snap.GetSubGraph(G, component)


def gen_G(D, Pi_minus, Pi_exo, V_exo, theta2, N):
    """
    Returns pairwise-stable network on N nodes. 
    
    D, Pi_minus, Pi_exo = outputs of gen_D().
    V_exo = 'exogenous' part of joint surplus (output of gen_V_exo).
    theta2 = transitivity parameter (theta[2]).
    """
    G = snap.GenRndGnm(snap.PUNGraph, N, 0) # initialize empty graph
    Components = snap.TCnComV()
    snap.GetWccs(D, Components) # collects components of D
    NIdV = snap.TIntV() # initialize vector
    for C in Components:
        if C.Len() > 1:
            NIdV.Clr()
            for i in C:
                NIdV.Add(i)
            tempnet = gen_G_subgraph(NIdV, D, Pi_minus, Pi_exo, V_exo, theta2)
            for edge in tempnet.Edges():
                G.AddEdge(edge.GetSrcNId(), edge.GetDstNId())
 
    # add robust links
    for edge in Pi_exo.Edges():
        G.AddEdge(edge.GetSrcNId(), edge.GetDstNId())

    return G

def gen_SNF(theta, d, N, r):
    """
    Generates pairwise-stable network on N nodes.
    """
        
    # positions uniformly distributed on unit cube
    positions = np.random.uniform(0,1,(N,d))
    # binary attribute
    Z = np.random.binomial(1, 0.5, N)
    # random utility terms
    eps = sparse.triu(np.random.normal(size=(N,N)), 1) * theta[3]
    # V minus endogenous statistic
    V_exo = gen_V_exo(Z, eps, theta)
    
    # generate random graphs 
    RGG = gen_RGG(positions, r) # initial RGG
    (D, RGG_minus, RGG_exo) = gen_D(RGG, V_exo, theta[2])
    # generate pairwise-stable network
    
    return gen_G(D, RGG_minus, RGG_exo, V_exo, theta[2], N)

def snap_to_nx(A):
    """ 
    Converts snap object to networkx object.
    """
    G = nx.Graph()
    G.add_nodes_from(range(A.GetNodes()))
    for edge in A.Edges():
        G.add_edge(edge.GetSrcNId(), edge.GetDstNId())
    return G

def leung_sparse_network(n = 20, s = 6):
    
    seed = s
    np.random.seed(seed=seed)

    #############
    ### Setup ###
    #############

    network_size = n

    theta_nf = np.array([-0.25, 0.5, 0.25, 1])
    kappa = gen_kappa(theta_nf, 2)

    ##################
    ### Estimation ###
    ##################

    n = network_size
    # network formation
    r = (kappa/n)**(1/float(2))
    G_temp = gen_SNF(theta_nf, 2, n, r)
    G = snap_to_nx(G_temp)
#     G_sparse = nx.to_scipy_sparse_matrix(G)
#     G = nx.to_numpy_array(G)
    
    return G
    
    
### Regression Functions ###

def reg_y_x(data_y, data_x):
    '''
    This function computes a regression of data_y on data_x
    args:
        data_y - N*1 vector
        data_x - N*K matrix
        
    output:
        beta_hat
        data_y_hat
    '''
    
    beta_hat   = np.linalg.inv(data_x.T@data_x)@(data_x.T@data_y)
    data_y_hat = data_x@beta_hat 
    
    return beta_hat, data_y_hat

def two_SLS( data_y, data_x_exog, data_x_endog, data_iv):

    data_x = np.hstack([data_x_exog, data_x_endog])
    data_z = np.hstack([data_x_exog, data_iv])

    pi_hat, data_x_hat = reg_y_x(data_x, data_z)

    beta_iv, data_y_iv = reg_y_x(data_y, data_x_hat)

    return beta_iv
    
def MC_linear_IV(params_dict, n_sim = 1000, seed = 2023):
    '''
    This function simulates a linear IV model where peer instruments can be used as own instruments.

    args:
        params_dict: specifies parameters, such as N data points, or the number of regressors for this run
        n_sim:       specifies number of MC simulations
        seed:        fixes the seed to make results replicable

    '''

    N       = params_dict['N'] 
    beta    = params_dict['beta']
    rho     = params_dict['rho']
    pi      = params_dict['pi']
    gamma   = params_dict['gamma']

    placeholder_beta_iv     = []
    placeholder_beta_netwiv = []

    for n in tqdm(range(n_sim)):
        np.random.seed(seed + n)
        # generate error terms
        u      = np.random.normal(size = (N, 1))
        xi     = np.random.normal(size = (N, 1))
        e      = rho*u + xi

        # generate graph data
        G           = leung_sparse_network(n = N)
        G           = nx.to_numpy_array(G)
        degree      = 0.0001 + G@np.ones((N,1)) # I add a small number to avoid division by zero for isolated agents

        # generate covariates and ivs
        data_z      = np.random.normal(size = (N, 1))
        data_z_netw = (G@data_z)/degree # weighted by degree average of peer z
        data_z_comb = np.hstack([data_z, data_z_netw])

        data_x_1    = np.random.normal(size = (N, 1))
        data_x_2    = pi*data_z + gamma*data_z_netw + u
        data_x      = np.hstack([data_x_1, data_x_2]) # should be N*2

        data_y = data_x@beta + e

        beta_iv = two_SLS(data_y, data_x_1, data_x_2, data_z)
        beta_netwiv = two_SLS(data_y, data_x_1, data_x_2, data_z_comb)

        placeholder_beta_iv.append(beta_iv)
        placeholder_beta_netwiv.append(beta_netwiv)
    
    return placeholder_beta_iv, placeholder_beta_netwiv

def simple_transform_instrument(x, max_deg = 2):
        """Chen 2007 instrument transform"""

        if x.shape[1] == 2:

            x1 = x[:, 0]
            x2 = x[:, 1]

            iv_basis = np.array(
            [np.ones(len(x1))]
            + [x1 ** i for i in range(1, max_deg + 1)]
            + [np.maximum(x1 - 0.5, 0) ** (max_deg)]
#             + [np.maximum(x1 - 0.25, 0) ** (max_deg)]
#             + [np.maximum(x1 - 0.75, 0) ** (max_deg)]
            + [x2 ** i for i in range(1, max_deg + 1)]
            + [np.maximum(x2 - 0.5, 0) ** (max_deg)]
#             + [np.maximum(x2 - 0.25, 0) ** (max_deg)]
#             + [np.maximum(x2 - 0.75, 0) ** (max_deg)]
            + [
                x1 * x2,
#                 x1 * np.maximum(x2 - 0.25, 0) ** (max_deg),
#                 x2 * np.maximum(x1 - 0.25, 0) ** (max_deg),
#                 x1 * np.maximum(x1 - 0.75, 0) ** (max_deg),
#                 x2 * np.maximum(x2 - 0.75, 0) ** (max_deg),
            ]
            ).T

            return iv_basis

        elif x.shape[1] == 1:

            x1 = x[:, 0]

            iv_basis = np.array(
            [np.ones(len(x1))]
            + [x1 ** i for i in range(1, max_deg + 1)]
            + [np.maximum(x1 - 0.5, 0) ** (max_deg)]
#             + [np.maximum(x1 - 0.25, 0) ** (max_deg)]
#             + [np.maximum(x1 - 0.75, 0) ** (max_deg)]
            ).T
            return iv_basis

        else:
            raise ValueError('The code can only accommodate 2 or less instruments!')




def sid_funct(x):
    '''
    This is an example of a non-linear function used in simulations for npiv
    '''
    return np.sin(6*x)*np.log(x)


def MC_NPIV_est(params_dict, seed = 2023):
    '''
    This code estimates an NPIV model, using two stage least squares. Basis functions are specified according to
    @simple_transform_instrument. Now they are just polynomials.
    '''

    N       = params_dict['N'] 
    Sigma   = params_dict['Sigma'] # error term covariance matrix
    rho     = params_dict['rho'] # ill-posedness degree
    gamma   = params_dict['gamma'] # network IV relevance
    n_b_terms = params_dict['n_b_terms']

    ## estimation 

    np.random.seed(seed)
    # generate error terms
    error_terms = np.random.multivariate_normal(np.zeros((3)), Sigma  , size = N)
    u           = error_terms[:, 0].reshape((N,1))
    v           = error_terms[:, 1].reshape((N,1))
    z           = error_terms[:, 2].reshape((N,1))
    I           = np.random.binomial(1, 0.5, size = (N,1))

    # generate graph data
    G           = leung_sparse_network(n = N)
    G           = nx.to_numpy_array(G)
    degree      = 0.0001 + G@np.ones((N,1)) # I add a small number to avoid division by zero for isolated agents
    z_netw      = (G@z)/degree # weighted by degree average of peer z

    # generate covariates and y
    pre_x = I*(z + gamma*z_netw + rho*v) + (1-I)*(v)
    x     = norm.cdf(pre_x)
    w_1   = norm.cdf(z)
    w_2   = norm.cdf(z_netw)
    w_c   = np.hstack([w_1, w_2])
    y     = np.sin(6*x)*np.log(x)+ u

    # generate basis terms
    
    P      = simple_transform_instrument(x, max_deg = n_b_terms) 
    Q      = simple_transform_instrument(w_1, max_deg = n_b_terms) 
    Q_netw = simple_transform_instrument(w_c, max_deg = n_b_terms) 


    # projections 
    _, P_hat             = reg_y_x(P, Q)
    _, P_hat_netw   = reg_y_x(P, Q_netw)
    beta, _           = reg_y_x(y, P_hat)
    beta_netw, _ = reg_y_x(y, P_hat_netw)

    return beta, beta_netw

def MC_NPIV_loss(params_dict, beta, beta_netw, seed = 2023):

    '''
    This function computes an L2 loss of an NPIV estimator that ignores
    network IV and that uses network IV. The user has to provide estimated betas
    '''

    N       = params_dict['N'] 
    Sigma   = params_dict['Sigma'] # error term covariance matrix
    rho     = params_dict['rho'] # ill-posedness degree
    gamma   = params_dict['gamma'] # network IV relevance
    n_b_terms = params_dict['n_b_terms']

    ## l_2 loss
    np.random.seed(seed+1)
    
    # generate error terms
    error_terms = np.random.multivariate_normal(np.zeros((3)), Sigma  , size = N)
    v           = error_terms[:, 1].reshape((N,1))
    z           = error_terms[:, 2].reshape((N,1))
    I           = np.random.binomial(1, 0.5, size = (N,1))

    # generate graph data
    G           = leung_sparse_network(n = N)
    G           = nx.to_numpy_array(G)
    degree      = 0.0001 + G@np.ones((N,1)) # I add a small number to avoid division by zero for isolated agents
    z_netw      = (G@z)/degree # weighted by degree average of peer z

    # generate covariates and y
    pre_x = I*(z + gamma*z_netw + rho*v) + (1-I)*(v)
    x     = norm.cdf(pre_x)
    P     = simple_transform_instrument(x, max_deg = n_b_terms) 
    
    h_true     = sid_funct(x)
    h_hat      = P@beta
    h_hat_netw = P@beta_netw


    l2_loss_no_netw = np.sqrt( np.mean((h_true - h_hat)**2 )) 
    l2_loss_netw    = np.sqrt( np.mean((h_true - h_hat_netw)**2 )) 


    return l2_loss_no_netw, l2_loss_netw

def MC_NPIV_plot(params_dict, beta, beta_netw, n_data_points = 1000):

    '''
    This function generates data to visualise estimated functions in @MC_NPIV_est
    for a range of equidistance points on a unit interval.
    '''

    n_b_terms = params_dict['n_b_terms']

    ## data for plots

    interval = np.array([i/n_data_points for i in range(1,n_data_points+1)]).reshape((n_data_points,1))
    P        = simple_transform_instrument(interval, max_deg = n_b_terms) 
    
    h_true     = sid_funct(interval)
    h_hat      = P@beta
    h_hat_netw = P@beta_netw

    return h_true, h_hat, h_hat_netw



    





