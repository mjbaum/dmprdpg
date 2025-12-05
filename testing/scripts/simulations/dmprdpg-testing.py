import dmprdpg
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
import os, glob, argparse
from collections import Counter
from scipy.stats import bernoulli
from numba import jit

# Initialize Static Parameters
K = 10
T = 3
d = 2

## PARSER to give parameter values
parser = argparse.ArgumentParser()
## Set destination folder for output
parser.add_argument("-f","--folder", type=str, dest="folder", default="simulation_1", const=True, nargs="?",\
    help="String: name of the folder for the input files.")
parser.add_argument("-n", type=int, dest="n", default=100, const=True, nargs="?",\
	help="Integer: number of nodes in each graph. Default: d=100.")
parser.add_argument("-eta", type=float, dest="eta", default=0.0, const=True, nargs="?",\
	help="Float: size of perturbation. Default: 0.0.")
parser.add_argument("-b", type=int, dest="b", default=1000, const=True, nargs="?",\
	help="Integer: number of bootstraps per test. Default: n=1000.")
parser.add_argument("-r", type=int, dest="r", default=1000, const=True, nargs="?",\
	help="Integer: number of replicates. Default: n=1000.")

## Parse arguments
args = parser.parse_args()
input_folder = args.folder

n = args.n
eta = args.eta
num_boot = args.b
num_replicates = args.r


# Define Functions to Calculate Bootstrap Results
def simulate_dmpsbm(n, B_dict, K=None, T=None, prior_G=None, prior_G_prime=None, seed=None, z_shared=False, undirected=False):
    # Initialise number of layers and time steps from B[0,0] (if present)
    if (0,0) in B_dict:
        G = B_dict[0,0].shape[0]
        G_prime = B_dict[0,0].shape[1]
    else:
        raise ValueError("B_dict must contain an entry for (0,0)")
    # Check if undirected is Boolean. If it is, check that the B matrices are symmetric.
    if not isinstance(undirected, bool):
        raise ValueError("undirected must be a boolean")
    # z_shared must be boolean
    if not isinstance(z_shared, bool):
        raise ValueError("z_shared must be a boolean")
    if undirected:
        if not all(np.allclose(B_dict[key], B_dict[key].T) for key in B_dict.keys()):
            raise ValueError("All matrices in B_dict must be symmetric")
        ## z_shared must be True if undirected is True
        if not z_shared:
            raise ValueError("z_shared must be True if undirected is True")
    # Check z_shared and return an error if G != G_prime
    if z_shared:
        if G != G_prime:
            raise ValueError("G must be equal to G_prime if z_shared is True")
    ## Check that all matrices in B_dict have the same dimensions
    if not all(B_dict[key].shape == (G, G_prime) for key in B_dict.keys()):
        raise ValueError("All matrices in B_dict must have the same dimension")
    ## If K and T are not provided, set them to the number of unique entries in the rows/columns of the keys of B_dict
    if K is None:
        K = len(set(key[0] for key in B_dict.keys()))
    if T is None:
        T = len(set(key[1] for key in B_dict.keys()))
    ## Check that the entries of B_dict are all possible pairs of range(K) and range(T)
    if not all(key in B_dict for key in [(k, t) for k in range(K) for t in range(T)]):
        raise ValueError("B_dict must contain all possible (k,t) pairs for k=0,...,K-1 and t=0,...,T-1")
    ## If priors are None, assume identical probability vectors for all layers and times
    if prior_G is None:
        prior_G = [1/G] * G
    else:
        if len(prior_G) != G:
            raise ValueError("Length of prior_G must match G.")
        if not np.isclose(sum(prior_G), 1):
            raise ValueError("Priors must sum to 1")
        if not all(p >= 0 for p in prior_G):
            raise ValueError("Priors must be non-negative")
    if not z_shared:
        if prior_G_prime is None:
            prior_G_prime = [1/G_prime] * G_prime
        else:
            if len(prior_G_prime) != G_prime:
                raise ValueError("Length of prior_G_prime must match G_prime.")
            if not np.isclose(sum(prior_G_prime), 1):
                raise ValueError("Priors must sum to 1")
            if not all(p >= 0 for p in prior_G_prime):
                raise ValueError("Priors must be non-negative")
    ## Set seed if provided
    if seed is not None:
        np.random.seed(seed)
    ## Generate the group labels
    z = np.random.choice(range(G), size=n, p=prior_G)
    if not z_shared:
        z_prime = np.random.choice(range(G_prime), size=n, p=prior_G_prime)
    else:
        z_prime = np.copy(z)
    ## Simulate a stochastic blockmodel for each matrix in B_dict, storing A_{kt} in a sparse matrix
    A_dict = {}
    ## Obtain the graph as an edgelist
    for k in range(K):
        for t in range(T):
            edgelist = []
            if undirected:
                for i in range(n):
                    for j in range(i, n):
                        if np.random.binomial(1, B_dict[k, t][z[i], z[j]]) == 1:
                            edgelist += [(i, j), (j, i)]
            else:
                for i in range(n):
                    for j in range(n):
                        if np.random.binomial(1, B_dict[k, t][z[i], z_prime[j]]) == 1:
                            edgelist += [(i, j)]
            # Extract nodes and weights from the edge list
            rows = [edge[0] for edge in edgelist]
            cols = [edge[1] for edge in edgelist]
            data = [1.0] * len(edgelist)
            # # Create the sparse adjacency matrix in COO format
            adjacency_matrix = coo_matrix((data, (rows, cols)), shape=(n,n))
            # Convert to CSR format
            A_dict[k,t] = adjacency_matrix.tocsr()
    ## Return output
    if undirected:
        return A_dict, z
    else:
        if z_shared:
            return A_dict, z
        else:
            return A_dict, z, z_prime

@jit
def generate_adjacency(P_mat):
    n, m  = P_mat.shape[0], P_mat.shape[1]
    adj = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            adj[i,j] = np.random.binomial(1, P_mat[i,j])
    return adj

def bootstrap(left_embeddings, right_embeddings, n_samples, K=10, T=3, d=2):
    if len(right_embeddings.shape) != 3:
        raise ValueError("Input must be a 3D tensor")
    ## Calculate the mean matrix on the third axis of the tensor
    Xbar = np.mean(left_embeddings, axis=2)
    Pmat_dict = {}
    for k in range(K):
        for t in range(T):
            Pmat_dict[(k,t)] = clean_Pmat(np.matmul(Xbar ,np.transpose(right_embeddings[:,:,t])))
    empirical_dist = list()
    for _ in range(n_samples):
        adj_dict = {}
        for k in range(K):
            for t in range(T):
                adj_dict[(k,t)] = sparse.csr_matrix(generate_adjacency(Pmat_dict[(k,t)]))
        Xboot, Yboot = dmprdpg.duase(adj_dict, K=K, T=T, d=d)
        bootstrap_value = test_statistic_tensor(Xboot)
        empirical_dist.append(bootstrap_value)
    return empirical_dist

@jit
def clean_Pmat(P_mat):
    n = P_mat.shape[0]
    for i in range(n):
        for j in range(n):
            if P_mat[i,j] < 0:
                P_mat[i,j] = 0
            if P_mat[i,j] > 1:
                P_mat[i,j] = 1
    return P_mat

def test_statistic_tensor(Y):
    ## Check that the input is a 3D tensor
    if len(Y.shape) != 3:
        raise ValueError("Input must be a 3D tensor")
    ## Calculate the mean matrix on the third axis of the tensor
    matbar = np.mean(Y, axis=2)
    result = 0
    for j in range(Y.shape[2]):
        result += (np.linalg.norm(Y[:,:,j] - matbar, ord='fro') ** 2)
    return result / Y.shape[2]

#Define Connection Probability Matrix
B_dict_equal = {}
eps2 = 0.1
for k in range(K):
    for t in range(T):
        B_dict_equal[(k,t)] = np.array([[0.25 + eta * k, 0.1 + np.sin(2 * np.pi * t / T) * eps2], [0.1 + np.sin(2 * np.pi * t / T) * eps2, 0.25]])


# Calculate P-values
p_values = np.zeros(num_replicates)


for i in range(num_replicates):
    A_dict, z, z_prime = simulate_dmpsbm(n=n, B_dict=B_dict_equal)
    Xhat, Yhat = dmprdpg.duase(A_dict, K=K, T=T, d=d)
    boot = bootstrap(Xhat, Yhat, n_samples=num_boot, K=K, T=T, d=d)
    observed = test_statistic_tensor(Xhat)
    p_values[i] = (np.sum(boot > observed) + 1) / (len(boot) + 1)

np.savetxt(input_folder + f"/bootstrap_n={n}_eta={eta}.csv", p_values, delimiter=",")