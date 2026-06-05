import dmprdpg
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
import os, glob, argparse
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
        B_dict_equal[(k,t)] = np.array([[0.25 + eta * k, 0.6 + np.sin(2 * np.pi * t / T) * eps2], [0.6 + np.sin(2 * np.pi * t / T) * eps2, 0.25]])


# Calculate P-values
p_values = np.zeros(num_replicates)


for i in range(num_replicates):
    A_dict, z = dmprdpg.simulate_dmpsbm(n=n, B_dict=B_dict_equal, undirected=True, z_shared=True)
    Xhat, Yhat = dmprdpg.duase(A_dict, K=K, T=T, d=d)
    boot = bootstrap(Xhat, Yhat, n_samples=num_boot, K=K, T=T, d=d)
    observed = test_statistic_tensor(Xhat)
    p_values[i] = (np.sum(boot > observed) + 1) / (len(boot) + 1)

np.savetxt(input_folder + f"/bootstrap_n={n}_eta={eta}.csv", p_values, delimiter=",")