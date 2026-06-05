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

def calculate_projections(matrix_dict, K, T, d):
    n = matrix_dict[(0,0)].shape[0]
    proj = np.zeros((n, n, K, T))
    for i in range (K):
        for j in range (T):
            U, S, V = dmprdpg.sparse_svd(matrix_dict[i,j], d=d)
            proj[:,:,i,j] = U @ np.diag(S) @ U.T
    return proj

def calulate_alt_test_statistic(matrix_dict, K, T, d):
    UhatT = np.zeros((n, n, T))
    proj = calculate_projections(matrix_dict, K, T, d)
    for i in range(T):
        UhatT[:,:,i] = (1/K)*np.sum(proj[:,:,:,i], axis=2)
    stat = 0
    for k in range(K):
        for t in range(T):
            stat += np.linalg.norm(proj[:,:,k,t] - UhatT[:,:,t])
    return stat

def calculate_Phat(matrix_dict, K, T, d):
    Phat = dict()
    for j in range (T):
        total = np.zeros((n,n))
        for i in range (K):
            U, S, V = dmprdpg.sparse_svd(matrix_dict[i,j], d=d)
            PhatKT = clean_Pmat(U@np.diag(S)@V.T)
            total += PhatKT
        avg = (1/K)* total
        for i in range(K):
            Phat[i,j] = avg
    return Phat

def bootstrap_alternative(Phat, K, T, d, n_samples):
    empirical_dist = list()
    for _ in range(n_samples):
        adj_dict = {}
        for k in range(K):
            for t in range(T):
                adj_dict[k,t] = sparse.csr_matrix(generate_adjacency(Phat[k,t]))
        bootstrap_value = calulate_alt_test_statistic(adj_dict, K, T, d)
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
    observed = calulate_alt_test_statistic(A_dict, K, T, d)
    Phat = calculate_Phat(A_dict, K, T, d)
    boot = bootstrap_alternative(Phat, K, T, d, num_boot)
    p_values[i] = (np.sum(boot > observed) + 1) / (len(boot) + 1)

np.savetxt(input_folder + f"/bootstrap_individual_modified_n={n}_eta={eta}.csv", p_values, delimiter=",")