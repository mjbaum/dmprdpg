import zipfile
import pickle
import dmprdpg
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
import os, glob, argparse
import pandas as pd
from collections import Counter
from scipy.stats import bernoulli
from numba import jit, prange

## PARSER to give parameter values
parser = argparse.ArgumentParser()
## Set destination folder for output
parser.add_argument("-f","--folder", type=str, dest="folder", default="simulation_1", const=True, nargs="?",\
    help="String: name of the folder for the input files.")

## Parse arguments
args = parser.parse_args()
input_folder = args.folder

## Load Clayton Data

# ZIP file names
data_directory = input_folder + '/data/clayton'
# Get all file names in the directory
zip_filenames = np.sort([f"{data_directory}/{f}" for f in os.listdir(data_directory) if f.endswith(".zip")])
#
labels = []
with zipfile.ZipFile(zip_filenames[['2' in zf for zf in zip_filenames]][0], 'r') as zip_ref:
    for file_name in zip_ref.namelist():
        if file_name.endswith(".pkl"):
            labels += [file_name.split('.')[0].split('/')[-1]]

names_to_labels = {}
labels_to_names = {}
for label, name in enumerate(labels):
    names_to_labels[name] = label
    labels_to_names[label] = name

# Define dictionary of adjacency matrices
A = {}
# Counter for the number of matrices
k = 0
#
k_labels = []
# Loop over all ZIP files
for zip_filename in zip_filenames:
    # Process ZIP file
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        # Iterate through the files in the ZIP
        for file_name in zip_ref.namelist():
            if file_name.endswith(".pkl"):  # Process only .pkl files
                # Read the file without extracting
                with zip_ref.open(file_name) as f:
                    # Read binary content of the .pkl file
                    data = pickle.load(f)
                # Extract file name
                if file_name.split('.')[0].split('/')[-1].split('_')[-1].isdigit():
                    fn = '_'.join(file_name.split('.')[0].split('/')[-1].split('_')[:-1])
                else:
                    fn = file_name.split('.')[0].split('/')[-1]
                # Extract the label
                k_labels += [names_to_labels[fn]]
                # Extract the adjacency matrix data
                adjacencyMatrix = data["adjacencyMatrix"]
                shape = adjacencyMatrix.shape
                # Verify the shape matches expectations
                if len(shape) == 3 and shape[1] == 140 and shape[2] == 140:
                    # Continue processing
                    pass
                else:
                    raise ValueError(f"Unexpected shape: {shape}, filename: {file_name} in {zip_filename}")
                # Iterate over the 160 slices
                for t in range(shape[0]):  # Iterate over the 160 slices
                    # Extract the t-th adjacency matrix as a 140x140 sparse matrix
                    A[(k,t)] = csr_matrix(adjacencyMatrix[t])
                # Increase the counter
                k += 1

@jit(nopython=True, parallel=True)
def generate_adjacency(P_mat, rep=1):
    n, m  = P_mat.shape[0], P_mat.shape[1]
    adj = np.zeros((n, m))
    for i in prange(n):
        for j in prange(m):
            adj[i,j] = np.random.binomial(rep, P_mat[i,j])/rep
    return adj

def bootstrap(left_embeddings, right_embeddings, n_samples, K=3, T=10, d=2, rep=1):
    if len(left_embeddings.shape) != 3:
        raise ValueError("Input must be a 3D tensor")
    ## Calculate the mean matrix on the third axis of the tensor
    Xbar = np.mean(left_embeddings, axis=2)
    Pmat_dict = {}
    for k in range(K):
        for t in range(T):
            Pmat_dict[(k,t)] = clean_Pmat(np.matmul(Xbar, np.transpose(right_embeddings[:,:,t])))
    empirical_dist = list()
    for _ in range(n_samples):
        adj_dict = {}
        for k in range(K):
            for t in range(T):
                adj_dict[(k,t)] = sparse.csr_matrix(generate_adjacency(Pmat_dict[(k,t)], rep=rep))
        Xboot, Yboot = dmprdpg.duase(adj_dict, K=K, T=T, d=d)
        bootstrap_value = test_statistic_tensor(Xboot)
        empirical_dist.append(bootstrap_value)
    return empirical_dist

@jit(nopython=True, parallel=True)
def clean_Pmat(P_mat):
    n = P_mat.shape[0]
    for i in prange(n):
        for j in prange(n):
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

for layer in range(13):
    results = pd.DataFrame(columns=['Observed', 'Bootstrap'])
    A_group = {}
    counter = 0
    for i in range(len(k_labels)):
        if k_labels[i] == layer:
            for j in range(160):
                A_group[(counter,j)] = A[(i,j)]
            counter += 1

    A_long = dmprdpg.double_unfolding(A_group, rows=11, cols=160, n=140)
    U, S, Vh = dmprdpg.sparse_svd(A_long, 100)
    dim=dmprdpg.zhu(S)[1]

    Xhat, Yhat = dmprdpg.duase(A_group, K=11, T=160, d=dim)

    boot = bootstrap(Xhat, Yhat, n_samples=1000, K=11, T=160, d=dim)
    results['Bootstrap'] = boot
    results['Observed'] = test_statistic_tensor(Xhat)
    results.to_csv(input_folder+f"/Test_Within_Layer_{layer+1}.csv")