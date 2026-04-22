import zipfile
import pickle
import dmprdpg
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
import os, glob, argparse
import pandas as pd
from numba import jit, prange

## PARSER to give parameter values
parser = argparse.ArgumentParser()
## Set destination folder for output
parser.add_argument("-f","--folder", type=str, dest="folder", default="simulation_1", const=True, nargs="?",\
    help="String: name of the folder for the input files.")

## Parse arguments
args = parser.parse_args()
input_folder = args.folder
n_samples=1000
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

A_avg = dict()
for layer in range(13):
    replicates = [a for a,b in enumerate(k_labels) if b == layer]
    for time in range(160):
        total = np.zeros((140,140))
        for index in replicates:
            total += A[(index, time)].todense()
        A_avg[(layer,time)] = total/len(replicates)


## Global Test
results = pd.DataFrame(columns=['Observed', 'Bootstrap'])
A_long = dmprdpg.double_unfolding(A_avg, rows=13, cols=160, n=140)
U, S, Vh = dmprdpg.sparse_svd(A_long, 100)
dim=dmprdpg.zhu(S)[1]

Xhat, Yhat = dmprdpg.duase(A_avg, K=13, T=160, d=dim)

boot = bootstrap(Xhat, Yhat, n_samples=n_samples, K=13, T=160, d=dim, rep=11)
results['Bootstrap'] = boot
results['Observed'] = test_statistic_tensor(Xhat)
results.to_csv(input_folder+f"/Test_global.csv")

## Pairwise Tests
rejected = pd.DataFrame(columns=['Layer', 'Rejections'])
rejected['Layer'] = [a+1 for a in range(13)]
rejected['Rejections'] = 0

for layer1 in range(13):
    for layer2 in range(layer1+1,13):
        results = pd.DataFrame(columns=['Observed', 'Bootstrap'])
        A_pair = dict()
        for time in range(160):
            A_pair[(0,time)] = A_avg[(layer1,time)]
            A_pair[(1,time)] = A_avg[(layer2,time)]

        A_long = dmprdpg.double_unfolding(A_pair, rows=2, cols=160, n=140)
        U, S, Vh = dmprdpg.sparse_svd(A_long, 100)
        dim=dmprdpg.zhu(S)[1]

        Xhat, Yhat = dmprdpg.duase(A_pair, K=2, T=160, d=dim)

        boot = bootstrap(Xhat, Yhat, n_samples=n_samples, K=2, T=160, d=dim, rep=11)
        results['Bootstrap'] = boot
        results['Observed'] = test_statistic_tensor(Xhat)
        results.to_csv(input_folder + f"/Test_pairwise_{layer1}vs{layer2}.csv")

        observed = test_statistic_tensor(Xhat)
        p = (np.sum(boot > observed) + 1) / (len(boot) + 1)

        if p <= .001:
            rejected.loc[layer1, 'Rejections'] += 1
            rejected.loc[layer2, 'Rejections'] += 1

rejected.to_csv(input_folder + f"/Test_pairwise_rejections.csv")