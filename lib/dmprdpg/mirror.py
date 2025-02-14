#!/usr/bin/env python3
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall, connected_components
from scipy.spatial import distance_matrix
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import MDS, Isomap
from sklearn.manifold import Isomap
import itertools
from tqdm import tqdm

## Get in input a matrix of size (n,d,K) and return a matrix of size (K,K) with the Euclidean distance
def distance_matrix_three_tensor(Y, ord=2):
    n, _, K = Y.shape
    D = np.zeros((K,K))
    for i in range(K):
        for j in range(i+1,K):
            D[i,j] = np.linalg.norm(Y[:,:,i] - Y[:,:,j], ord=ord) / np.sqrt(n)
            D[j,i] = D[i,j]
    return D

## Function to generate cross pairs
def cross_pairs(A, B):
    return list(itertools.product(range(A), range(B)))

## Get in input a matrix of size (n,d,K,T) and return a matrix of size (KT,KT) with the Euclidean distance
def distance_matrix_four_tensor(Y, ord=2, verbose=True):
    n, _, K, T = Y.shape
    D = np.zeros((K * T, K * T))
    ## Calculate all cross pairs of K and T
    pairs = cross_pairs(K, T)
    ## Pre-define the distance matrix
    D = np.zeros((K * T, K * T))
    ## Loop over all pairs and calculate the distance
    for q, pair in tqdm(enumerate(pairs)) if verbose else enumerate(pairs):
        k, t = pair[0], pair[1]
        for q_prime, pair_prime in enumerate(pairs):
            if q_prime > q:
                k_prime, t_prime = pair_prime[0], pair_prime[1]
                ## Calculate the distance
                D[q, q_prime] = np.linalg.norm(Y[:,:,k,t] - Y[:,:,k_prime,t_prime], ord=ord) / np.sqrt(n)
                D[q_prime, q] = D[q, q_prime]
    ## Return the distance matrix
    return D

## Apply classic multidimensional scaling to the distance matrix
def cmds(D, n_components=2, square=True):
    K = D.shape[0]
    H = np.eye(K) - np.ones((K,K)) / K
    B = -0.5 * ((H @ (D ** (2 if square else 1))) @ H)
    eigvals, eigvecs = np.linalg.eigh(B)
    return (eigvecs[:,::-1][:,:n_components]) @ np.diag(np.sqrt(eigvals[::-1][:n_components]))

# Function to check if the graph is connected
def is_connected(graph):
    n_components, _ = connected_components(csgraph=graph, directed=False, return_labels=True)
    return n_components == 1

## Function to calculate ISO-MAP
def isomap_custom(X, n_neighbors=None, n_components=1, verbose=False):
    """ Perform ISOMAP on dataset X.
    Parameters:
    X : ndarray of shape (n_samples, n_features)
    n_neighbors : int, default None (the number of neighbors to consider for each point)
    n_components : int, default 1 (The number of dimensions in which to embed the dataset)
    verbose : bool, default False (Whether to print the number of neighbors)
    """
    # Compute the full pairwise distance matrix and number of neighbours (if not provided)
    D = distance_matrix(X, X) ## / np.sqrt(X.shape[1])
    n = D.shape[0]
    # If n_neighbors is not specified, find the smallest number of neighbors
    if n_neighbors is None:
        n_neighbors = 1
        while True:
            knn_distances = np.sort(D, axis=1)[:, 1:n_neighbors+1]
            knn_indices = np.argsort(D, axis=1)[:, 1:n_neighbors+1]
            # Construct the neighborhood graph (sparse matrix)
            graph = np.zeros((n, n))
            for i in range(n):
                graph[i, knn_indices[i]] = knn_distances[i]
            # Symmetrise the graph
            graph = np.minimum(graph, graph.T)
            # Check if the graph is connected
            sparse_graph = csr_matrix(graph)
            if is_connected(sparse_graph):
                break
            n_neighbors += 1
        if verbose: 
            print(f"ISOMAP n_neighbors = {n_neighbors}")
    else:
        knn_distances = np.sort(D, axis=1)[:, 1:n_neighbors+1]
        knn_indices = np.argsort(D, axis=1)[:, 1:n_neighbors+1]
        # Construct the neighborhood graph (sparse matrix)
        graph = np.zeros((n, n))
        for i in range(n):
            graph[i, knn_indices[i]] = knn_distances[i]
        # Symmetrise the graph
        graph = np.minimum(graph, graph.T)
    # Compute the shortest paths via Floyd-Warshall
    distances = floyd_warshall(graph, directed=False)
    # Apply MDS to the distance matrix
    Y = cmds(distances, n_components=n_components)
    # Return the embedding
    return Y

## Function to calculate ISO-MAP based on sklearn
def isomap(X, n_neighbors=None, n_components=1):
    ## Create the object
    iso = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    ## Fit to the data
    Y = iso.fit_transform(X)
    ## Return the embedding
    return Y

## Full procedure to obtain the mirror
def mirror(Y, n_neighbors=None, n_components_cmds=2, n_components_isomap=1, verbose=True, ord='fro', custom=False):
    ## Calculate distance matrix
    if Y.ndim == 3:
        D = distance_matrix_three_tensor(Y, ord=ord)
    elif Y.ndim == 4:
        D = distance_matrix_four_tensor(Y, ord=ord)
    else:
        raise ValueError("The input tensor must be of shape (n,d,K) or (n,d,K,T)")
    ## Apply classic multidimensional scaling
    U = cmds(D, n_components=n_components_cmds)
    ## Return ISOMAP
    if custom:
        return isomap_custom(U, n_neighbors=n_neighbors, n_components=n_components_isomap, verbose=verbose)
    else:
        return isomap(U, n_neighbors=n_neighbors, n_components=n_components_isomap)

## Reshape output of CMDS if needed
def reshape_cmds(U, K, T):
    return U.reshape(K, T, -1)

## Reshape the mirror to the original shape
def reshape_mirror(M, K, T):
    return M.reshape(K, T)

## Calculate the distance matrix from separate embeddings in a dictionary
def distance_matrix_separate(Y, ord=2):
    ## Pre-define objects
    T = len(Y)
    D = np.zeros((T, T))
    ## Loop over all pairs of embeddings and calculate the distance
    for i in range(T):
        for j in range(i+1, T):
            ## Procrustes alignment of Y[j] onto Y[i]
            Y_tilde = Y[j] @ orthogonal_procrustes(Y[j], Y[i])[0]
            D[i,j] = np.linalg.norm(Y[i] - Y_tilde, ord=ord) / np.sqrt(Y[i].shape[0])
            D[j,i] = D[i,j]
    ## Return the distance matrix
    return D