#!/usr/bin/env python3
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall, connected_components
from scipy.spatial import distance_matrix
from scipy.linalg import orthogonal_procrustes
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

## Get in input a matrix of size (n,d,K,T) and return a matrix of size (K,K) or (T,T) with the Euclidean distance
def distance_matrix_collapsed_tensor(Y, ord=2, collapse='T'):
    n, _, K, T = Y.shape
    ## Collapse can only be 'K' or 'T'
    if collapse not in ['K', 'T']:
        raise ValueError("The collapse parameter must be 'K' or 'T' (in text).")
    if collapse == 'T':
        D = np.zeros((K,K))
        for i in range(K-1):
            for j in range(i+1,K):
                for t in range(T):
                    D[i,j] += np.linalg.norm(Y[:,:,i,t] - Y[:,:,j,t], ord=ord) / np.sqrt(n)
                D[i,j] /= T 
                D[j,i] = D[i,j]
    else:
        D = np.zeros((T,T))
        for i in range(T-1):
            for j in range(i+1,T):
                for k in range(K):
                    D[i,j] += np.linalg.norm(Y[:,:,k,i] - Y[:,:,k,j], ord=ord) / np.sqrt(n)
                D[i,j] /= K
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
    # Compute the full pairwise distance matrix and number of neighbors (if not provided)
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
def mirror(Y, n_neighbors=None, n_components_cmds=2, n_components_isomap=1, verbose=True, ord='fro', custom=False, return_full=False):
    ## Calculate distance matrix
    if Y.ndim == 3:
        D = distance_matrix_three_tensor(Y, ord=ord)
    elif Y.ndim == 4:
        D = distance_matrix_four_tensor(Y, ord=ord, verbose=verbose)
    else:
        raise ValueError("The input tensor must be of shape (n,d,K) or (n,d,K,T)")
    ## Apply classic multidimensional scaling
    U = cmds(D, n_components=n_components_cmds)
    ## Return ISOMAP
    if return_full:
        res = {}
        res['U'] = U
        if custom:
            res['phi'] = isomap_custom(U, n_neighbors=n_neighbors, n_components=n_components_isomap, verbose=verbose)
        else:
            res['phi'] = isomap(U, n_neighbors=n_neighbors, n_components=n_components_isomap)
        res['D'] = D
        return res
    else:
        if custom:
            return isomap_custom(U, n_neighbors=n_neighbors, n_components=n_components_isomap, verbose=verbose)
        else:
            return isomap(U, n_neighbors=n_neighbors, n_components=n_components_isomap)

## Distance matrix from output (res) of the mirror function with return_full=True
def distance_matrix_DUASE(res, K, T):
    U_tilde = res['U'].reshape((K, T, res['U'].shape[1]), order='C')
    ## Calculate the average of distances for each t for every 
    L = np.zeros((K, K))
    for k in range(K-1):
        for l in range(k+1, K):
            ## Frobenius norm between U_tilde[k] and U_tilde[l]
            L[k,l] = np.linalg.norm(U_tilde[k] - U_tilde[l], 'fro') / np.sqrt(T)
            L[l,k] = L[k,l]
    return L

## Distance matrix from output (res) of the mirror function with return_full=True
def distance_matrix_scarf(res, K, T, collapse='T', ord='fro', matrix_norm=True):
    U_tilde = res['U'].reshape((K, T, res['U'].shape[1]), order='C')
    ## Collapse can only be 'K' or 'T'
    if collapse not in ['K', 'T']:
        raise ValueError("The collapse parameter must be 'K' or 'T' (in text).")
    ## Calculate distances
    if collapse == 'T':
        L = np.zeros((K, K))
    else:
        L = np.zeros((T, T))
    ## Calculate distance matrix
    if collapse == 'T':
        for k in range(K-1):
            for l in range(k+1, K):
                if matrix_norm:
                    L[k,l] = np.linalg.norm(U_tilde[k] - U_tilde[l], ord=ord) / np.sqrt(T)
                else:
                    for t in range(T):
                        ## Frobenius norm between U_tilde[k] and U_tilde[l]
                        L[k,l] += np.linalg.norm(U_tilde[k,t] - U_tilde[l,t])
                    L[k,l] /= T
                L[l,k] = L[k,l]
    else:
        for t in range(T-1):
            for s in range(t+1, T):
                if matrix_norm:
                    L[t,s] = np.linalg.norm(U_tilde[k] - U_tilde[l], ord=ord) / np.sqrt(K)
                else:
                    for k in range(K):
                        L[t,s] += np.linalg.norm(U_tilde[k,t] - U_tilde[k,s])
                    L[t,s] /= K
                L[s,t] = L[t,s]
    return L

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
    for i in range(T-1):
        for j in range(i+1, T):
            ## Procrustes alignment of Y[j] onto Y[i]
            Y_tilde = Y[j] @ orthogonal_procrustes(Y[j], Y[i])[0]
            D[i,j] = np.linalg.norm(Y[i] - Y_tilde, ord=ord) / np.sqrt(Y[i].shape[0])
            D[j,i] = D[i,j]
    ## Return the distance matrix
    return D

## Align embeddings calculated separately (receiving as input a dictionary with double indices)
from scipy.linalg import orthogonal_procrustes
def align_embeddings(X):
    rows = len(X)
    cols = len(X[0])
    ## Check if cols is identical for all matrices
    if not all(len(X[i]) == cols for i in range(rows)):
        raise ValueError("All matrices must have the same number of associated time points.")
    ## Number of nodes
    n = X[0][0].shape[0]
    d = X[0][0].shape[1]
    ## Check if all matrices have the same dimension
    if not all(X[i][j].shape == (n,d) for i in range(rows) for j in range(cols)):
        raise ValueError("All matrices must have the same dimension.")
    ## Distance matrix
    D = {}
    for i in range(rows):
        D[i] = np.zeros((cols,cols))
        for j in range(cols-1):
            for j_prime in range(j+1,cols):
                R, _ = orthogonal_procrustes(X[i][j_prime], X[i][j])
                D[i][j,j_prime] = np.linalg.norm(X[i][j]- X[i][j_prime] @ R , ord='fro') / np.sqrt(n)
                D[i][j_prime,j] = D[i][j,j_prime]
    return D

## Calculate the mirror from separate embeddings in a dictionary
def marginal_mirror(Y, n_components_cmds=2, n_components_isomap=1, ord='fro', n_neighbors=None, verbose=True, custom=False, 
                    calculate_distance_matrix=True, calculate_iso_mirror=True):
    ## Calculate distance matrix
    D = align_embeddings(Y)
    K = len(D)
    ## Check that all matrices have the same shape
    T = D[0].shape[0]
    if not all(D[i].shape[0] == T for i in range(K)):
        raise ValueError("All matrices must have the same number of associated time points.")
    ## Apply classic multidimensional scaling to each distance matrix
    M = {}; phi = {}
    for k in range(K):
        M[k] = cmds(D[k], n_components=n_components_cmds)
        ## Calculate the iso-mirror
        if calculate_iso_mirror:
            if custom:
                phi[k] = isomap_custom(M[k], n_neighbors=n_neighbors, n_components=n_components_isomap, verbose=verbose)
            else:
                phi[k] = isomap(M[k], n_neighbors=n_neighbors, n_components=n_components_isomap)
    ## Calculate distance matrix from separate M[k]'s after Procrustes alignment
    D_star = np.zeros((K,K))
    if calculate_distance_matrix:
        for i in range(K-1):
            for j in range(i+1, K):
                R, _ = orthogonal_procrustes(M[j], M[i])
                D_star[i,j] = np.linalg.norm(M[i] - M[j] @ R, ord=ord) / np.sqrt(T)
                D_star[j,i] = D_star[i,j]
    ## Return output
    return {'M': M, 'phi': phi, 'D_star': D_star}