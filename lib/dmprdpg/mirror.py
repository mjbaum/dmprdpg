#!/usr/bin/env python3
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.spatial import distance_matrix

## Get in input a matrix of size (n,d,K) and return a matrix of size (K,K) with the Euclidean distance
def distance_matrix_tensor(Y):
    n, _, K = Y.shape
    D = np.zeros((K,K))
    for i in range(K):
        for j in range(i+1,K):
            D[
                i,j] = np.linalg.norm(Y[:,:,i] - Y[:,:,j]) / np.sqrt(n)
            D[j,i] = D[i,j]
    return D

## Apply classic multidimensional scaling to the distance matrix
def cmds(D, n_components=2):
    K = D.shape[0]
    H = np.eye(K) - np.ones((K,K)) / K
    B = -0.5 * H @ (D ** 2) @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    return eigvecs[:,::-1][:,:n_components] @ np.diag(np.sqrt(eigvals[::-1][:n_components]))

## Function to calculate ISO-MAP
def isomap(X, n_neighbors, n_components=1):
    """ Perform ISOMAP on dataset X.
    Parameters:
    X : ndarray of shape (n_samples, n_features)
    n_neighbors : int (the number of neighbors to consider for each point)
    n_components : int, default 1 (The number of dimensions in which to embed the dataset)
    """
    # Step 1: Compute the full pairwise distance matrix
    D = distance_matrix(X, X)
    # Step 2: Find the k-nearest neighbors for each point
    knn_distances = np.sort(D, axis=1)[:, 1:n_neighbors+1]
    knn_indices = np.argsort(D, axis=1)[:, 1:n_neighbors+1]
    # Step 3: Construct the neighborhood graph (sparse matrix)
    n = X.shape[0]
    graph = np.inf * np.ones((n, n))
    for i in range(n):
        graph[i, knn_indices[i]] = knn_distances[i]
    # Symmetrize the graph
    graph = np.minimum(graph, graph.T)
    # Step 4: Compute the shortest paths
    graph = csr_matrix(graph)
    distances = floyd_warshall(graph, directed=False)
    # Step 5: Apply classical MDS
    Y = cmds(distances, n_components=n_components)
    # Return the embedding
    return Y

## Full procedure to obtain the mirror
def mirror(Y, n_neighbors, n_components_cmds=2, n_components_isomap=1):
    ## Calculate distance matrix
    D = distance_matrix_tensor(Y)
    ## Apply classic multidimensional scaling
    U = cmds(D, n_components=n_components_cmds)
    ## Return ISOMAP
    return isomap(U, n_neighbors=n_neighbors, n_components=n_components_isomap)