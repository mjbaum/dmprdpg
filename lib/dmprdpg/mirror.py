#!/usr/bin/env python3
import numpy as np

## Get in input a matrix of size (n,d,K) and return a matrix of size (K,K) with the Euclidean distance
def distance_matrix(Y):
    n, d, K = Y.shape
    D = np.zeros((K,K))
    for i in range(K):
        for j in range(i+1,K):
            D[i,j] = np.linalg.norm(Y[:,:,i]-Y[:,:,j]) / np.sqrt(n)
            D[j,i] = D[i,j]
    return D

## Apply classic multidimensional scaling to the distance matrix
def cmds(D):
    K = D.shape[0]
    H = np.eye(K) - np.ones((K,K)) / K
    B = -0.5 * H @ D @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    return eigvecs[:,-2:] @ np.diag(np.sqrt(eigvals[-2:]))