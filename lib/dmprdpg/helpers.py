from itertools import accumulate
from networkx import stochastic_block_model, adjacency_matrix
import random
import numpy as np
import pandas as pd

def generate_groups(num_nodes, len_groups, randomize = False):
    node_list = [i+1 for i in range(num_nodes)]
    if randomize:
        random.shuffle(node_list)
    output = [node_list[x - y: x] for x, y in zip(
        accumulate(len_groups), len_groups)]
    return output

def generate_group_labels(len_groups, randomize = False):
    output = []
    for i in range(len(len_groups)):
        output = output + [i]*len_groups[i]
    if randomize:
        random.shuffle(output)
    return output

def generate_adjacency_matrix(len_groups, probabilities):
    model = stochastic_block_model(sizes = len_groups, p=probabilities)
    matrix = adjacency_matrix(model)
    return matrix.toarray()

def get_embedding(A, dimension = 4, type = 'left'):
    decomp = np.linalg.svd(A, full_matrices = False)
    U = decomp.U
    D = decomp.S
    D_half = np.diag(np.sqrt(D))
    V = np.transpose(decomp.Vh)
    if type == 'right':
        right_embedding = V @ D_half
        return right_embedding[:,0:dimension]
    left_embedding = U @ D_half
    return left_embedding[:,0:dimension]

def group_by_label(matrix, labels):
    labels = np.array(labels)
    for label in set(labels):
        print(label)
        print(labels == label)
        print(matrix[labels == label])

## Function to test the library
def test():
    print("The library dmprdpg is working correctly.")

## Find indices of nearest neighbors
from scipy.spatial import distance_matrix
def find_nearest_neighbours(X, n_neighbors=1):
    ## Calculate the distance matrix from X
    D = distance_matrix(X, X)
    ## Get the indices of the nearest neighbors 
    return np.argsort(D, axis=1)[:, :(n_neighbors+1)]