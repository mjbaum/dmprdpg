#!/usr/bin/env python3
import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

## Define a function to perform the double unfolding into a block matrix
def double_unfolding(matrix_dict, rows, cols, n, output='sparse'):
    if len(matrix_dict) != rows * cols:
        raise ValueError("Number of matrices in the dictionary must match rows*cols")
    # Create a list of lists for rows and columns
    matrix_grid = [[None for _ in range(cols)] for _ in range(rows)]
    # Check if all row and column indices are in the correct range
    if not all(row_idx in range(rows) and col_idx in range(cols) for row_idx, col_idx in matrix_dict.keys()):
        raise ValueError("Row and column indices must be in the range [0, rows) and [0, cols) respectively")
    # Check if all indices are covered
    if not all((row_idx, col_idx) in matrix_dict for row_idx in range(rows) for col_idx in range(cols)):
        raise ValueError("Matrix dictionary must contain all indices in the range [0, rows) x [0, cols)")
    # Fill the grid with matrices from the dictionary
    for (row_idx, col_idx), matrix in matrix_dict.items():
        if matrix.shape != (n, n):
            raise ValueError("All matrices must have the same dimension.")
        if output == 'sparse' and not isinstance(matrix, csr_matrix):
            matrix = coo_matrix(matrix)  # Convert to sparse format if not already
        matrix_grid[row_idx][col_idx] = matrix
    # Create the block matrix
    if output == 'sparse':
        # Use sparse block matrix assembly
        A_tilde = bmat(matrix_grid, format='csr')
    else:
        # Convert to dense if necessary
        A_tilde = np.bmat(matrix_grid).A  # .A converts to a dense numpy array
    # Return output
    return A_tilde

## Inverse of the previous function, decomposes a doubly unfolded block matrix into a dictionary of submatrices
def inverse_double_unfolding(A_tilde, n, K, T, output='sparse'):
    if A_tilde.shape != (n * K, n * T):
        raise ValueError("Large matrix dimensions do not match nK x nT")
    # Convert to CSR for efficient slicing, COO does not support slicing
    if isinstance(A_tilde, coo_matrix):
        A_tilde = A_tilde.tocsr()
    # Create a dictionary to store the submatrices
    matrix_dict = {}
    # Iterate over the blocks and store them in the dictionary
    for i in range(K):
        for j in range(T):
            submatrix = A_tilde[i*n:(i+1)*n, j*n:(j+1)*n]
            if output == 'dense':
                submatrix = submatrix.toarray()  # Convert to dense array if needed
            elif output == 'sparse':
                submatrix = csr_matrix(submatrix)  # Ensure it stays in sparse format
            matrix_dict[(i, j)] = submatrix
    # Return the dictionary of submatrices
    return matrix_dict

## Function to perform the truncated SVD of a sparse matrix
def sparse_svd(A_tilde, d):
    # Check if the matrix is in an appropriate sparse format or convert it
    if not isinstance(A_tilde, csr_matrix):
        A_tilde = csr_matrix(A_tilde)  # Convert to CSR format if not already
    # Compute the truncated SVD
    U, S, Vt = svds(A_tilde, k=d)
    # Sort the singular values (and corresponding singular vectors) in descending order and return output
    return U[:,::-1], S[::-1], Vt[::-1].T

## Obtain embeddings from the singular value decomposition
def get_embeddings(U, S, V):
    return U @ np.diag(np.sqrt(S), V @ np.diag(np.sqrt(S)))

## Unstack embeddings
def extract_and_concatenate(matrix, n, K):
    """
    Extracts K matrices of size nxd from an nKxd matrix and concatenates them into a tensor along the third dimension.
    Args:
    matrix (numpy array): The input matrix of size nKxd.
    n (int): The number of rows in each block, leading to matrices of size nxd.
    K (int): The number of blocks to extract and concatenate.
    Returns:
    tensor (numpy array): The resulting tensor of shape n x d x K.
    """
    if matrix.shape[0] != n * K:
        raise ValueError("The number of rows in the matrix must be n * K")
    # Initialize an empty list to store the matrices
    matrices = []
    # Extract each block of n rows
    for i in range(K):
        start_row = i * n
        end_row = start_row + n
        block = matrix[start_row:end_row, :]
        matrices.append(block)
    # Concatenate the list of matrices into a tensor along the third dimension
    tensor = np.stack(matrices, axis=2)
    # Return the resulting tensor
    return tensor

## Doubly unfolded adjacency spectral embedding (DUASE)
def duase(A_dict, K, T, d):
    # Get the size of the matrices
    n = A_dict[(0,0)].shape[0]
    # Perform the double unfolding
    A_tilde = double_unfolding(A_dict, K, T, n, output='sparse')
    # Perform the truncated SVD
    U, S, V = sparse_svd(A_tilde, d)
    # Obtain the embeddings
    X, Y = get_embeddings(U, S, V)
    # Reshape X and Y using the extract_and_concatenate function
    X = extract_and_concatenate(X, n, K)
    Y = extract_and_concatenate(Y, n, T)
    # Return the DUASE left and right embeddings
    return X, Y