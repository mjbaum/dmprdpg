#!/usr/bin/env python3
import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix
from scipy.sparse.linalg import svds, eigsh
from scipy.stats import norm


## Long unfolding (totem, scarf, or blanket)
def double_unfolding(matrix_dict, rows, cols, n, mode='blanket', lead_order='row', output='sparse'):
    if len(matrix_dict) != rows * cols:
        raise ValueError("Number of matrices in the dictionary must match rows*cols")
    # Create a list of lists for rows and columns
    if mode == 'blanket':
        matrix_grid = [[None for _ in range(cols)] for _ in range(rows)]
    elif mode == 'totem':
        matrix_grid = [[None] for _ in range(rows * cols)]
    elif mode == 'scarf':
        matrix_grid = [[None for _ in range(rows * cols)]]
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
        if mode == 'blanket':
            matrix_grid[row_idx][col_idx] = matrix
        elif mode == 'totem':
            if lead_order == 'row':
                matrix_grid[row_idx * cols + col_idx][0] = matrix
            else: 
                matrix_grid[row_idx + col_idx * rows][0] = matrix
        elif mode == 'scarf':
            if lead_order == 'row':
                matrix_grid[0][row_idx * cols + col_idx] = matrix
            else:
                matrix_grid[0][row_idx + col_idx * rows] = matrix
    # Create the block matrix
    if output == 'sparse':
        # Use sparse block matrix assembly via bmat
        A_tilde = bmat(matrix_grid, format='csr')
    else:
        # Convert to dense if necessary
        A_tilde = np.bmat(matrix_grid).A  # .A converts to a dense numpy array
    # Return output
    return A_tilde

## Average all matrices in matrix_dict to create a single matrix for each value of t
def average_matrices(matrix_dict, rows, cols, n, include_zero=True):
    if len(matrix_dict) != rows * cols:
        raise ValueError("Number of matrices in the dictionary must match rows*cols")
    # Check if all row and column indices are in the correct range
    if not all(row_idx in range(rows) and col_idx in range(cols) for row_idx, col_idx in matrix_dict.keys()):
        raise ValueError("Row and column indices must be in the range [0, rows) and [0, cols) respectively")
    # Check if all indices are covered
    if not all((row_idx, col_idx) in matrix_dict for row_idx in range(rows) for col_idx in range(cols)):
        raise ValueError("Matrix dictionary must contain all indices in the range [0, rows) x [0, cols)")
    # Check if all matrices have the same dimension
    for (row_idx, col_idx), matrix in matrix_dict.items():
        if matrix.shape != (n, n):
            raise ValueError("All matrices must have the same dimension.")
    # Pre-define smaller dictionary of matrices for each value of t
    matrix_dict_t = {}
    # Iterate over the columns to average the matrices
    for col_idx in range(cols):
        # Extract the matrices for the current value of t
        matrices_t = [matrix_dict[(row_idx, col_idx)] for row_idx in range(rows)]
        # Average the matrices
        matrix_t = sum(matrices_t) / rows
        # Store the matrix in the dictionary
        matrix_dict_t[(0,col_idx) if include_zero else col_idx] = matrix_t
    # Return the average matrices
    return matrix_dict_t

## Inverse of the previous function, decomposes a doubly unfolded block matrix into a dictionary of submatrices
def inverse_double_unfolding(A_tilde, n, K, T, lead_order='row', output='sparse'):
    if A_tilde.shape == (n * K, n * T):
        mode = 'blanket'
    elif A_tilde.shape == (n * K * T, n):
        mode = 'totem'
    elif A_tilde.shape == (n, n * K * T):
        mode = 'scarf'
    else:
        raise ValueError("Large matrix dimensions do not match nK x nT, nKT x n, or n x nKT. Please check the dimensions.")
    # Convert to CSR for efficient slicing, COO does not support slicing
    if isinstance(A_tilde, coo_matrix):
        A_tilde = A_tilde.tocsr()
    # Create a dictionary to store the submatrices
    matrix_dict = {}
    # Iterate over the blocks and store them in the dictionary
    for i in range(K):
        for j in range(T):
            # Extract the correct submatrix A(k,t) based on the mode and lead_order
            if mode == 'blanket':
                submatrix = A_tilde[i*n:(i+1)*n, j*n:(j+1)*n]
            elif mode == 'totem':
                if lead_order == 'row':
                    submatrix = A_tilde[(i*T*n + j*n):(i*T*n + (j+1)*n)]
                else:
                    submatrix = A_tilde[(j*K*n + i*n):(j*K*n + (i+1)*n)]
            elif mode == 'scarf':
                if lead_order == 'row':
                    submatrix = A_tilde[:, (i*T*n + j*n):(i*T*n + (j+1)*n)]
                else:
                    submatrix = A_tilde[:, (j*K*n + i*n):(j*K*n + (i+1)*n)]
            # Convert to dense array if needed
            if output == 'dense':
                submatrix = submatrix.toarray()  # Convert to dense array if needed
            elif output == 'sparse':
                submatrix = csr_matrix(submatrix)  # Ensure it stays in sparse format
            # Store the submatrix in the dictionary
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

## Function to perform the truncated eigendecomposition of a sparse matrix
def sparse_eigendecomp(A_tilde, d):
    # Check if the matrix is in an appropriate sparse format or convert it
    if not isinstance(A_tilde, csr_matrix):
        A_tilde = csr_matrix(A_tilde)  # Convert to CSR format if not already
    # Compute the truncated eigendecomposition
    S, U = eigsh(A_tilde, k=d, which='LA')
    # Sort the eigenvalues (and corresponding eigenvectors) in descending order and return output (including embeddings)
    return U[:,::-1], S[::-1], U[:,::-1] @ np.diag(np.sqrt(S[::-1]))

## Obtain embeddings from the singular value decomposition
def get_embeddings(U, S, V):
    return U @ np.diag(np.sqrt(S)), V @ np.diag(np.sqrt(S))

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

## Unstack embeddings to obtain a four-dimensional tensor
def extract_tensor(matrix, n, K, T, order='row'):
    """
    Extracts a four-dimensional tensor from a matrix of size nKT x d.
    Args:
    matrix (numpy array): The input matrix of size nK x nT.
    n (int): The number of rows in each block, leading to matrices of size n x n.
    K (int): The number of rows in the tensor
    T (int): The number of columns in the tensor
    order (str): The order in which the blocks are extracted, either 'row' or 'column'
    Returns:
    tensor (numpy array): The resulting tensor of shape n x n x K x T.
    """
    # Check if order is either 'row' or 'column'
    if order not in ['row', 'column']:
        raise ValueError("The order must be either 'row' or 'column'")
    # Check if the matrix dimensions match the tensor dimensions
    if matrix.shape[0] != (n * K * T):
        raise ValueError("The matrix dimensions do not match nKT")
    # Initialize an empty list to store the matrices
    tensor = np.zeros((n, matrix.shape[1], K, T))
    # Extract each block of size n x n
    for k in range(K):
        for t in range(T):
            if order == 'row':
                start = (k * T + t) * n
                end = start + n
            else:
                start = (k + t * K) * n
                end = start + n
            block = matrix[start:end]
            tensor[:, :, k, t] = block
    # Return the resulting tensor
    return tensor

## Calculate elbow of the scree-plot using the criterion of Zhu and Ghodsi (2006)
def zhu(d):
    d = np.sort(d)[::-1]
    p = len(d)
    profile_likelihood = np.zeros(p)
    for q in range(1,p-1):
        mu1 = np.mean(d[:q])
        mu2 = np.mean(d[q:])
        sd = np.sqrt(((q-1) * (np.std(d[:q]) ** 2) + (p-q-1) * (np.std(d[q:]) ** 2)) / (p-2))
        profile_likelihood[q] = norm.logpdf(d[:q], loc=mu1, scale=sd).sum() + norm.logpdf(d[q:], loc=mu2, scale=sd).sum()
    return profile_likelihood[1:(p-1)], np.argmax(profile_likelihood[1:(p-1)]) + 1

## Find the first x elbows of the scree-plot, iterating the criterion of Zhu and Ghodsi (2006)
def iterate_zhu(d, x=4):
    d = np.sort(d)[::-1]
    results = np.zeros(x,dtype=int)
    results[0] = zhu(d)[1]
    for i in range(x-1):
        results[i+1] = results[i] + zhu(d[results[i]:])[1]
    return results

## Eigengap
def eigengap(S, x=4):
    Q = np.argsort(np.diff(np.sort(S)[::-1]))
    # Get the first 4 numbers of Q, but discard a value if it's smaller than *any* value that came before it
    Q_mod = np.zeros(len(Q), dtype=int)
    Q_mod[0] = Q[0]
    for i in range(1, len(Q)):
        Q_mod[i] = 0 if Q[i] < np.max(Q[:i]) else Q[i]
    ## Remove zeros from Q_mod
    Q_mod = Q_mod[Q_mod != 0]
    ## Return indices for the first x eigengaps
    return Q_mod[:x] + 1

## Visualise singular values of A_tilde
def singular_values_A_tilde(A_dict, K, T, mode='blanket', lead_order='row', d_max=None):
    n = A_dict[(0,0)].shape[0]
    A_tilde = double_unfolding(A_dict, K, T, n, mode=mode, lead_order=lead_order, output='sparse')
    if d_max is None:
        if mode == 'blanket':
            d_max = n * min([K,T]) - 1
        else:
            d_max = n - 1
    else:
        ## Check suitability of d_max (integer)
        if not isinstance(d_max, int):
            raise ValueError("d_max must be an integer")
    _, S, _ = sparse_svd(A_tilde, d_max)
    return S

## Visualise eigenvalues of A
def eigenvalues_A(A, d_max=None):
    # Check A is suitable
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)
    # Check d_max is suitable
    if d_max is None:
        d_max = min(A.shape) - 1
    else:
        if not isinstance(d_max, int):
            raise ValueError("d_max must be an integer")
    # Perform the eigendecomposition
    S, _ = eigsh(A, d_max)
    return S

## Doubly unfolded adjacency spectral embedding (DUASE)
def duase(A_dict, K, T, d=None, mode='blanket', lead_order='row', d_selection='zhu', order=1, d_max=None, verbose=False):
    # Check if mode is admissible
    if mode not in ['blanket', 'scarf', 'totem']:
        return ValueError("Mode must be either 'blanket', 'scarf', or 'totem'")
    # Check if lead_order is admissible
    if lead_order not in ['row', 'column']:
        return ValueError("Lead order must be either 'row' or 'column'")
    # Check if d_selection is admissible
    if d_selection not in ['zhu', 'eigengap']:
        return ValueError("d_selection must be either 'zhu' or 'eigengap'")
    # Get the size of the matrices
    n = A_dict[(0,0)].shape[0]
    # Perform the double unfolding
    A_tilde = double_unfolding(A_dict, K, T, n, mode=mode, lead_order=lead_order, output='sparse')
    # Check if d_max is specified and admissible
    if d_max is None:
        if mode == 'blanket':
            d_max = n * min([K,T]) - 1
        else:
            d_max = n - 1
    else:
        ## Check suitability of d_max (integer)
        if not isinstance(d_max, int):
            raise ValueError("d_max must be an integer")
    # If the number of components is not specified, use the Zhu and Ghodsi criterion
    if d is None:
        if d_selection == 'eigengap':
            _, S, _ = svds(A_tilde, k=d_max)
            d = eigengap(S, x=order)[-1]
        elif d_selection == 'zhu':
            _, S, _ = svds(A_tilde, k=d_max)
            d = iterate_zhu(S, x=order)[-1]
        if verbose:
            print('Selected embedding dimension: ', d)
    # Perform the truncated SVD
    U, S, V = sparse_svd(A_tilde, d)
    # Obtain the embeddings
    X, Y = get_embeddings(U, S, V)
    # Reshape X and Y using the extract_and_concatenate function
    if mode == 'blanket':
        X = extract_and_concatenate(X, n, K)
        Y = extract_and_concatenate(Y, n, T)
    elif mode == 'scarf':
        Y = extract_tensor(Y, n, K, T, order=lead_order)
    elif mode == 'totem':
        X = extract_tensor(X, n, K, T, order=lead_order)
    # Return the left and right embeddings
    return X, Y

## Calculate separate embeddings for each matrix in the dictionary
def separate_embeddings(matrix_dict, rows, cols, d=None, d_selection='zhu', order=1, d_max=None, verbose=False, directed=False, double_index=True):
    # Obtain n
    n = matrix_dict[(0,0)].shape[0]
    # Check if all row and column indices are in the correct range
    if not all(row_idx in range(rows) and col_idx in range(cols) for row_idx, col_idx in matrix_dict.keys()):
        raise ValueError("Row and column indices must be in the range [0, rows) and [0, cols) respectively")
    # Check if all indices are covered
    if not all((row_idx, col_idx) in matrix_dict for row_idx in range(rows) for col_idx in range(cols)):
        raise ValueError("Matrix dictionary must contain all indices in the range [0, rows) x [0, cols)")
    # Check if all matrices have the same dimension and are symmetric
    for (row_idx, col_idx), matrix in matrix_dict.items():
        if matrix.shape != (n, n):
            raise ValueError("All matrices must have the same dimension.")
    # Check if d_max is specified and admissible
    if d_max is None:
        d_max = n - 1
    else:
        ## Check suitability of d_max (integer)
        if not isinstance(d_max, int):
            raise ValueError("d_max must be an integer")
    # Initialize dictionaries to store the embeddings
    X_dict = {}
    if double_index:
        for i in range(rows):
            X_dict[i] = {}
    if directed:
        Y_dict = {}
        if double_index:
            for i in range(rows):
                Y_dict[i] = {}
    # Iterate over the matrices and calculate the embeddings
    for i in range(rows):
        for j in range(cols):
            # Perform the embedding for each matrix
            if directed: 
                U, S, V = sparse_svd(matrix_dict[(i,j)], d=d)
                X, Y = get_embeddings(U, S, V)
                # Store the embeddings in the dictionaries
                if double_index:
                    X_dict[i][j] = X
                    Y_dict[i][j] = Y
                else:
                    X_dict[(i,j)] = X
                    Y_dict[(i,j)] = Y
            else:
                # Perform the truncated eigendecomposition
                d_hat = d if d is not None else iterate_zhu(eigenvalues_A(matrix_dict[(i,j)], d_max), x=order)[-1]
                _, _, X = sparse_eigendecomp(matrix_dict[(i,j)], d=d_hat)
                # Store the embeddings in the dictionaries
                if double_index:
                    X_dict[i][j] = X
                else:
                    X_dict[(i,j)] = X
    # Return the dictionaries of embeddings
    if directed:
        return X_dict, Y_dict
    else:
        return X_dict