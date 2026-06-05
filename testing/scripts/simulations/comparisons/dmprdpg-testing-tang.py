from __future__ import annotations
import dmprdpg
import numpy as np
from scipy import sparse
from scipy.stats import cauchy
import argparse
from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray

# Initialize Static Parameters
K = 10
T = 3

## PARSER to give parameter values
parser = argparse.ArgumentParser()
## Set destination folder for output
parser.add_argument("-f","--folder", type=str, dest="folder", default="simulation_1", const=True, nargs="?",\
    help="String: name of the folder for the input files.")
parser.add_argument("-n", type=int, dest="n", default=100, const=True, nargs="?",\
	help="Integer: number of nodes in each graph. Default: n=100.")
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


@dataclass(frozen=True)
class GraphTwoSampleResult:
    """Result returned by :func:`rdpg_two_sample_pvalue`."""

    p_value: float
    p_value_from_a: float
    p_value_from_b: float
    statistic: float
    embedding_dim: int


def adjacency_spectral_embedding(
    adjacency: ArrayLike | sparse.spmatrix,
    embedding_dim: int,
) -> NDArray[np.float64]:

    adjacency = _validate_adjacency(adjacency, name="adjacency")
    n = adjacency.shape[0]
    if not 1 <= embedding_dim <= n:
        raise ValueError("embedding_dim must be between 1 and the number of vertices")

    eigenvalues, eigenvectors = np.linalg.eigh(adjacency)
    indices = np.argsort(np.abs(eigenvalues))[::-1][:embedding_dim]
    values = np.abs(eigenvalues[indices])
    vectors = eigenvectors[:, indices]
    return vectors * np.sqrt(values)


def rdpg_two_sample_pvalue(
    adjacency_a: ArrayLike | sparse.spmatrix,
    adjacency_b: ArrayLike | sparse.spmatrix,
    embedding_dim: int | None = None,
    n_bootstraps: int = 200,
    random_state: int | np.random.Generator | None = None,
) -> GraphTwoSampleResult:

    adjacency_a = _validate_adjacency(adjacency_a, name="adjacency_a")
    adjacency_b = _validate_adjacency(adjacency_b, name="adjacency_b")
    if adjacency_a.shape != adjacency_b.shape:
        raise ValueError("adjacency_a and adjacency_b must have the same shape")
    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive")

    rng = _as_generator(random_state)
    if embedding_dim is None:
        embedding_dim = _choose_embedding_dim(adjacency_a, adjacency_b)

    x_hat = adjacency_spectral_embedding(adjacency_a, embedding_dim)
    y_hat = adjacency_spectral_embedding(adjacency_b, embedding_dim)
    statistic = procrustes_distance(x_hat, y_hat)

    p_value_from_a = _bootstrap_pvalue(x_hat, statistic, n_bootstraps, rng)
    p_value_from_b = _bootstrap_pvalue(y_hat, statistic, n_bootstraps, rng)
    p_value = max(p_value_from_a, p_value_from_b)

    return GraphTwoSampleResult(
        p_value=float(p_value),
        p_value_from_a=float(p_value_from_a),
        p_value_from_b=float(p_value_from_b),
        statistic=float(statistic),
        embedding_dim=embedding_dim,
    )


def procrustes_distance(
    x: ArrayLike,
    y: ArrayLike,
) -> float:
    """Return ``min_W ||X - YW||_F`` over orthogonal matrices ``W``."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x and y must be two-dimensional arrays with the same shape")

    u, _, vt = np.linalg.svd(y.T @ x, full_matrices=False)
    w = u @ vt
    return float(np.linalg.norm(x - y @ w, ord="fro"))


def _bootstrap_pvalue(
    latent_positions: NDArray[np.float64],
    observed_statistic: float,
    n_bootstraps: int,
    rng: np.random.Generator,
) -> float:
    bootstrap_statistics = np.empty(n_bootstraps, dtype=float)
    embedding_dim = latent_positions.shape[1]

    for bootstrap_index in range(n_bootstraps):
        adjacency_x = _sample_rdpg(latent_positions, rng)
        adjacency_y = _sample_rdpg(latent_positions, rng)
        x_boot = adjacency_spectral_embedding(adjacency_x, embedding_dim)
        y_boot = adjacency_spectral_embedding(adjacency_y, embedding_dim)
        bootstrap_statistics[bootstrap_index] = procrustes_distance(x_boot, y_boot)

    exceedances = np.count_nonzero(bootstrap_statistics >= observed_statistic)
    return float(min(1-1e-3, (exceedances + 1) / (n_bootstraps + 1)))


def _sample_rdpg(
    latent_positions: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    probabilities = latent_positions @ latent_positions.T
    probabilities = np.clip(probabilities, 0.0, 1.0)
    np.fill_diagonal(probabilities, 0.0)

    n = probabilities.shape[0]
    uniforms = rng.random((n, n))
    upper = np.triu(uniforms < probabilities, k=1)
    adjacency = upper + upper.T
    return adjacency.astype(float, copy=False)


def _validate_adjacency(
    adjacency: ArrayLike | sparse.spmatrix,
    name: str,
) -> NDArray[np.float64]:
    if sparse.issparse(adjacency):
        matrix = adjacency.toarray().astype(float, copy=False)
    else:
        matrix = np.asarray(adjacency, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric")
    if not np.allclose(np.diag(matrix), 0):
        raise ValueError(f"{name} must have a zero diagonal")
    if np.any((matrix != 0) & (matrix != 1)):
        raise ValueError(f"{name} must be binary")
    return matrix


def _choose_embedding_dim(
    adjacency_a: NDArray[np.float64],
    adjacency_b: NDArray[np.float64],
) -> int:
    values_a = np.sort(np.abs(np.linalg.eigvalsh(adjacency_a)))[::-1]
    values_b = np.sort(np.abs(np.linalg.eigvalsh(adjacency_b)))[::-1]
    mean_values = (values_a + values_b) / 2
    dim = dmprdpg.zhu(mean_values)[1]
    return dim


def _as_generator(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)

def cauchy_combination_test(p_values, weights=None):
    """
    Calculates the combined p-value using the Cauchy Combination Test (CCT).
    """
    p_vals = np.array(p_values)

    # Clip p-values
    p_vals = np.clip(p_vals, 1e-16, 1 - 1e-16)

    # Default to equal weights
    if weights is None:
        w = np.repeat(1.0 / len(p_vals), len(p_vals))
    else:
        w = np.array(weights) / np.sum(weights)

    transformed = np.tan((0.5 - p_vals) * np.pi)

    t_stat = np.sum(w * transformed)

    combined_p = cauchy.sf(t_stat)

    return combined_p

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
    individual_pvals = list()
    for j in range(T):
        for k in range(K - 1):
            for l in range(k + 1, K):
                result = rdpg_two_sample_pvalue(A_dict[k, j], A_dict[l, j], embedding_dim=None, n_bootstraps=num_boot)
                individual_pvals.append(result.p_value)
    p_values[i] = cauchy_combination_test(individual_pvals)

np.savetxt(input_folder + f"/test_tang_semiparametric_n={n}_eta={eta}.csv", p_values, delimiter=",")