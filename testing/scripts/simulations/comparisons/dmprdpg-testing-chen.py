import argparse
from scipy.stats import cauchy
import numpy as np
from scipy.stats import gamma
from typing import List, Optional, Dict, Any

# Initialize Static Parameters
K = 10
T = 3
reps = 50

## PARSER to give parameter values
parser = argparse.ArgumentParser()
## Set destination folder for output
parser.add_argument("-f","--folder", type=str, dest="folder", default="simulation_1", const=True, nargs="?",\
    help="String: name of the folder for the input files.")
parser.add_argument("-n", type=int, dest="n", default=100, const=True, nargs="?",\
	help="Integer: number of nodes in each graph. Default: d=100.")
parser.add_argument("-eta", type=float, dest="eta", default=0.0, const=True, nargs="?",\
	help="Float: size of perturbation. Default: 0.0.")
parser.add_argument("-r", type=int, dest="r", default=1000, const=True, nargs="?",\
	help="Integer: number of replicates. Default: n=1000.")

## Parse arguments
args = parser.parse_args()
input_folder = args.folder

n = args.n
eta = args.eta
num_replicates = args.r

# Generate multiple copies of a DMPSBM
def simulate_dmpsbm_reps(n, B_dict, K=None, T=None, prior_G=None, prior_G_prime=None, seed=None, z_shared=False, undirected=False, reps=1):
    # Initialise number of layers and time steps from B[0,0] (if present)
    if (0,0) in B_dict:
        G = B_dict[0,0].shape[0]
        G_prime = B_dict[0,0].shape[1]
    else:
        raise ValueError("B_dict must contain an entry for (0,0)")
    # Check if undirected is Boolean. If it is, check that the B matrices are symmetric.
    if not isinstance(undirected, bool):
        raise ValueError("undirected must be a boolean")
    # z_shared must be boolean
    if not isinstance(z_shared, bool):
        raise ValueError("z_shared must be a boolean")
    if undirected:
        if not all(np.allclose(B_dict[key], B_dict[key].T) for key in B_dict.keys()):
            raise ValueError("All matrices in B_dict must be symmetric")
        ## z_shared must be True if undirected is True
        if not z_shared:
            raise ValueError("z_shared must be True if undirected is True")
    # Check z_shared and return an error if G != G_prime
    if z_shared:
        if G != G_prime:
            raise ValueError("G must be equal to G_prime if z_shared is True")
    ## Check that all matrices in B_dict have the same dimensions
    if not all(B_dict[key].shape == (G, G_prime) for key in B_dict.keys()):
        raise ValueError("All matrices in B_dict must have the same dimension")
    ## If K and T are not provided, set them to the number of unique entries in the rows/columns of the keys of B_dict
    if K is None:
        K = len(set(key[0] for key in B_dict.keys()))
    if T is None:
        T = len(set(key[1] for key in B_dict.keys()))
    ## Check that the entries of B_dict are all possible pairs of range(K) and range(T)
    if not all(key in B_dict for key in [(k, t) for k in range(K) for t in range(T)]):
        raise ValueError("B_dict must contain all possible (k,t) pairs for k=0,...,K-1 and t=0,...,T-1")
    ## If priors are None, assume identical probability vectors for all layers and times
    if prior_G is None:
        prior_G = [1/G] * G
    else:
        if len(prior_G) != G:
            raise ValueError("Length of prior_G must match G.")
        if not np.isclose(sum(prior_G), 1):
            raise ValueError("Priors must sum to 1")
        if not all(p >= 0 for p in prior_G):
            raise ValueError("Priors must be non-negative")
    if not z_shared:
        if prior_G_prime is None:
            prior_G_prime = [1/G_prime] * G_prime
        else:
            if len(prior_G_prime) != G_prime:
                raise ValueError("Length of prior_G_prime must match G_prime.")
            if not np.isclose(sum(prior_G_prime), 1):
                raise ValueError("Priors must sum to 1")
            if not all(p >= 0 for p in prior_G_prime):
                raise ValueError("Priors must be non-negative")
    ## Set seed if provided
    if seed is not None:
        np.random.seed(seed)
    ## Generate the group labels
    z = np.random.choice(range(G), size=n, p=prior_G)
    if not z_shared:
        z_prime = np.random.choice(range(G_prime), size=n, p=prior_G_prime)
    else:
        z_prime = np.copy(z)
    ## Simulate a stochastic blockmodel for each matrix in B_dict, storing A_{kt} in a sparse matrix
    A_dict = {}
    ## Obtain the graph as an edgelist
    for k in range(K):
        for t in range(T):
            for r in range(reps):
                adjacency_matrix = np.zeros((n,n))
                if undirected:
                    for i in range(n):
                        for j in range(i+1, n):
                            adjacency_matrix[i,j] = adjacency_matrix[j,i] = np.random.binomial(1, B_dict[k, t][z[i], z[j]])
                else:
                    for i in range(n):
                        for j in range(n):
                            if i != j:
                                adjacency_matrix[i,j] = np.random.binomial(1, B_dict[k, t][z[i], z_prime[j]])
                A_dict[k,t,r] = adjacency_matrix
    ## Return output
    if undirected:
        return A_dict, z
    else:
        if z_shared:
            return A_dict, z
        else:
            return A_dict, z, z_prime


def _validate_group(A: np.ndarray, name: str) -> None:
    """
    Validate a network sample tensor.

    Expected shape:
        (m, n, n)
    """

    if A.ndim != 3:
        raise ValueError(f"{name} must have shape (m, n, n)")

    m, n1, n2 = A.shape

    if n1 != n2:
        raise ValueError(f"{name} matrices must be square")

    if not np.allclose(A, np.transpose(A, (0, 2, 1))):
        raise ValueError(f"{name} matrices must be symmetric")

    diag = np.diagonal(A, axis1=1, axis2=2)

    if not np.allclose(diag, 0):
        raise ValueError(f"{name} matrices must have zero diagonal")


def _sample_mean(A: np.ndarray) -> np.ndarray:
    """
    Compute sample mean adjacency matrix.
    """
    return A.mean(axis=0)


def _estimate_P_avg(A: np.ndarray) -> np.ndarray:
    """
    SPE-AVG estimator.
    """
    return _sample_mean(A)


def _build_Zs(
    Abar_s: np.ndarray,
    Abar: np.ndarray,
    P_hat_s: np.ndarray,
    P_hats: List[np.ndarray],
    ms: np.ndarray,
    s: int,
    rng: np.random.Generator,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Construct Z^(s).

    Vectorized implementation.
    """

    n = Abar.shape[0]

    m_total = ms.sum()
    ms_s = ms[s]

    # Variance term
    pooled_var = np.zeros((n, n), dtype=float)

    for r, P_hat_r in enumerate(P_hats):
        pooled_var += ms[r] * P_hat_r * (1.0 - P_hat_r)

    pooled_var /= (m_total ** 2)

    Vs = (
        (1.0 / ms_s - 2.0 / m_total)
        * P_hat_s
        * (1.0 - P_hat_s)
        + pooled_var
    )

    Vs = np.clip(Vs, eps, None)

    # Construct Z^(s)

    Zs = (Abar_s - Abar) / np.sqrt(n * Vs)

    Zs = 0.5 * (Zs + Zs.T)

    diag = rng.choice([-1.0, 1.0], size=n) / np.sqrt(n)

    np.fill_diagonal(Zs, diag)

    return Zs


def _theta(Z: np.ndarray) -> float:
    """
    Compute:
        theta = (1/sqrt(15)) Tr(Z^3)

    Efficient implementation:
        Tr(Z^3) = sum((Z @ Z) * Z)
    """

    Z2 = Z @ Z

    return np.sum(Z2 * Z) / np.sqrt(15.0)


def spectral_multi_sample_test(
    groups: List[np.ndarray],
    Q: int = 500,
    random_state: Optional[int] = None,
    return_details: bool = False,
) -> Dict[str, Any]:

    # Check inputs
    if len(groups) < 2:
        raise ValueError("At least two groups are required")

    for idx, G in enumerate(groups):
        _validate_group(G, f"group[{idx}]")

    n = groups[0].shape[1]

    for G in groups:
        if G.shape[1] != n:
            raise ValueError("All groups must have same number of nodes")

    rng = np.random.default_rng(random_state)

    S = len(groups)

    # Calculate sample sizes
    ms = np.array([G.shape[0] for G in groups])

    # Calculate group means
    Abar_s = [_sample_mean(G) for G in groups]

    # Pooled mean
    m_total = ms.sum()

    Abar = sum(ms[s] * Abar_s[s] for s in range(S)) / m_total

    # Link probability estimates
    P_hats = [_estimate_P_avg(G) for G in groups]

    # Monte Carlo theta^(s)
    theta_mc = np.empty((Q, S))

    for q in range(Q):

        for s in range(S):

            Zs = _build_Zs(
                Abar_s=Abar_s[s],
                Abar=Abar,
                P_hat_s=P_hats[s],
                P_hats=P_hats,
                ms=ms,
                s=s,
                rng=rng,
            )

            theta_mc[q, s] = _theta(Zs)

    # Monte Carlo averages
    theta_bar = theta_mc.mean(axis=0)

    # Omnibus statistic
    omnibus_stat = np.sum(theta_bar ** 2)

    # Estimate correlations rho_qr
    squared_mc = theta_mc ** 2

    corr_matrix = np.corrcoef(squared_mc, rowvar=False)

    # Compute gamma scale parameter u
    off_diag_sum = (
        corr_matrix.sum() - np.trace(corr_matrix)
    )

    u = 2.0 * (1.0 + 2.0 * off_diag_sum / S)

    # Gamma approximation: theta ~ Gamma(S/u, scale=u)
    gamma_shape = S / u
    gamma_scale = u

    # p-value
    p_value = gamma.sf(
        omnibus_stat,
        a=gamma_shape,
        scale=gamma_scale,
    )

    results = {
        "p_value": p_value,
        "statistic": omnibus_stat,
        "gamma_shape": gamma_shape,
        "gamma_scale": gamma_scale,
    }

    if return_details:
        results.update({
            "theta_bar": theta_bar,
            "theta_mc": theta_mc,
            "corr_matrix": corr_matrix,
        })

    return results

def cauchy_combination_test(p_values, weights=None):
    """
    Calculate combined p-value using the Cauchy Combination Test
    """
    p_vals = np.array(p_values)

    # Clip p-values to avoid infinity at 0 or 1
    p_vals = np.clip(p_vals, 1e-16, 1 - 1e-16)

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
        B_dict_equal[(k,t)] = np.array([[0.25 + eta * k, 0.1 + np.sin(np.pi * t / T) * eps2], [0.1 + np.sin(np.pi * t / T) * eps2, 0.25]])

# Calculate P-values
p_values = np.zeros(num_replicates)


for i in range(num_replicates):
    A_dict, z = simulate_dmpsbm_reps(n=n, B_dict=B_dict_equal, undirected=True, z_shared=True, reps=reps)
    individual_pvals = list()
    for time in range(T):
        G1 = np.stack([A_dict[0,time,r] for r in range(reps)])
        G2 = np.stack([A_dict[1,time,r] for r in range(reps)])
        G3 = np.stack([A_dict[2,time,r] for r in range(reps)])
        G4 = np.stack([A_dict[3,time,r] for r in range(reps)])
        G5 = np.stack([A_dict[4,time,r] for r in range(reps)])
        G6 = np.stack([A_dict[5,time,r] for r in range(reps)])
        G7 = np.stack([A_dict[6,time,r] for r in range(reps)])
        G8 = np.stack([A_dict[7,time,r] for r in range(reps)])
        G9 = np.stack([A_dict[8,time,r] for r in range(reps)])
        G10 = np.stack([A_dict[9,time,r] for r in range(reps)])

        # Run test
        results = spectral_multi_sample_test(
            [G1, G2, G3, G4, G5, G6, G7, G8, G9, G10],
            Q=200,
            random_state=1,
        )
        individual_pvals.append(results["p_value"])
    p_values[i] = cauchy_combination_test(individual_pvals)

# Save results
np.savetxt(input_folder + f"/testing_chen_beta_n={n}_eta={eta}.csv", p_values, delimiter=",")