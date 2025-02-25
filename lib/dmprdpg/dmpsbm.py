from .helpers import generate_adjacency_matrix, get_embedding, generate_group_labels, group_by_label
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab as py
from scipy.linalg import orthogonal_procrustes
from scipy.sparse import coo_matrix

## Simulate a DMP-SBM model
def simulate_dmpsbm(n, B_dict, K=None, T=None, prior_G=None, prior_G_prime=None, seed=None, z_shared=False, undirected=False):
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
    ## Simulate a stochastic blockmodel for each matrix in B_dict, storing A_{kt} in a sparse matrix
    A_dict = {}
    ## Obtain the graph as an edgelist
    for k in range(K):
        for t in range(T):
            edgelist = []
            if undirected:
                for i in range(n):
                    for j in range(i+1, n):
                        if np.random.binomial(1, B_dict[k, t][z[i], z[j]]) == 1:
                            edgelist += [(i, j), (j, i)]
            else:
                for i in range(n):
                    for j in range(n):
                        if i != j and np.random.binomial(1, B_dict[k, t][z[i], z_prime[j]]) == 1:
                            edgelist += [(i, j)] 
            # Extract nodes and weights from the edge list
            rows = [edge[0] for edge in edgelist]
            cols = [edge[1] for edge in edgelist]
            data = [1.0] * len(edgelist)
            # # Create the sparse adjacency matrix in COO format
            adjacency_matrix = coo_matrix((data, (rows, cols)), shape=(n,n))
            # Convert to CSR format
            A_dict[k,t] = adjacency_matrix.tocsr()
    ## Return output
    if undirected:
        return A_dict, z
    else: 
        return A_dict, z, z_prime

## Full class for simulation
class dmpsbm:

    # Initialize the model with the number of layers, timesteps, groups, and the dictionary of probabilities
    def __init__(self, layers, timesteps, groups, prob_dict):
        # Store the model parameters (after checking that they are valid)
        if not isinstance(layers, int) or layers <= 0:
            raise ValueError("The number of layers must be a positive integer")
        self.layers = layers
        if not isinstance(timesteps, int) or timesteps <= 0:
            raise ValueError("The number of timesteps must be a positive integer")
        self.timesteps = timesteps
        if not isinstance(groups, list) or len(groups) == 0 or not all(isinstance(x, int) for x in groups):
            raise ValueError("The groups must be a non-empty list of integers")
        self.groups = groups
        if not isinstance(prob_dict, dict) or not all(isinstance(key, tuple) and len(key) == 2 and isinstance(value, list) for key, value in prob_dict.items()):
            raise ValueError("The probability dictionary must be a dictionary with keys as tuples and values as lists")
        self.prob_dict = prob_dict
        # Initialize other model attributes to None
        self.A = None
        self.left_embedding = None
        self.right_embedding = None
        self.left_centroids = None
        self.right_centroids = None
        self.left_embedding_theo = None
        self.right_embedding_theo = None
        self.rotation_left = None
        self.rotation_right = None
        self.error = 0

    # Sample the adjacency matrices and calculate the embeddings
    def sample(self):
        list_longs = []
        for i in range(self.layers):
            curr_long = []
            for j in range(self.timesteps):
                curr_A = generate_adjacency_matrix(len_groups=self.groups, probabilities=self.prob_dict[(i, j)])
                curr_long.append(curr_A)
            final_long = np.concatenate(curr_long, axis=1)
            list_longs.append(final_long)
        final_embedding = np.concatenate(list_longs, axis=0)
        self.A = final_embedding
        left_embedding = get_embedding(final_embedding, type='left')
        self.left_embedding = left_embedding
        right_embedding = get_embedding(final_embedding, type='right')
        self.right_embedding = right_embedding

    # Calculate the theoretical embeddings and rotate them to match the sampled embeddings
    def get_centroids_theo(self):
        list_longs = []
        for i in range(self.layers):
            curr_long = []
            for j in range(self.timesteps):
                curr_B = self.prob_dict[(i, j)]
                curr_long.append(curr_B)
            final_long = np.concatenate(curr_long, axis=1)
            list_longs.append(final_long)
        final_embedding = np.concatenate(list_longs, axis=0)
        self.left_embedding_theo = get_embedding(final_embedding, type='left')
        self.right_embedding_theo = get_embedding(final_embedding, type='right')
        self.rotate()

    # Calculate the rotation matrices to align the theoretical embeddings with the sampled embeddings
    def get_rotation(self):
        left_stacked = np.concatenate(self.left_centroids, axis = 0)
        rotation = orthogonal_procrustes(self.left_embedding_theo, left_stacked)[0]
        self.rotation_left = rotation
        right_stacked = np.concatenate(self.right_centroids, axis=0)
        rotation = orthogonal_procrustes(self.right_embedding_theo, right_stacked)[0]
        self.rotation_right = rotation

    # Rotate the theoretical embeddings to match the sampled embeddings
    def rotate(self):
        self.get_rotation()
        self.left_embedding_theo = self.left_embedding_theo @ self.rotation_left
        self.right_embedding_theo = self.right_embedding_theo @ self.rotation_right
        self.calculate_error()
        print("Total Error: ", self.error)

    # Calculate the error between the sampled and theoretical embeddings
    def calculate_error(self):
        left_stacked = np.concatenate(self.left_centroids, axis=0)
        right_stacked = np.concatenate(self.right_centroids, axis=0)
        self.error = sum(sum((self.left_embedding_theo - left_stacked)**2)) + sum(sum((self.right_embedding_theo - right_stacked)**2))
        self.calculate_variance()

    # Calculate the variance of the embeddings within each community
    def calculate_variance(self):
        num_nodes = sum(self.groups)
        for layer in range(self.layers):
            current_layer = self.left_embedding[num_nodes*layer:num_nodes*(layer+1), :]
            start = 0
            variances = []
            for size in self.groups:
                variances.append(sum(np.var(current_layer[start:start + size, :], axis=0)))
                start += size
            plt.bar(x = range(len(self.groups)), height = variances, color = 'darkblue')
            plt.title("Community Variances Layer " + str(layer+1))
            plt.show()
        for time in range(self.timesteps):
            current_time = self.right_embedding[num_nodes*time:num_nodes*(time+1), :]
            start = 0
            variances = []
            for size in self.groups:
                variances.append(sum(np.var(current_time[start:start + size, :], axis=0)))
                start += size
            plt.bar(x = range(len(self.groups)), height = variances, color = 'darkblue')
            plt.title("Community Variances Time " + str(time+1))
            plt.show()

    # Calculate the centroids of the communities in the embeddings
    def get_centroids(self):
        left_centroids = []
        right_centroids = []
        total_nodes = sum(self.groups)
        for layer in range(self.layers):
            current_layer = self.left_embedding[total_nodes*layer:total_nodes*(layer+1), :]
            current_embeddings = []
            for label in set(generate_group_labels(len_groups=self.groups)):
                labels = np.array(generate_group_labels(len_groups=self.groups))
                community = current_layer[labels == label]
                current_embeddings.append([np.mean(community[:, 0]), np.mean(community[:, 1]), np.mean(community[:, 2]), np.mean(community[:, 3])])
            left_centroids.append(current_embeddings)
        self.left_centroids = left_centroids
        for time in range(self.timesteps):
            current_time = self.right_embedding[total_nodes*time:total_nodes*(time+1), :]
            current_embeddings = []
            for label in set(generate_group_labels(len_groups=self.groups)):
                labels = np.array(generate_group_labels(len_groups=self.groups))
                community = current_time[labels == label]
                current_embeddings.append([np.mean(community[:, 0]), np.mean(community[:, 1]), np.mean(community[:, 2]), np.mean(community[:, 3])])
            right_centroids.append(current_embeddings)
        self.right_centroids = right_centroids

    # Plot the embeddings and centroids
    def plot(self):
        total_nodes  = sum(self.groups)
        num_groups = len(self.groups)
        for layer in range(self.layers):
            fig, ax = plt.subplots()
            ax.grid()
            ax.scatter(x=self.left_embedding[total_nodes*layer:total_nodes*(layer+1), 0], y=self.left_embedding[total_nodes*layer:total_nodes*(layer+1), 1], c=generate_group_labels(len_groups=self.groups))
            ax.scatter(x=self.left_embedding_theo[num_groups * layer:num_groups * (layer + 1), 0], y=self.left_embedding_theo[num_groups * layer:num_groups * (layer + 1), 1], c='orange', marker='x', s=80)
            for point in self.left_centroids[layer]:
                ax.scatter(point[0], point[1], c='red')
            plt.title("Left Embedding Layer " + str(layer+1))
            plt.show()
        for time in range(self.timesteps):
            fig, ax = plt.subplots()
            ax.grid()
            ax.scatter(x=self.right_embedding[total_nodes*time:total_nodes*(time+1), 0], y=self.right_embedding[total_nodes*time:total_nodes*(time+1), 1], c=generate_group_labels(len_groups=self.groups))
            ax.scatter(x=self.right_embedding_theo[num_groups * time:num_groups * (time + 1), 0], y=self.right_embedding_theo[num_groups * time:num_groups * (time + 1), 1], c='orange', marker='x', s=80)
            for point in self.right_centroids[time]:
                ax.scatter(point[0], point[1], c='red')
            plt.title("Right Embedding Time " + str(time+1))
            plt.show()

    # Generate a QQ plot for the embeddings (marginally for each dimension)
    def qq_plot(self):
        num_nodes = sum(self.groups)
        for layer in range(self.layers):
            current_layer = self.left_embedding[num_nodes * layer:num_nodes * (layer + 1), :]
            start = 0
            for size in self.groups:
                community = current_layer[start:start + size, :]
                mean = np.mean(community, axis=0)
                community = community - mean
                for dimension in range(4):
                    sm.qqplot(community[:, dimension], fit=True, line=45)
                    py.show()
                start += size