from helpers import generate_adjacency_matrix, get_embedding, generate_group_labels, group_by_label
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import orthogonal_procrustes


class Dynamic_Multilayer_SBM:

    def __init__(self, layers, timesteps, groups, prob_dict):
        self.layers = layers
        self.timesteps = timesteps
        self.groups = groups
        self.prob_dict = prob_dict
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
        print(np.shape(final_embedding))
        self.A = final_embedding
        left_embedding = get_embedding(final_embedding, type='left')
        self.left_embedding = left_embedding
        right_embedding = get_embedding(final_embedding, type='right')
        self.right_embedding = right_embedding


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
        print(np.shape(final_embedding))
        self.left_embedding_theo = get_embedding(final_embedding, type='left')
        self.right_embedding_theo = get_embedding(final_embedding, type='right')
        self.rotate()

    def get_rotation(self):
        left_stacked = np.concatenate(self.left_centroids, axis = 0)
        rotation = orthogonal_procrustes(self.left_embedding_theo, left_stacked)[0]
        self.rotation_left = rotation
        right_stacked = np.concatenate(self.right_centroids, axis=0)
        rotation = orthogonal_procrustes(self.right_embedding_theo, right_stacked)[0]
        self.rotation_right = rotation

    def rotate(self):
        self.get_rotation()
        self.left_embedding_theo = self.left_embedding_theo @ self.rotation_left
        self.right_embedding_theo = self.right_embedding_theo @ self.rotation_right
        self.calculate_error()
        print("Total Error: ", self.error)

    def calculate_error(self):
        left_stacked = np.concatenate(self.left_centroids, axis=0)
        right_stacked = np.concatenate(self.right_centroids, axis=0)
        self.error = sum(sum((self.left_embedding_theo - left_stacked)**2)) + sum(sum((self.right_embedding_theo - right_stacked)**2))
        self.calculate_variance()

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





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Index is (layer, time)
    B_dict = {
        (0,0):[[0.08,0.02,0.18,0.1], [0.02,0.2,0.04,0.1], [0.18,0.04,0.02,0.02], [0.1,0.1,0.02,0.06]],
        (0,1):[[0.16,0.16,0.04,0.1], [0.16,0.16,0.04,0.1], [0.04,0.04,0.09,0.02], [0.1,0.1,0.02,0.06]],
        (0, 2): [[0.08, 0.02, 0.18, 0.1], [0.02, 0.2, 0.04, 0.1], [0.18, 0.04, 0.02, 0.02], [0.1, 0.1, 0.02, 0.06]],
        (1, 0): [[0.08, 0.02, 0.18, 0.1], [0.02, 0.2, 0.04, 0.1], [0.18, 0.04, 0.02, 0.02], [0.1, 0.1, 0.02, 0.06]],
        (1, 1): [[0.16, 0.16, 0.04, 0.1], [0.16, 0.16, 0.04, 0.1], [0.04, 0.04, 0.09, 0.02], [0.1, 0.1, 0.02, 0.06]],
        (1, 2): [[0.08, 0.02, 0.18, 0.1], [0.02, 0.2, 0.04, 0.1], [0.18, 0.04, 0.02, 0.02], [0.1, 0.1, 0.02, 0.06]],
        (2, 0): [[0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08]],
        (2, 1): [[0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08]],
        (2, 2): [[0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08]]
    }
    model = Dynamic_Multilayer_SBM(layers=3, timesteps=3, groups=[250,250,250,250], prob_dict=B_dict)
    model.sample()
    model.get_centroids()
    model.get_centroids_theo()
    model.plot()

    B_dict = {
        (0,0):[[0.08,0.02,0.18,0.10], [0.02,0.20,0.04,0.10], [0.18,0.04,0.02,0.02], [0.10,0.10,0.02,0.06]],
        (0,1):[[0.16,0.16,0.04,0.04], [0.16,0.16,0.04,0.04], [0.04,0.04,0.12,0.18], [0.04,0.04,0.18,0.16]],
        (1, 0): [[0.06, 0.18, 0.04, 0.04], [0.18, 0.12, 0.04, 0.04], [0.04, 0.04, 0.02, 0.02], [0.04, 0.04, 0.02, 0.02]],
        (1, 1): [[0.16, 0.16, 0.04, 0.04], [0.16, 0.16, 0.04, 0.04], [0.04, 0.04, 0.02, 0.02], [0.04, 0.04, 0.02, 0.02]],
    }
    model2 = Dynamic_Multilayer_SBM(layers=2, timesteps=2, groups=[250,250,250,200], prob_dict=B_dict)
    model2.sample()
    model2.get_centroids()
    model2.get_centroids_theo()
    model2.plot()
