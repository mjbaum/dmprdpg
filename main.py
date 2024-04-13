#!/usr/bin/env python3
# Import package
from dmprdpg import dmpsbm

# Press the green button in the gutter to run the script
if __name__ == '__main__':
    # Pre-define matrix B - Index is (layer, time)
    B_dict = {
        (0, 0): [[0.08, 0.02, 0.18, 0.10], [0.02, 0.20, 0.04, 0.10], [0.18, 0.04, 0.02, 0.02], [0.10, 0.10, 0.02, 0.06]],
        (0, 1): [[0.16, 0.16, 0.04, 0.10], [0.16, 0.16, 0.04, 0.10], [0.04, 0.04, 0.09, 0.02], [0.10, 0.10, 0.02, 0.06]],
        (0, 2): [[0.08, 0.02, 0.18, 0.10], [0.02, 0.20, 0.04, 0.10], [0.18, 0.04, 0.02, 0.02], [0.10, 0.10, 0.02, 0.06]],
        (1, 0): [[0.08, 0.02, 0.18, 0.10], [0.02, 0.20, 0.04, 0.10], [0.18, 0.04, 0.02, 0.02], [0.10, 0.10, 0.02, 0.06]],
        (1, 1): [[0.16, 0.16, 0.04, 0.10], [0.16, 0.16, 0.04, 0.10], [0.04, 0.04, 0.09, 0.02], [0.10, 0.10, 0.02, 0.06]],
        (1, 2): [[0.08, 0.02, 0.18, 0.10], [0.02, 0.20, 0.04, 0.10], [0.18, 0.04, 0.02, 0.02], [0.10, 0.10, 0.02, 0.06]],
        (2, 0): [[0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08]],
        (2, 1): [[0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08]],
        (2, 2): [[0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08], [0.08, 0.08, 0.08, 0.08]]
    }
    # Define a DMP-SBM model with 3 layers and 3 timesteps, n=100 nodes and 4 communities of 25 nodes each
    model = dmpsbm(layers=3, timesteps=3, groups=[25,25,25,25], prob_dict=B_dict)
    # Sample the adjacency matrices from the DMP-SBM model and calculate the DUASE embedding
    model.sample()
    # Calculate the centroids of the sampled embeddings
    model.get_centroids()
    model.get_centroids_theo()
    # model.qq_plot()
    model.plot()
    # Pre-define a different matrix B - Index is (layer, time)
    B_dict = {
        (0, 0): [[0.08, 0.02, 0.18, 0.10], [0.02, 0.20, 0.04, 0.10], [0.18, 0.04, 0.02, 0.02], [0.10, 0.10, 0.02, 0.06]],
        (0, 1): [[0.16, 0.16, 0.04, 0.04], [0.16, 0.16, 0.04, 0.04], [0.04, 0.04, 0.12, 0.18], [0.04, 0.04, 0.18, 0.16]],
        (1, 0): [[0.06, 0.18, 0.04, 0.04], [0.18, 0.12, 0.04, 0.04], [0.04, 0.04, 0.02, 0.02], [0.04, 0.04, 0.02, 0.02]],
        (1, 1): [[0.16, 0.16, 0.04, 0.04], [0.16, 0.16, 0.04, 0.04], [0.04, 0.04, 0.02, 0.02], [0.04, 0.04, 0.02, 0.02]],
    }
    # Repeat the process for a different set of matrices
    model2 = dmpsbm(layers=2, timesteps=2, groups=[250,250,250,250], prob_dict=B_dict)
    model2.sample()
    model2.get_centroids()
    model2.get_centroids_theo()
    model2.plot()
