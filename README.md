# Dynamic Multiplex Embedding

In this work, we explore a spectral embedding method for dynamic graphs with multiple layers. The base model for this work is a network in which a set of shared nodes exhibit connections across a number of different layers and this multiplex network is observed at a fixed number of points in time. We extend the theory of Unfolded Adjacency Spectral Embedding (UASE) to the dynamic case and plan to provide stability guarantees as well as a central limit theorem.

## Background and Notation
We consider the case of an undirected network with $K$ layers observed at $T$ points in time for $K, T \in \mathbb{N}$.  This network can be encoded in a collection of adjacency matrices $\textbf{A} = \{\textbf{A}_{k,t}\}$ where $k = 1,\dots , K$ and $t = 1,\dots , T$. Currently, we consider only the case of undirected networks.

## Methodology

Given a set of adjacency matrices $\textbf{A}_{k,t} \in \{0,1\}^{n \times n}$ we define the adjacency unfolding $\textbf{A} \in \{0,1\}^{nK \times nT}$ as 
$$ \textbf{A} = \begin{bmatrix} \textbf{A}_{1,1} & \dots & \textbf{A}_{1,T}\\ \vdots & \ddots & \vdots \\ \textbf{A}_{K,1} & \dots & \textbf{A}_{K,T} \end{bmatrix} $$.

In order to estimate the latent positions $\textbf{X}$ and $\textbf{Y}$ from the realized matrix $\textbf{A}$, we propose Doubly Unfolded Adjacency Spectral Embedding (DUASE) for dynamic multiplex graphs. Given the realized adjacency matrices $\textbf{A}_{k,t}$ $k = 1,\dots , K$ and $t = 1,\dots , T$ we make use of a truncated SVD of rank $d$ to obtain a low-rank approximation of the doubly unfolded matrix $\textbf{A}$ as $\textbf{A} \approx \textbf{UDV}^{\intercal}$ where $\textbf{D}$ contains the top $d$ singular values on the diagonal and $\textbf{U}$ and $\textbf{V}$ contain the corresponding singular vectors. The estimates for the latent positions for each node are recovered according to 

$$\hat{\textbf{X}} = \textbf{UD}^{1/2}, \hspace{1cm} \hat{\textbf{Y}} = \textbf{VD}^{1/2}.$$

We then retrieve the estimates $\hat{\textbf{X}}^k$ and $\hat{\textbf{Y}}^t$ by unstacking the $n \times d$ chunks of $\hat{\textbf{X}}$ and $\hat{\textbf{Y}}$ .
