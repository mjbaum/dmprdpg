# Dynamic Multiplex Embedding

In this work, we explore a spectral embedding method for dynamic graphs with multiple layers. The base model for this work is a network in which a set of shared nodes exhibit connections across a number of different layers and this multiplex network is observed at a fixed number of points in time. We extend the theory of Unfolded Adjacency Spectral Embedding (UASE) to the dynamic case and plan to provide stability guarantees as well as a central limit theorem.

## Background and Notation
We consider the case of an undirected network with $K$ layers observed at $T$ points in time for $K, T \in \mathbb{N}$.  This network can be encoded in a collection of adjacency matrices $\textbf{A} = \{\textbf{A}_{k,t}\}$ where $k = 1,\dots , K$ and $t = 1,\dots , T$. Currently, we consider only the case of undirected networks. For our model, we adopt the concept of the latent position model in which the connection probabilities between nodes are defined by each node's latent position in an underlying $d$ dimensional embedding space. Specifically, each node in our model is represented by a position in two different embedding spaces $\mathcal{X} \subset \mathbb{R}^d$ and $\mathcal{Y} \subset \mathbb{R}^d$ where the positions $\textbf{X}^{k}_i \in \mathcal{X}$ are shared across time but are different across layers and the positions $\textbf{Y}^{t}_j \in \mathcal{Y}$ are shared across layers but vary over time. The connection probability for nodes $i$ and $j$ at time $t$ in layer $k$ is given by the inner product of these positions. We can therefore express the adjacency matrices probabilistically as 
```math
\textbf{A}_{k,t, i,j} \sim \mathrm{Bernoulli}\left(\textbf{X}^{k \intercal}_i \textbf{Y}^{t}_j\right).
```

## Methodology

Given a set of adjacency matrices $\textbf{A}_{k,t} \in \{0,1\}^{n \times n}$ we define the adjacency unfolding $\textbf{A} \in \{0,1\}^{nK \times nT}$ as:  
```math
\textbf{A} = 
\begin{bmatrix}
\textbf{A}_{1,1} & \dots & \textbf{A}_{1,T} \\
\vdots & \ddots & \vdots \\
\textbf{A}_{K,1} & \dots & \textbf{A}_{K,T}
\end{bmatrix}.
```

In order to estimate the latent positions $\textbf{X}$ and $\textbf{Y}$ from the realized matrix $\textbf{A}$, we propose Doubly Unfolded Adjacency Spectral Embedding (DUASE) for dynamic multiplex graphs. Given the realized adjacency matrices $\textbf{A}_{k,t}$ $k = 1,\dots , K$ and $t = 1,\dots , T$ we make use of a truncated SVD of rank $d$ to obtain a low-rank approximation of the doubly unfolded matrix $\textbf{A}$ as $\textbf{A} \approx \textbf{UDV}^{\intercal}$ where $\textbf{D}$ contains the top $d$ singular values on the diagonal and $\textbf{U}$ and $\textbf{V}$ contain the corresponding singular vectors. The estimates for the latent positions for each node are recovered according to 

$$\hat{\textbf{X}} = \textbf{UD}^{1/2}, \hspace{1cm} \hat{\textbf{Y}} = \textbf{VD}^{1/2}.$$

We then retrieve the estimates $\hat{\textbf{X}}^k$ and $\hat{\textbf{Y}}^t$ by unstacking the $n \times d$ chunks of $\hat{\textbf{X}}$ and $\hat{\textbf{Y}}$ .

## Simulation Results

We have generated a number of computer simulations to empirically test the properties of our embedding technique in the simplified case of a stochastic block model. This represents the setting where each node is assigned to one of a fixed number of communities which have a shared latent position. In this case, our model has $1000$ nodes and these are equally distributed among four communities with latent positions defined such that at time $t$ in layer $k$ the $(i,j)^{th}$ entry of $B^{(k)}_t$ denotes the connection probability between a node in community $i$ and a node in community $j$. 

```math
B^{(1)}_1 = \begin{bmatrix}
0.08 & 0.02 & 0.18 & 0.10\\
0.02 & 0.20 & 0.04 & 0.10\\
0.18 & 0.04 & 0.02 & 0.02\\
0.10 & 0.10 & 0.02 & 0.06
\end{bmatrix},
%
B^{(1)}_2 = \begin{bmatrix}
0.16 & 0.16 & 0.04 & 0.10\\
0.16 & 0.16 & 0.04 & 0.10\\
0.04 & 0.04 & 0.09 & 0.02\\
0.10 & 0.10 & 0.02 & 0.06
\end{bmatrix},
%
B^{(1)}_3 = \begin{bmatrix}
0.08 & 0.02 & 0.18 & 0.10\\
0.02 & 0.20 & 0.04 & 0.10\\
0.18 & 0.04 & 0.02 & 0.02\\
0.10 & 0.10 & 0.02 & 0.06
\end{bmatrix}
```
```math

B^{(2)}_1 = \begin{bmatrix}
0.08 & 0.02 & 0.18 & 0.10\\
0.02 & 0.20 & 0.04 & 0.10\\
0.18 & 0.04 & 0.02 & 0.02\\
0.10 & 0.10 & 0.02 & 0.06
\end{bmatrix},
%
B^{(2)}_2 = \begin{bmatrix}
0.16 & 0.16 & 0.04 & 0.10\\
0.16 & 0.16 & 0.04 & 0.10\\
0.04 & 0.04 & 0.09 & 0.02\\
0.10 & 0.10 & 0.02 & 0.06
\end{bmatrix},
%
B^{(2)}_3 = \begin{bmatrix}
0.08 & 0.02 & 0.18 & 0.10\\
0.02 & 0.20 & 0.04 & 0.10\\
0.18 & 0.04 & 0.02 & 0.02\\
0.10 & 0.10 & 0.02 & 0.06
\end{bmatrix}
```
```math

B^{(3)}_1 = \begin{bmatrix}
0.08 & 0.08 & 0.08 & 0.08\\
0.08 & 0.08 & 0.08 & 0.08\\
0.08 & 0.08 & 0.08 & 0.08\\
0.08 & 0.08 & 0.08 & 0.08
\end{bmatrix},
%
B^{(3)}_2 = \begin{bmatrix}
0.08 & 0.08 & 0.08 & 0.08\\
0.08 & 0.08 & 0.08 & 0.08\\
0.08 & 0.08 & 0.08 & 0.08\\
0.08 & 0.08 & 0.08 & 0.08
\end{bmatrix},
%
B^{(3)}_3 = \begin{bmatrix}
0.08 & 0.08 & 0.08 & 0.08\\
0.08 & 0.08 & 0.08 & 0.08\\
0.08 & 0.08 & 0.08 & 0.08\\
0.08 & 0.08 & 0.08 & 0.08
\end{bmatrix} 
```
In Figures \ref{fig:leftembeddings} and \ref{fig:rightembeddings}, the true latent position for each node is one of four points denoted with an orange $\times$ and the estimated latent positions for each community are plotted by color. The mean embedding for each group is denoted with a red dot. These simulations suggest that the desired stability properties are likely to apply. It also appears that our estimators for the latent positions are consistent and that the errors are normally distributed. Our comparisons of estimator variance for networks of different sizes in Figure \ref{fig:variance } appears to be roughly consistent with $1/n$ scaling which along with the Q-Q plots in Figure \ref{fig:qqplots} is consistent with the central limit theorem we aim to prove.
