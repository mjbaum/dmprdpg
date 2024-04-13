# Spectral embedding of dynamic multiplex networks

This repository contains a _Python_ library supporting the work on spectral embedding of dynamic multiplex graphs by Maximilian Baum, Francesco Sanna Passino, and Axel Gandy (Department of Mathematics, Imperial College London). 

The library `dmprdpg` can be installed in edit mode as follows:
```
pip3 install -e lib/
```
The library can then be imported in any _Python_ session:
```python3
import dmprdpg
```

A demo on how to use the library can be found in `notebooks/test_library.ipynb`.

In this work, we explore a spectral embedding method for dynamic graphs with multiple layers. The base model for this work is a network in which a set of shared nodes exhibit connections across a number of different layers and this multiplex network is observed at a fixed number of points in time. We extend the theory of Unfolded Adjacency Spectral Embedding (UASE) to the dynamic case and plan to provide stability guarantees as well as a central limit theorem.

## Background and Notation
We consider the case of an undirected network with $K$ layers observed at $T$ points in time for $K, T \in \mathbb{N}$.  This network can be encoded in a collection of adjacency matrices $\textbf{A} = \{\textbf{A}_{k,t}\}$ where $k = 1,\dots , K$ and $t = 1,\dots , T$. Currently, we consider only the case of undirected networks. For our model, we adopt the concept of the latent position model in which the connection probabilities between nodes are defined by each node's latent position in an underlying $d$ dimensional embedding space. Specifically, each node in our model is represented by a position in two different embedding spaces $\mathcal{X} \subset \mathbb{R}^d$ and $\mathcal{Y} \subset \mathbb{R}^d$ where the positions $\textbf{X}^{k}_i \in \mathcal{X}$ are shared across time but are different across layers and the positions $\textbf{Y}^{t}_j \in \mathcal{Y}$ are shared across layers but vary over time. The connection probability for nodes $i$ and $j$ at time $t$ in layer $k$ is given by the inner product of these positions. We can therefore express the adjacency matrices probabilistically as 
```math
\textbf{A}_{k,t, i,j} \sim \mathrm{Bernoulli}\left(\textbf{X}^{k \intercal}_i \textbf{Y}^{t}_j\right).
```

## Methodology

Given a set of adjacency matrices $\textbf{A}_{k,t} \in \lbrace 0,1\rbrace^{n \times n}$ we define the adjacency unfolding $\textbf{A} \in \lbrace 0,1\rbrace ^{nK \times nT}$ as:  
```math
\textbf{A} = 
\begin{bmatrix}
\textbf{A}_{1,1} & \dots & \textbf{A}_{1,T} \\
\vdots & \ddots & \vdots \\
\textbf{A}_{K,1} & \dots & \textbf{A}_{K,T}
\end{bmatrix}.
```

In order to estimate the latent positions $\textbf{X}$ and $\textbf{Y}$, we propose Doubly Unfolded Adjacency Spectral Embedding (DUASE) for dynamic multiplex graphs. Given the realized adjacency matrices $\textbf{A}_{k,t}$ $k = 1,\dots , K$ and $t = 1,\dots , T$ we make use of a truncated SVD of rank $d$ to obtain a low-rank approximation of the doubly unfolded matrix $\textbf{A}$ as $\textbf{A} \approx \textbf{UDV}^{\intercal}$ where $\textbf{D}$ contains the top $d$ singular values on the diagonal and $\textbf{U}$ and $\textbf{V}$ contain the corresponding singular vectors. The estimates for the latent positions for each node are recovered according to 

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
In the figures below, the true latent position for each node is one of four points denoted with an orange $\times$ and the estimated latent positions for each community are plotted by color. The mean embedding for each group is denoted with a red dot. These simulations show that our estimates for each group are generally clustered around the true latent positions and suggest that the desired stability properties are likely to apply. 


<table>
  <tr>
    <td><img src="https://github.com/mjbaum/Dynamic_Multiplex_Embedding/assets/150443188/3758e888-c3ac-4777-a9c8-7e5ab7629e0e" width="300" </td>
    <td><img src="https://github.com/mjbaum/Dynamic_Multiplex_Embedding/assets/150443188/0c0655af-f97e-4a5d-b3bc-805b99c391b4" width="300"</td>
    <td><img src="https://github.com/mjbaum/Dynamic_Multiplex_Embedding/assets/150443188/481bc902-554d-465c-95ab-9faf2c7000d4" width="300"</td>
  </tr>
 </table>

 <table>
  <tr>
    <td><img src="https://github.com/mjbaum/Dynamic_Multiplex_Embedding/assets/150443188/502c8497-aa6e-435b-ab9c-be4a8d41c748" width="300" </td>
    <td><img src="https://github.com/mjbaum/Dynamic_Multiplex_Embedding/assets/150443188/e8c72cf4-1199-41c5-a074-fca75909785f" width="300"</td>
    <td><img src="https://github.com/mjbaum/Dynamic_Multiplex_Embedding/assets/150443188/a38c4784-b27f-4935-941f-c3ae55700f06" width="300"</td>
  </tr>
 </table>


It also appears that our estimators for the latent positions have normally distributed errors and a comparison of estimator variance for networks of different sizes appears to be roughly consistent with $1/n$ scaling. These results are consistent with the central limit theorem we aim to prove.
<table>
  <tr>
    <td><img src="https://github.com/mjbaum/Dynamic_Multiplex_Embedding/assets/150443188/44dda3fe-6cf1-4d5f-b988-693ef16213a4" width="450" </td>
    <td><img src="https://github.com/mjbaum/Dynamic_Multiplex_Embedding/assets/150443188/0c481992-ae18-49c1-8b8c-2f87882a74d9" width="450" </td>
  </tr>
 </table>
<table>
  <tr>
    <td><img src="https://github.com/mjbaum/Dynamic_Multiplex_Embedding/assets/150443188/456ca546-ad08-48fe-bd93-ea2d6930e64a" width="300" </td>
    <td><img src="https://github.com/mjbaum/Dynamic_Multiplex_Embedding/assets/150443188/3d765b24-e025-4f02-af3f-b5e50c25c4b3" width="300"</td>
    <td><img src="https://github.com/mjbaum/Dynamic_Multiplex_Embedding/assets/150443188/713e5b83-84fa-4192-b74b-2849a15541ff" width="300"</td>
  </tr>
 </table>



