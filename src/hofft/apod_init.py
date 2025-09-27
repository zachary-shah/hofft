import torch
import numpy as np
from typing import Optional
from einops import einsum

from mr_recon.utils import gen_grd, quantize_data
from mr_recon.algs import eigen_decomp_operator
from mr_recon.imperfections.field import alpha_segementation
from mr_recon.linops import type3_nufft_naive, type3_nufft
from mr_recon.dtypes import complex_dtype

from hofft.als import als_iterations
from hofft.model import hofft_params

def pick_K_vectors(vectors: torch.Tensor,
                   K: int,
                   sigma: Optional[float] = 0.0,
                   method: Optional[str] = 'minmax') -> torch.Tensor:
    """
    Given N vectors, pick K represenative vectors that are far apart from each other.
    
    Args
    ----
    vectors : torch.Tensor
        Vectors to pick from with shape (N, d) 
    K : int
        Number of vectors to pick
    sigma : float
        Adds noise to N vectors to 'blur out' the distribution
    method : str
        'kmeans' will use k-means clustering to pick the vectors.
        'random' will pick K random vectors from the input.
        'minmax' will pick K vectors that are farthest apart from each other.
        'convhull' TODO need to implement this.
        
    Returns
    -------
    kvectors : torch.Tensor
        K vectors with shape (K, d)
    idxs : torch.Tensor
        Indices of the picked vectors in the original input with shape (K,) in [0, N)
    """
    # Consts
    N, d = vectors.shape
    assert N > K, f'N={N} is not greater than K={K}.'
    
    # Add noise
    vectors_noisy = vectors + torch.randn_like(vectors) * sigma
    
    # Kmeans clustering
    if method == 'kmeans':
        kvectors, idxs = quantize_data(data=vectors_noisy, K=K, method='cluster')
    elif method == 'random':
        idxs = torch.randperm(N)[:K]
        kvectors = vectors_noisy[idxs]
    elif method == 'minmax':
        picked = [torch.randint(0, N, (1,))]
        dist = torch.linalg.norm(vectors_noisy - vectors_noisy[picked], dim=-1)
        for _ in range(1, K):
            nxt = torch.argmax(dist)
            picked.append(nxt)
            dist = torch.minimum(dist, torch.linalg.norm(vectors_noisy - vectors_noisy[nxt], dim=-1))
        idxs = torch.tensor(picked, dtype=torch.long, device=vectors_noisy.device)
        kvectors = vectors_noisy[idxs]
    else:
        raise ValueError(f'Unknown method {method}.')
        
    return kvectors, idxs

def K_alphas_apod_init(phis: torch.Tensor,
                       alphas: torch.Tensor,
                       hparams: hofft_params,
                       method: str = 'minmax',
                       apod_init_method: str = 'eigen',
                       num_als_iter: int = 100,
                       check_convergence: bool = True,
                       verbose: bool = True,
                       K: int = 500,) -> torch.Tensor:
    """
    Initialize apodizations by running ALS on K representative alphas.
    
    Args
    ----
    phis : torch.Tensor
        spatial phase basis functions, shape (B, *im_size).
    alphas : torch.Tensor
        temporal phase basis functions, shape (B, *trj_size).
    hparams : hofft_params
        hofft parameters.
    method : str
        method to pick K representative alphas. Options are 'minmax' and 'kmeans', and 'random'
    apod_init_method : str
        method to initialize apodizations before ALS. Options are 'seg' and 'eigen
    num_als_iter : int
        number of ALS iterations to run.
    K : int
        number of representative alphas to pick.
        
    Returns
    -------
    apods : torch.Tensor
        initialized apodizations, shape (L, *im_size).
    """
    # Consts
    im_size = phis.shape[1:]
    torch_dev = phis.device
    B = phis.shape[0]
    d = len(im_size)
    use_type3 = hparams.use_type3
    kern_size = hparams.kern_size
    os = hparams.os
    
    # Prep ALS algorithm
    rs = gen_grd(im_size).to(torch_dev)
    kern = gen_grd(kern_size, kern_size).to(torch_dev)
    kern = kern.reshape((-1, d)) / os
    kern_bases = torch.exp(-2j * np.pi * einsum(kern, rs,
                                                'K d, ... d -> K ...'))
    
    # Use other apod_init functions to get initial apods
    if apod_init_method == 'seg':
        apods_init_init = alpha_seg_apod_init(phis, alphas, hparams)
    elif apod_init_method == 'eigen':
        apods_init_init = eigen_apod_init(phis, alphas, hparams)
    else:
        raise ValueError(f'Invalid apod_init_method {apod_init_method}. Supported methods are seg and eigen.')
    
    # Pick K representative alphas
    k_alphas, _ = pick_K_vectors(vectors=alphas.reshape((B,-1)).T, K=K, 
                                 sigma=0, method=method)
    k_alphas = k_alphas.T # shape (B, K)
    
    # Make type3 object using k_alphas
    if use_type3:
        t3n = type3_nufft(phis, k_alphas, use_toep=True)
    else:
        t3n = type3_nufft_naive(phis, k_alphas)
    
    # ALS agorithm
    _, apods = als_iterations(t3n, kern_bases, apods_init_init, 
                              max_iter=num_als_iter,
                              check_convergence=check_convergence,
                              verbose=verbose)
    
    return apods
    
def alpha_seg_apod_init(phis: torch.Tensor,
                         alphas: torch.Tensor,
                         hparams: hofft_params) -> torch.Tensor:
    """
    Initialize apodizations using alpha segmentation.
    
    Args
    ----
    phis : torch.Tensor
        spatial phase basis functions, shape (B, *im_size).
    alphas : torch.Tensor
        temporal phase basis functions, shape (B, *trj_size).
    hparams : hofft_params
        hofft parameters.
        
    Returns
    -------
    apods : torch.Tensor
        initialized apodizations, shape (L, *im_size).
    """
    L = hparams.L
    apods, _ = alpha_segementation(phis, alphas, L=L, L_batch_size=L, interp_type='zero', use_type3=False)
    return apods

def eigen_apod_init(phis: torch.Tensor,
                     alphas: torch.Tensor,
                     hparams: hofft_params) -> torch.Tensor:
    """
    Initialize apodizations using eigen-decomposition of the system matrix.
    
    Args
    ----
    phis : torch.Tensor
        spatial phase basis functions, shape (B, *im_size).
    alphas : torch.Tensor
        temporal phase basis functions, shape (B, *trj_size).
        
    Returns
    -------
    apods : torch.Tensor
        initialized apodizations, shape (L, *im_size).
    """
    # Consts
    im_size = phis.shape[1:]
    torch_dev = phis.device
    L = hparams.L
    use_type3 = hparams.use_type3
    
    # Make type3 object
    if use_type3:
        t3n = type3_nufft(phis, alphas, use_toep=True)
    else:
        t3n = type3_nufft_naive(phis, alphas)
    
    # Eigen-decomp
    x0 = torch.randn(im_size, dtype=complex_dtype, device=torch_dev)
    apods, _ = eigen_decomp_operator(t3n.normal, x0, num_eigen=L, 
                                     num_iter=15,
                                     lobpcg=True,
                                     largest=True)
    
    return apods

