import torch
from fast_pytorch_kmeans import KMeans

def uniform_quantization(coeffs: torch.Tensor, 
                         grid_spacing: float,) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Uniform quantization of phase coefficients.
    
    Args
    ----
    coeffs : torch.Tensor
        phi or alpha phase coefficients with shape (B, N)
    grid_spacing : float
        Grid spacing in units of the coefficient units
        
    Returns
    -------
    coeffs_quant : torch.Tensor
        Smaller set of quantized coefficients (B, M), M <= N
    inds_quant : torch.Tensor
        Indices mapping from original coefficients to quantized coefficients with shape (N,) in [0, M)
    """
    # Consts
    assert coeffs.ndim == 2
    
    # Quantize coefficients
    coeffs_quant = (coeffs / grid_spacing).round() * grid_spacing
    coeffs_quant, inds_quant = coeffs_quant.unique(dim=1, return_inverse=True)
    
    return coeffs_quant, inds_quant

def kmeans_quantization(coeffs: torch.Tensor, 
                        K: int,
                        max_iter: int = 1000,
                        mode: str = 'euclidean') -> tuple[torch.Tensor, torch.Tensor]:
    """
    K-means quantization of phase coefficients.
    
    Args
    ----
    coeffs : torch.Tensor
        phi or alpha phase coefficients with shape (B, N)
    K : int
        Number of coefficients to quantize to
    max_iter : int
        Maximum number of iterations for K-means
    mode : str
        Mode for K-means clustering
        
    Returns
    -------
    coeffs_quant : torch.Tensor
        Smaller set of quantized coefficients (B, M), M <= N
    inds_quant : torch.Tensor
        Indices mapping from original coefficients to quantized coefficients with shape (N,) in [0, M)
    """
    # Consts
    assert coeffs.ndim == 2
    
    # Quantize data using K-means
    torch_dev = coeffs.device
    verbose = 0
    if (torch_dev.index == -1) or (torch_dev.index is None):
        kmeans = KMeans(n_clusters=K,
                        max_iter=max_iter,
                        verbose=verbose,
                        mode=mode)
        inds_quant = kmeans.fit_predict(coeffs.T)
    else:
        with torch.cuda.device(torch_dev):
            kmeans = KMeans(n_clusters=K,
                            max_iter=max_iter,
                            verbose=verbose,
                            mode=mode)
            inds_quant = kmeans.fit_predict(coeffs.T)
    coeffs_quant = kmeans.centroids.T

    
    return coeffs_quant, inds_quant