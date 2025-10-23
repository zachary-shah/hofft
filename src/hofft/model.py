import torch
import numpy as np

from mr_recon.linops import linop, batching_params
from mr_recon.utils import gen_grd, batch_iterator, resize
from mr_recon.fourier import fft, ifft
from mr_recon.indexing import multi_index, multi_grid
from mr_recon.pad import PadLast
from .kb import _gen_kern_vectors

from typing import Optional, Union, Sequence, Tuple
from einops import einsum, rearrange
from dataclasses import dataclass

__all__ = [
    'multi_apod_kern_linop', 
    'multi_apod_kern_linop_batch',
    'hofft_params'
]

@dataclass
class hofft_params:
    kern_size: tuple
    os: Union[float, Sequence[float]]
    L: int
    apods_init: Union[torch.Tensor, str] = 'seg'
    use_type3: bool = False
    verbose: bool = True
    check_convergence: bool = True
    """
    Parameters for HOFFT models.
    
    Attributes
    ----------
    kern_size : tuple
        Size of the kernel, must have the same number of dimensions as the image.
    os : Union[float, Sequence[float]]
        Oversampling factor. If a float, uses the same oversampling factor for all dimensions.
        If a sequence, must have the same length as the number of dimensions in the image.
    L : Optional[int]
        Number of apodization functions.
    apods_init : Union[torch.Tensor, str]
        If string:
        'seg' - uses segmentation method to initialize apodization functions
        'eigen' - uses eigen-decomposition method to initialize apodization functions
        'k_alphas' - uses K representative alphas to initialize apodization functions
        If torch.Tensor:
        Initial apodization functions with shape (L, *solve_size)
    use_type3 : Optional[bool]
        If True, uses type3 nufft for the forward and adjoint operations.
    verbose : Optional[bool]
        If True, prints progress
    """

class multi_apod_kern_linop(linop):
    """
    Linear operator for multi-apodized kernels.
    """

    def __init__(self, 
                 trj: torch.Tensor,
                 mps: torch.Tensor,
                 weights: torch.Tensor,
                 apods: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 noise_cov: Optional[torch.Tensor] = None,
                 os_grid: Optional[Union[float, Sequence[float]]] = 1.0,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Initialize the HOFFT linear operator.
        
        Args:
        -----
        trj : torch.Tensor
            Trajectory of the k-space samples with shape (*trj_size, D)
        mps : torch.Tensor
            Sensitivity maps with shape (C, *im_size)
        weights : torch.Tensor
            the kernel weights with shape (L, *kern_size, *trj_size)
        apods : torch.Tensor
            the apodization functions with shape (L, *im_size)
        dcf : Optional[torch.Tensor]
            Density compensation function with shape (*trj_size)
        noise_cov : torch.tensor
            the k-space noise covariance matrix with shape (*trj_size, nc, nc)
        os_grid : Optional[float]
            Oversampling factor for the grid
            Can also be a sequence of floats for each dimension
        bparams : Optional[batching_params]
            Batching parameters for the linear operator
        """
        im_size = mps.shape[1:]
        assert all([im_size[i] % 2 == 0 for i in range(len(im_size))]), \
            f"Image size must be even in all dimensions for HOFFT. im_size: {im_size}"
        trj_size = trj.shape[:-1]
        kern_size = weights.shape[1:-len(trj_size)]
        oshape = (mps.shape[0], *trj_size)
        super().__init__(im_size, oshape)
        
        # Consts
        D = trj.shape[-1]
        L = weights.shape[0]
        torch_dev = trj.device
        assert mps.device == torch_dev
        assert weights.device == torch_dev
        assert apods.device == torch_dev
        assert apods.shape[0] == L
        
        # Make sure trajectory is on an oversampled grid
        if isinstance(os_grid, (int, float)):
            assert torch.allclose(trj, (trj * os_grid).round() / os_grid), \
                f"Trajectory is not on an oversampled grid. os_grid: {os_grid}"
            os_grid = [os_grid] * D
        else:
            assert len(os_grid) == D, f"os_grid must have length {D} for {D}-D trajectory."
            for i in range(D):
                assert torch.allclose(trj[..., i], (trj[..., i] * os_grid[i]).round() / os_grid[i]), \
                    f"Trajectory is not on an oversampled grid in dimension {i}. os_grid: {os_grid}"
            
        # Default dcf
        if dcf is None:
            dcf = torch.ones(trj.shape[:-1], dtype=torch.float32, device=torch_dev)
        else:
            assert dcf.device == torch_dev
        
        # Trajectory of kernels
        if np.prod(kern_size) == 1:
            kern_vecs = torch.zeros(D, device=torch_dev)
        else:
            kern_vecs = gen_grd(kern_size, kern_size).reshape((-1, D)).to(torch_dev)
        
        # Set noise covariance
        if noise_cov is None:
            self.inv_noise_cov = None
        else:
            self.inv_noise_cov = torch.linalg.inv(noise_cov)

        im_size_os = [round(im_size[i] * os_grid[i]) for i in range(D)]
        im_size_os_tensor = torch.tensor(im_size_os, device=torch_dev)
        os_grid_tensor = torch.tensor(os_grid, device=torch_dev)
        idx_kerns = (einsum(trj, os_grid_tensor, "... d, d -> ... d")).round() + im_size_os_tensor // 2
        idx_kerns = (idx_kerns[..., None, :] + kern_vecs).type(torch.int32) # (*trj_size, K, d)
        idx_kerns = idx_kerns % im_size_os_tensor.type(torch.int32)
        
        # Store params
        self.padder = PadLast(im_size_os, list(im_size))
        self.im_size_os = im_size_os
        self.im_size = im_size
        self.mps = mps
        self.os_grid = os_grid
        self.dcf = dcf
        self.idx_kerns = idx_kerns
        self.bparams = bparams
        self.weights = weights.reshape((L, -1, *trj_size))
        self.apods = apods

    def apply_kspace_coil_mat(self,
                              ksp: torch.Tensor,
                              ksp_coil_mat: torch.Tensor) -> torch.Tensor:
        """
        Applies a coil matrix to each point in kspace

        currently copied from sense_img_batch.py, need to improve later.
        
        Parameters
        ----------
        ksp : torch.tensor
            the k-space data with shape (N, C, *trj_size)
        ksp_coil_mat : torch.tensor 
            the coil matrix with shape (N, *trj_size, C, C)
        
        Returns
        ---------
        ksp_new : torch.tensor
            the k-space data with shape (N, C, *trj_size) after applying the coil matrix
        """
        ksp_new = torch.zeros_like(ksp)
        C = ksp.shape[1]
        first_coil_batch = C
        for c1, c2 in batch_iterator(C, first_coil_batch):
            second_coil_batch = C
            for d1, d2 in batch_iterator(C, second_coil_batch):
                ksp_new[:, c1:c2] = einsum(ksp[:, d1:d2], ksp_coil_mat[..., c1:c2, d1:d2], 'N ci ..., N ... co ci -> N co ...')
        return ksp_new  
    
    def forward(self,
                img: torch.Tensor) -> torch.Tensor:
        """
        Applies forward model to image to get k-space data.
        
        Parameters
        ----------
        img : torch.Tensor
            The image to be transformed with shape (*im_size)
        
        Returns
        -------
        torch.Tensor
            The k-space data with shape (C, *trj_size)
        """
        # Consts
        D = self.idx_kerns.shape[-1]
        C = self.mps.shape[0]
        cbs = self.bparams.coil_batch_size
        
        # Output tensor
        ksp = torch.zeros(self.oshape, device=img.device, dtype=torch.complex64)
        
        # Batch over coils
        for c1, c2 in batch_iterator(C, cbs):
            
            # Apply sensitivity maps to image
            Sx = self.mps[c1:c2] * img
            
            # Apply apods to image
            MSx = einsum(Sx, self.apods,
                         'C ..., L ... -> C L ...')
            
            # Oversampled FFT
            MSx = self.padder(MSx)
            FMSx = fft(MSx, dim=tuple(range(-D, 0)))
            
            # Extract blocks of k-space data
            blocks = multi_index(FMSx, D, self.idx_kerns) # (C, L, *trj_size, K)
            blocks = blocks.moveaxis(-1, 2) # (C, L, K, *trj_size)
            
            # Apply kernels
            KFSx = einsum(blocks, self.weights, 'C L K ..., L K ... -> C ...')
            ksp[c1:c2] = KFSx
            
        return ksp
    
    def adjoint(self,
                ksp: torch.Tensor) -> torch.Tensor:
        """
        Applies adjoint model to k-space data to get image.
        
        Parameters
        ----------
        ksp : torch.Tensor
            The k-space data with shape (C, *trj_size)
        
        Returns
        -------
        torch.Tensor
            The image with shape (*im_size)
        """
        # Consts
        D = self.idx_kerns.shape[-1]
        C = self.mps.shape[0]
        cbs = self.bparams.coil_batch_size
        
        # Output tensor
        img = torch.zeros(self.ishape, device=ksp.device, dtype=torch.complex64)
        
        # noise cov
        if self.inv_noise_cov is not None:
            ksp = self.apply_kspace_coil_mat(ksp[None,], self.inv_noise_cov[None,])[0]

        # Batch over coils
        for c1, c2 in batch_iterator(C, cbs):

            # Get Kernels
            y = ksp[c1:c2] * self.dcf
            Ky = einsum(y, self.weights.conj(), 'C ..., L K ... -> C L ... K')
            
            # Gridding 
            Ky = multi_grid(Ky, self.idx_kerns, self.im_size_os) # (C, L, *im_size_os)
            FKy = ifft(Ky, dim=tuple(range(-D, 0)))
            FKy = self.padder.adjoint(FKy) # (C, L, *im_size)
            
            # Apply adjoint sensitivity maps
            SFKy = (self.mps[c1:c2, None,].conj() * FKy).sum(dim=0) # L, *im_size
            
            # Apply adjoint source maps
            MSFKy = (SFKy * self.apods.conj()).sum(dim=0)
            
            # Update image
            img += MSFKy
        
        return img
    
    def normal(self,
               img: torch.Tensor) -> torch.Tensor:
        """
        Applies forward model and adjoint model to image to get normal operator.
        
        Parameters
        ----------
        img : torch.Tensor
            The image to be transformed with shape (*im_size)
            
        Returns
        -------
        torch.Tensor
            The response image with shape (*im_size)
        """
        
        return self.adjoint(self.forward(img))




class multi_apod_kern_linop_batch(linop):
    """
    Linear operator for multi-apodized kernels, with batching over N images.
    """

    def __init__(self, 
                 trj: torch.Tensor,
                 mps: torch.Tensor,
                 weights: torch.Tensor,
                 apods: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 noise_cov: Optional[torch.Tensor] = None,
                 inv_nc_lr: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 os_grid: Optional[Union[float, Sequence[float]]] = 1.0,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Initialize the HOFFT linear operator.
        
        Args:
        -----
        trj : torch.Tensor
            Trajectory of the k-space samples with shape (*trj_size, D)
        mps : torch.Tensor
            Sensitivity maps with shape (C, *im_size)
        weights : torch.Tensor
            the kernel weights with shape (L, *kern_size, *trj_size)
        apods : torch.Tensor
            the apodization functions with shape (L, *im_size)
        dcf : Optional[torch.Tensor]
            Density compensation function with shape (*trj_size)
        noise_cov : torch.tensor
            the k-space noise covariance matrix with shape (N, *trj_size, nc, nc)
        inv_nc_lr : Optional[Tuple[torch.Tensor, torch.Tensor]]
            Inverse noise covariance in a low-rank approximation of size (N, *trj_size, L) and (N, L, nc, nc)
        os_grid : Optional[float]
            Oversampling factor for the grid
            Can also be a sequence of floats for each dimension
        bparams : Optional[batching_params]
            Batching parameters for the linear operator
        """
        im_size = mps.shape[1:]
        assert all([im_size[i] % 2 == 0 for i in range(len(im_size))]), \
            f"Image size must be even in all dimensions for HOFFT. im_size: {im_size}"
        trj_size = trj.shape[:-1]
        kern_size = weights.shape[1:-len(trj_size)]
        oshape = (mps.shape[0], *trj_size)
        super().__init__((1, *im_size), oshape)
        
        # Consts
        D = trj.shape[-1]
        L = weights.shape[0]
        torch_dev = trj.device
        assert mps.device == torch_dev
        assert weights.device == torch_dev
        assert apods.device == torch_dev
        assert apods.shape[0] == L
        
        # Make sure trajectory is on an oversampled grid
        if isinstance(os_grid, (int, float)):
            assert torch.allclose(trj, (trj * os_grid).round() / os_grid), \
                f"Trajectory is not on an oversampled grid. os_grid: {os_grid}"
            os_grid = [os_grid] * D
        else:
            assert len(os_grid) == D, f"os_grid must have length {D} for {D}-D trajectory."
            for i in range(D):
                assert torch.allclose(trj[..., i], (trj[..., i] * os_grid[i]).round() / os_grid[i]), \
                    f"Trajectory is not on an oversampled grid in dimension {i}. os_grid: {os_grid}"
            
        # Default dcf
        if dcf is None:
            dcf = torch.ones(trj.shape[:-1], dtype=torch.float32, device=torch_dev)
        else:
            assert dcf.device == torch_dev
        
        # Trajectory of kernels
        if np.prod(kern_size) == 1:
            kern_vecs = torch.zeros(D, device=torch_dev)
        else:
            kern_vecs = gen_grd(kern_size, kern_size).reshape((-1, D)).to(torch_dev)
        
        # Set noise covariance
        self.set_noise_cov(noise_cov, inv_nc_lr)

        im_size_os = [round(im_size[i] * os_grid[i]) for i in range(D)]
        im_size_os_tensor = torch.tensor(im_size_os, device=torch_dev)
        os_grid_tensor = torch.tensor(os_grid, device=torch_dev)
        idx_kerns = (einsum(trj, os_grid_tensor, "... d, d -> ... d")).round() + im_size_os_tensor // 2
        idx_kerns = (idx_kerns[..., None, :] + kern_vecs).type(torch.int32) # (*trj_size, K, d)
        idx_kerns = idx_kerns % im_size_os_tensor.type(torch.int32)
        
        # Store params
        self.padder = PadLast(im_size_os, list(im_size))
        self.im_size_os = im_size_os
        self.im_size = im_size
        self.mps = mps
        self.os_grid = os_grid
        self.dcf = dcf
        self.idx_kerns = idx_kerns
        self.bparams = bparams
        self.weights = weights.reshape((L, -1, *trj_size))
        self.apods = apods

    def set_noise_cov(self, noise_cov, inv_nc_lr):
        """
        Either compute inverse noise cov, or set from low-rank factors.
        """
        self.noisecovmode = None
        self.inv_noise_cov = None
        self.inc_temporal = None
        self.inc_spatial = None
        if inv_nc_lr is not None:
            assert noise_cov is None, "Only supply noise_cov or inc_nc_lr, not both."
            # dont pre-compute for memory purposes
            self.inc_temporal = inv_nc_lr[0]
            self.inc_spatial = inv_nc_lr[1]
            self.noisecovmode = "lr"
        elif noise_cov is not None:
            self.inv_noise_cov = torch.zeros_like(noise_cov)
            for i in range(noise_cov.shape[0]):
                self.inv_noise_cov[i] = torch.linalg.inv(noise_cov[i])
            self.noisecovmode = "full"
            
    def apply_kspace_coil_mat(self,
                              ksp: torch.Tensor,
                              n1: int, n2: int) -> torch.Tensor:
        """
        Applies a coil matrix to each point in kspace

        currently copied from sense_img_batch.py, need to improve later.
        
        Parameters
        ----------
        ksp : torch.tensor
            the k-space data with shape (N, C, *trj_size)
        n1: int
            start index along N
        n2 : int
            end along N
        
        Returns
        ---------
        ksp_new : torch.tensor
            the k-space data with shape (N, C, *trj_size) after applying the coil matrix
        """
        if self.noisecovmode is None:
            return ksp

        nslc = slice(n1, n2)

        ksp_new = torch.zeros_like(ksp)
        C = ksp.shape[1]
        first_coil_batch = C

        assert self.noisecovmode in ["full", "lr"], "noise cov mode not set properly."
        
        for c1, c2 in batch_iterator(C, first_coil_batch):
            second_coil_batch = C
            for d1, d2 in batch_iterator(C, second_coil_batch):
                if self.noisecovmode == "full":
                    # pre-computed
                    kmat_batch = self.inv_noise_cov[nslc, ..., c1:c2, d1:d2]
                else:
                    # mem efficient version
                    kmat_batch = einsum(
                        self.inc_temporal[nslc], 
                        self.inc_spatial[nslc, ..., c1:c2, d1:d2],
                        "N ... L, N L c1 c2 -> N ... c1 c2"
                    )
                ksp_new[:, c1:c2] = einsum(ksp[:, d1:d2], kmat_batch, 'N ci ..., N ... co ci -> N co ...')

        return ksp_new  
    
    def forward(self,
                img: torch.Tensor) -> torch.Tensor:
        """
        Applies forward model to image to get k-space data.
        
        Parameters
        ----------
        img : torch.Tensor
            The image to be transformed with shape (N, *im_size)
        
        Returns
        -------
        torch.Tensor
            The k-space data with shape (N, C, *trj_size)
        """
        # Consts
        N = img.shape[0]
        D = self.idx_kerns.shape[-1]
        C = self.mps.shape[0]
        cbs = self.bparams.coil_batch_size
        ibs = self.bparams.img_batch_size

        # Output tensor
        ksp = torch.zeros((N, *self.oshape), device=img.device, dtype=torch.complex64)
        
        # batch over images
        for n1, n2 in batch_iterator(N, ibs):

            # Batch over coils
            for c1, c2 in batch_iterator(C, cbs):
                
                # Apply sensitivity maps to image
                Sx = self.mps[None, c1:c2] * img[n1:n2, None]
                
                # Apply apods to image
                MSx = einsum(Sx, self.apods, 'N C ..., L ... -> N C L ...')
                
                # Oversampled FFT
                MSx = self.padder(MSx)
                FMSx = fft(MSx, dim=tuple(range(-D, 0)))
                
                # Extract blocks of k-space data
                blocks = multi_index(FMSx, D, self.idx_kerns) # (C, L, *trj_size, K)
                blocks = blocks.moveaxis(-1, 3) # (N, C, L, K, *trj_size)
                
                # Apply kernels
                KFSx = einsum(blocks, self.weights, 'N C L K ..., L K ... -> N C ...')
                ksp[n1:n2, c1:c2] = KFSx
                
        return ksp
    
    def adjoint(self,
                ksp: torch.Tensor) -> torch.Tensor:
        """
        Applies adjoint model to k-space data to get image.
        
        Parameters
        ----------
        ksp : torch.Tensor
            The k-space data with shape (N, C, *trj_size)
        
        Returns
        -------
        torch.Tensor
            The image with shape (N, *im_size)
        """
        # Consts
        N = ksp.shape[0]
        D = self.idx_kerns.shape[-1]
        C = self.mps.shape[0]
        cbs = self.bparams.coil_batch_size
        ibs = self.bparams.img_batch_size

        # Output tensor
        img = torch.zeros((N, *self.ishape[1:]), device=ksp.device, dtype=torch.complex64)

        # Batch over coils
        for n1, n2 in batch_iterator(N, ibs):
            # noise cov
            # if self.inv_noise_cov is not None:
            #     kspb = self.apply_kspace_coil_mat(ksp[n1:n2], self.inv_noise_cov[n1:n2])
            # else:
            #     kspb = ksp[n1:n2]

            kspb = self.apply_kspace_coil_mat(ksp[n1:n2], n1, n2)
                
            for c1, c2 in batch_iterator(C, cbs):

                # Get Kernels
                y = kspb[:, c1:c2] * self.dcf
                Ky = einsum(y, self.weights.conj(), 'N C ..., L K ... -> N C L ... K')
                
                # Gridding 
                Ky = multi_grid(Ky, self.idx_kerns, self.im_size_os) # (N, C, L, *im_size_os)
                FKy = ifft(Ky, dim=tuple(range(-D, 0)))
                FKy = self.padder.adjoint(FKy) # (N, C, L, *im_size)
                
                # Apply adjoint sensitivity maps
                SFKy = (self.mps[None, c1:c2, None,].conj() * FKy).sum(dim=1) # N, L, *im_size
                
                # Apply adjoint source maps
                MSFKy = (SFKy * self.apods.conj()[None,]).sum(dim=1)
                
                # Update image
                img[n1:n2] += MSFKy
            
        return img
    
    def normal(self,
               img: torch.Tensor) -> torch.Tensor:
        """
        Applies forward model and adjoint model to image to get normal operator.
        
        Parameters
        ----------
        img : torch.Tensor
            The image to be transformed with shape (*im_size)
            
        Returns
        -------
        torch.Tensor
            The response image with shape (*im_size)
        """
        
        return self.adjoint(self.forward(img))





class multi_apod_kern_linop_multishot(linop):
    """
    Linear operator for multi-apodized kernels.

    With explicit modeling of shot-to-shot phase variations from different interleaves.
    """

    def __init__(self, 
                 trj: torch.Tensor,
                 phi: torch.Tensor,
                 mps: torch.Tensor,
                 weights: torch.Tensor,
                 apods: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 os_grid: Optional[Union[float, Sequence[float]]] = 1.0,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Initialize the HOFFT linear operator.
        T: timepoints
        P: number of shots/interleaves

        Args:
        -----
        trj : torch.Tensor
            Trajectory of the k-space samples with shape (T, P, D)
        phi: torch.Tensor
            Shot-to-shot phase variations of shape (P, *im_size)
        mps : torch.Tensor
            Sensitivity maps with shape (C, *im_size)
        weights : torch.Tensor
            the kernel weights with shape (L, *kern_size, T, P)
        apods : torch.Tensor
            the apodization functions with shape (L, *im_size)
        dcf : Optional[torch.Tensor]
            Density compensation function with shape (T, P)
        os_grid : Optional[float]
            Oversampling factor for the grid
            Can also be a sequence of floats for each dimension
        bparams : Optional[batching_params]
            Batching parameters for the linear operator
        """
        im_size = mps.shape[1:]
        assert all([im_size[i] % 2 == 0 for i in range(len(im_size))]), \
            f"Image size must be even in all dimensions for HOFFT. im_size: {im_size}"
        trj_size = trj.shape[:-1]
        kern_size = weights.shape[1:-len(trj_size)]
        oshape = (mps.shape[0], *trj_size)
        super().__init__(im_size, oshape)
        
        # Consts
        D = trj.shape[-1]
        L = weights.shape[0]
        torch_dev = trj.device
        assert mps.device == torch_dev
        assert weights.device == torch_dev
        assert apods.device == torch_dev
        assert apods.shape[0] == L
        
        # Make sure trajectory is on an oversampled grid
        if isinstance(os_grid, (int, float)):
            assert torch.allclose(trj, (trj * os_grid).round() / os_grid), \
                f"Trajectory is not on an oversampled grid. os_grid: {os_grid}"
            os_grid = [os_grid] * D
        else:
            assert len(os_grid) == D, f"os_grid must have length {D} for {D}-D trajectory."
            for i in range(D):
                assert torch.allclose(trj[..., i], (trj[..., i] * os_grid[i]).round() / os_grid[i]), \
                    f"Trajectory is not on an oversampled grid in dimension {i}. os_grid: {os_grid}"
            
        # Default dcf
        if dcf is None:
            dcf = torch.ones(trj.shape[:-1], dtype=torch.float32, device=torch_dev)
        else:
            assert dcf.device == torch_dev
        
        # Trajectory of kernels
        if np.prod(kern_size) == 1:
            kern_vecs = torch.zeros(D, device=torch_dev)
        else:
            kern_vecs = gen_grd(kern_size, kern_size).reshape((-1, D)).to(torch_dev)
        
        im_size_os = [round(im_size[i] * os_grid[i]) for i in range(D)]
        im_size_os_tensor = torch.tensor(im_size_os, device=torch_dev)
        os_grid_tensor = torch.tensor(os_grid, device=torch_dev)
        idx_kerns = (einsum(trj, os_grid_tensor, "T P d, d -> T P d")).round() + im_size_os_tensor // 2
        idx_kerns = (idx_kerns[..., None, :] + kern_vecs).type(torch.int32) # (T, P, K, d)
        idx_kerns = idx_kerns % im_size_os_tensor.type(torch.int32)
        
        # Store params
        self.padder = PadLast(im_size_os, list(im_size))
        self.im_size_os = im_size_os
        self.im_size = im_size
        self.mps = mps[None, :] * phi[:, None] # (P, C, *im_size) # apply phase to sensitivity maps
        self.os_grid = os_grid
        self.dcf = dcf
        self.idx_kerns = idx_kerns
        self.bparams = bparams
        self.weights = weights.reshape((L, -1, *trj_size))
        self.apods = apods
        self.P = trj.shape[1]

    def forward(self,
                img: torch.Tensor) -> torch.Tensor:
        """
        Applies forward model to image to get k-space data.
        
        Parameters
        ----------
        img : torch.Tensor
            The image to be transformed with shape (*im_size)
        
        Returns
        -------
        torch.Tensor
            The k-space data with shape (C, *trj_size)
        """
        # Consts
        D = self.idx_kerns.shape[-1]
        C = self.mps.shape[0]
        cbs = self.bparams.coil_batch_size
        
        # Output tensor
        ksp = torch.zeros(self.oshape, device=img.device, dtype=torch.complex64)
        
        # Batch over coils
        for c1, c2 in batch_iterator(C, cbs):
            # Apply sensitivity maps to image
            Sx = self.mps[:, c1:c2] * img[None, None].abs() # (C, P, *im_size)
            
            # Apply apods to image
            MSx = einsum(Sx, self.apods, 'P C ..., L ... -> P C L ...')
            
            # Oversampled FFT per shot
            MSx = self.padder(MSx)

            for p in range(self.P):
                FMSx = fft(MSx[p], dim=tuple(range(-D, 0))) # (C, L, ...)
                
                # Extract blocks of k-space data
                blocks = multi_index(FMSx, D, self.idx_kerns[:, p]) # (C, L, T, K)
                blocks = blocks.moveaxis(-1, 2) # (C, L, K, T)
                
                # Apply kernels
                ksp[c1:c2, :, p] = einsum(blocks, self.weights[..., p], 'C L K ..., L K ... -> C ...')
            
        return ksp
    
    def adjoint(self,
                ksp: torch.Tensor) -> torch.Tensor:
        """
        Applies adjoint model to k-space data to get image.
        
        Parameters
        ----------
        ksp : torch.Tensor
            The k-space data with shape (C, *trj_size)
        
        Returns
        -------
        torch.Tensor
            The image with shape (*im_size)
        """
        # Consts
        D = self.idx_kerns.shape[-1]
        C = self.mps.shape[0]
        cbs = self.bparams.coil_batch_size
        
        # Output tensor
        img = torch.zeros(self.ishape, device=ksp.device, dtype=torch.complex64)
        
        # Batch over coils
        for c1, c2 in batch_iterator(C, cbs):
            for p in range(self.P):
                # Get Kernels
                y = ksp[c1:c2, :, p] * self.dcf[:, p]
                Ky = einsum(y, self.weights.conj()[..., p], 'C T, L K T -> C L T K')
                
                # Gridding 
                Ky = multi_grid(Ky, self.idx_kerns[:, p], self.im_size_os) # (C, L, *im_size_os)
                FKy = ifft(Ky, dim=tuple(range(-D, 0)))
                FKy = self.padder.adjoint(FKy) # (C, L, *im_size)
                
                # Apply adjoint sensitivity maps
                SFKy = (self.mps[p, c1:c2, None,].conj() * FKy).sum(dim=0) # L, *im_size
                
                # Apply adjoint source maps
                MSFKy = (SFKy * self.apods.conj()).sum(dim=0)
                
                # Update image
                img += MSFKy.abs()
        
        return img
    
    def normal(self,
               img: torch.Tensor) -> torch.Tensor:
        """
        Applies forward model and adjoint model to image to get normal operator.
        
        Parameters
        ----------
        img : torch.Tensor
            The image to be transformed with shape (*im_size)
            
        Returns
        -------
        torch.Tensor
            The response image with shape (*im_size)
        """
        
        return self.adjoint(self.forward(img))
