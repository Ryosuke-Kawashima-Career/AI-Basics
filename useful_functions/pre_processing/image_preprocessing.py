import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# Preprocessing
def gcn(img):
    """
    Global Contrast Normalization - normalizes image contrast.
    
    Args:
        img: Input image tensor of shape (C, H, W)
            - C = 3 (RGB)
            - H = 32 (height)
            - W = 32 (width)
            - Total: 3,072 values
    
    Returns:
        normalized_img: Output image tensor of shape (C, H, W)
                       Same shape as input, but normalized
    """
    # Step 1: Flatten image to 1D
    # Input:  (C, H, W) = (3, 32, 32)
    # Output: (C*H*W,) = (3072,)
    flattened_img = torch.flatten(img)
    
    # Step 2: Compute global statistics
    # std_img: scalar (single value)
    # mean_img: scalar (single value)
    std_img = torch.std(flattened_img)
    mean_img = torch.mean(flattened_img)
    
    # Step 3: Normalize (broadcasting applies normalization to all elements)
    # (C, H, W) - scalar = (C, H, W)
    # (C, H, W) / scalar = (C, H, W)
    epsilon = 1e-5
    normalized_img = (img - mean_img) / (std_img + epsilon)
    
    # Return shape: (C, H, W) = (3, 32, 32)
    return normalized_img

def deprocess(img):
    """
    Min-Max Normalization - scales image to [0, 1] range.
    
    Args:
        img: Input image tensor of shape (C, H, W)
            - C = 3 (RGB)
            - H = 32 (height)
            - W = 32 (width)
    
    Returns:
        normalized_img: Output image tensor of shape (C, H, W)
                       Values in range [0, 1]
    """
    # Step 1: Flatten to compute global min/max
    # Input:  (C, H, W) = (3, 32, 32)
    # Output: (C*H*W,) = (3072,)
    flattened_img = torch.flatten(img)
    
    # Step 2: Find global min and max
    # max_img: scalar (single value)
    # min_img: scalar (single value)
    max_img = torch.max(flattened_img)
    min_img = torch.min(flattened_img)
    
    # Step 3: Min-Max Normalization (broadcasting)
    # (C, H, W) - scalar = (C, H, W)
    # (C, H, W) / scalar = (C, H, W)
    normalized_img = (img - min_img) / (max_img - min_img)
    
    # Return shape: (C, H, W) = (3, 32, 32)
    # Values now in range [0, 1]
    return normalized_img

def zca_whitening(images):
    """
    Zero-phase Component Analysis Whitening.
    
    Args:
        images: Batch of images, shape (N, C, H, W)
            - N = batch size
            - C = 3 (RGB channels)
            - H = 32 (height)
            - W = 32 (width)
    
    Returns:
        whitened_images: Whitened batch, shape (N, C, H, W)
        ZCA_matrix: Transformation matrix for future images, shape (C*H*W, C*H*W)
    """
    # Get dimensions
    N, C, H, W = images.shape
    # N = batch size (e.g., 1)
    # C = 3 (RGB)
    # H = 32
    # W = 32
    
    # Flatten each image to 1D
    # Input:  (N, C, H, W) = (N, 3, 32, 32)
    # Output: (N, C*H*W) = (N, 3072)
    images_flat = images.reshape(N, -1)
    
    # ===== Step 1: Compute Mean =====
    # Sum across all images in batch
    # (N, 3072) → (3072,) after mean
    mean_vec = images_flat.mean(dim=0, keepdim=True)  # Shape: (1, 3072)
    
    print(f"\nZCA Whitening Computation:")
    print(f"  Flattened images shape: {images_flat.shape}")  # (N, 3072)
    print(f"  Mean vector shape: {mean_vec.shape}")          # (1, 3072)
    
    # ===== Step 2: Center Data (subtract mean) =====
    # (N, 3072) - (1, 3072) = (N, 3072) [broadcasting]
    images_centered = images_flat - mean_vec
    
    print(f"  Centered images shape: {images_centered.shape}")  # (N, 3072)
    
    # ===== Step 3: Compute Covariance Matrix =====
    # Covariance = E[(X - μ)(X - μ)ᵀ]
    # Mathematical: (1/N) * X_centered^T @ X_centered
    #
    # X_centered shape: (N, 3072)
    # X_centered.T shape: (3072, N)
    # Result: (3072, N) @ (N, 3072) = (3072, 3072)
    cov_matrix = torch.mm(images_centered.T, images_centered) / (N - 1)
    
    print(f"  Covariance matrix shape: {cov_matrix.shape}")  # (3072, 3072)
    print(f"  Covariance matrix size: {3072*3072:,} elements")
    
    # ===== Step 4: Eigendecomposition =====
    # cov_matrix = V @ Λ @ V^T
    # where:
    #   V (eigenvectors) shape: (3072, 3072)
    #   Λ (eigenvalues) shape: (3072,)
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    except RuntimeError:
        print("  Warning: SVD used instead of eigh for numerical stability")
        U, S, Vt = torch.linalg.svd(cov_matrix)
        eigenvectors = U
        eigenvalues = S
    
    print(f"  Eigenvectors shape: {eigenvectors.shape}")  # (3072, 3072)
    print(f"  Eigenvalues shape: {eigenvalues.shape}")    # (3072,)
    
    # ===== Step 5: Compute ZCA Matrix =====
    # ZCA_matrix = V @ diag(1/√Λ) @ V^T
    # Step 5a: Compute 1/√eigenvalues
    # eigenvalues shape: (3072,)
    # pow(-0.5) computes element-wise: λᵢ^(-0.5) = 1/√λᵢ
    sqrt_inv_evals = eigenvalues.pow(-0.5)  # Shape: (3072,)
    
    print(f"  Sqrt inv eigenvalues shape: {sqrt_inv_evals.shape}")  # (3072,)
    
    # Step 5b: Create diagonal matrix from 1/√λ
    # diag converts vector to diagonal matrix
    # (3072,) → (3072, 3072)
    diag_matrix = torch.diag(sqrt_inv_evals)
    
    print(f"  Diagonal matrix shape: {diag_matrix.shape}")  # (3072, 3072)
    
    # Step 5c: Compute ZCA_matrix = V @ diag @ V^T
    # V shape: (3072, 3072)
    # diag shape: (3072, 3072)
    # V^T shape: (3072, 3072)
    # (3072, 3072) @ (3072, 3072) = (3072, 3072)
    # (3072, 3072) @ (3072, 3072) = (3072, 3072)
    ZCA_matrix = torch.mm(torch.mm(eigenvectors, diag_matrix), eigenvectors.T)
    
    print(f"  ZCA matrix shape: {ZCA_matrix.shape}")  # (3072, 3072)
    
    # ===== Step 6: Apply ZCA Transform =====
    # X_whitened = X_centered @ ZCA_matrix^T
    # X_centered shape: (N, 3072)
    # ZCA_matrix shape: (3072, 3072)
    # Result: (N, 3072) @ (3072, 3072) = (N, 3072)
    images_whitened_flat = torch.mm(images_centered, ZCA_matrix.T)
    
    print(f"  Whitened flat images shape: {images_whitened_flat.shape}")  # (N, 3072)
    
    # Reshape back to image dimensions
    # (N, 3072) → (N, C, H, W) = (N, 3, 32, 32)
    images_whitened = images_whitened_flat.reshape(N, C, H, W)
    
    print(f"  Whitened images shape: {images_whitened.shape}")  # (N, 3, 32, 32)
    
    return images_whitened, ZCA_matrix, mean_vec
