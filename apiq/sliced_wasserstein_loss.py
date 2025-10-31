import torch

def sliced_wasserstein_loss(y_fp, y_quant, n_projections=16, block_size=None, device='cuda'):
    """
    Compute Sliced-Wasserstein distance between FP and quantized outputs.
    
    Args:
        y_fp: [batch, seq_len, hidden_dim] Full-precision output
        y_quant: [batch, seq_len, hidden_dim] Quantized output
        n_projections: Number of random projections (default: 16)
        block_size: Size of each block. If None, use token-level (original behavior).
                   If specified, treats each block as one sample. (default: None)
        device: Device for computation
        
    Returns:
        Scalar tensor representing SW distance
    """
    batch_size, seq_len, hidden_dim = y_fp.shape
    
    if block_size is None:
        # Original token-level implementation: N = batch*seq_len, d = hidden_dim
        y_fp_flat = y_fp.reshape(-1, hidden_dim).float()
        y_quant_flat = y_quant.reshape(-1, hidden_dim).float()
    else:
        # Block-level implementation: N = batch*num_blocks, d = block_size*hidden_dim
        assert seq_len % block_size == 0, f"seq_len ({seq_len}) must be divisible by block_size ({block_size})"
        
        num_blocks = seq_len // block_size
        # Reshape: [batch, seq_len, hidden_dim] -> [batch, num_blocks, block_size, hidden_dim]
        #       -> [batch*num_blocks, block_size*hidden_dim]
        y_fp_flat = y_fp.reshape(batch_size, num_blocks, block_size * hidden_dim).reshape(-1, block_size * hidden_dim).float()
        y_quant_flat = y_quant.reshape(batch_size, num_blocks, block_size * hidden_dim).reshape(-1, block_size * hidden_dim).float()
    
    # y_fp_flat: [N, proj_dim], y_quant_flat: [N, proj_dim]
    N, proj_dim = y_fp_flat.shape

    # Sample all projection vectors at once: [proj_dim, n_projections]
    u = torch.randn(proj_dim, n_projections, device=device, dtype=torch.float32)
    u = u / (u.norm(dim=0, keepdim=True) + 1e-8)  # L2-normalize each projection vector

    # Project both FP and quant outputs onto all random directions in parallel
    # proj_fp / proj_quant: [N, n_projections]
    proj_fp = y_fp_flat @ u
    proj_quant = y_quant_flat @ u

    # Sort along the sample dimension (dim=0) for each projection independently
    # sorted_fp / sorted_quant: [N, n_projections]
    sorted_fp, _ = torch.sort(proj_fp, dim=0)
    sorted_quant, _ = torch.sort(proj_quant, dim=0)

    # Compute 1D Wasserstein-1 distance for each projection:
    # take elementwise |diff|, mean over samples N -> [n_projections]
    w1_per_proj = torch.abs(sorted_fp - sorted_quant).mean(dim=0)

    # Final SW distance is the mean across projections -> scalar
    return w1_per_proj.mean()