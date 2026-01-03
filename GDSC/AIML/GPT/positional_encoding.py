"""
Positional Encoding Module for Transformer Architecture

This module implements sinusoidal positional encodings as described in
"Attention is All You Need" (Vaswani et al., 2017).

Mathematical Foundation:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
Where:
    - pos: position in the sequence (0, 1, 2, ..., max_len-1)
    - i: dimension index (0, 1, 2, ..., d_model/2 - 1)
    - d_model: embedding dimension

Key Properties:
    1. Each position gets a unique encoding
    2. Values are bounded between -1 and 1
    3. Relative positions can be represented as linear transformations
    4. Can extrapolate to unseen sequence lengths

Usage:
    >>> pe = PositionalEncoding(d_model=512, max_len=1024)
    >>> x = torch.randn(batch_size, seq_len, d_model)
    >>> x = pe(x)
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    
    Injects position information into token embeddings using sine and cosine
    functions of different frequencies.
    
    Args:
        d_model (int): The dimension of the embeddings (must match input)
        max_len (int): Maximum sequence length to pre-compute encodings for
        dropout (float): Dropout probability applied after adding positional encoding
    
    Input:
        x: Tensor of shape (batch_size, seq_len, d_model)
        
    Output:
        Tensor of same shape with positional encoding added
    
    Example:
        >>> pe = PositionalEncoding(d_model=512, max_len=1024)
        >>> embeddings = torch.randn(2, 100, 512)
        >>> output = pe(embeddings)
        >>> print(output.shape)
        torch.Size([2, 100, 512])
    """
    
    def __init__(
        self, 
        d_model: int, 
        max_len: int = 5000, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = self._create_positional_encoding(d_model, max_len)
        
        # Register as buffer (not a parameter, doesn't require gradients)
        # This ensures it moves to GPU with the model
        self.register_buffer('pe', pe)
    
    def _create_positional_encoding(
        self, 
        d_model: int, 
        max_len: int
    ) -> torch.Tensor:
        """
        Create the positional encoding matrix.
        
        Implementation Details:
        1. Create position indices: [0, 1, 2, ..., max_len-1]
        2. Create dimension indices: [0, 2, 4, ..., d_model-2]
        3. Compute division term using log-space for numerical stability
        4. Apply sin to even dimensions, cos to odd dimensions
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            
        Returns:
            Tensor of shape (1, max_len, d_model)
        """
        # Step 1: Create position indices as column vector
        # Shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Step 2 & 3: Create division term using log-space for numerical stability
        # div_term = 1 / (10000^(2i/d_model))
        # Using exp(log(x)) = x and log(a^b) = b*log(a):
        # = exp(-2i * log(10000) / d_model)
        # Shape: (d_model/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            (-math.log(10000.0) / d_model)
        )
        
        # Initialize PE matrix
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Step 4: Apply sine and cosine
        # angles = position * div_term has shape (max_len, d_model/2)
        # via broadcasting: (max_len, 1) * (d_model/2,) = (max_len, d_model/2)
        angles = position * div_term
        
        # Even dimensions get sine
        pe[:, 0::2] = torch.sin(angles)
        
        # Odd dimensions get cosine
        pe[:, 1::2] = torch.cos(angles)
        
        # Add batch dimension
        # Shape: (max_len, d_model) -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor of same shape with positional encoding added
            
        Raises:
            RuntimeError: If seq_len exceeds max_len
        """
        seq_len = x.size(1)
        
        if seq_len > self.max_len:
            raise RuntimeError(
                f"Sequence length ({seq_len}) exceeds maximum length ({self.max_len})"
            )
        
        # Add positional encoding
        # self.pe[:, :seq_len, :] selects positions 0 to seq_len-1
        # Broadcasting handles the batch dimension
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)
    
    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Get the positional encoding for a given sequence length.
        
        Useful for visualization or debugging.
        
        Args:
            seq_len: Length of sequence
            
        Returns:
            Tensor of shape (seq_len, d_model)
        """
        return self.pe[0, :seq_len, :].clone()


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding (as used in GPT-2).
    
    Unlike sinusoidal encoding, this learns a separate embedding for each
    position from data. Each position gets a trainable d_model-dimensional
    vector.
    
    Args:
        d_model (int): The dimension of the embeddings
        max_len (int): Maximum sequence length
        dropout (float): Dropout probability
        
    Pros:
        - Can learn task-specific position patterns
        - Simple implementation
        
    Cons:
        - Cannot extrapolate beyond max_len
        - More parameters to train
    """
    
    def __init__(
        self, 
        d_model: int, 
        max_len: int = 1024, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Learnable position embeddings
        # Shape: (max_len, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize the embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize position embeddings with small random values."""
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor of same shape with positional encoding added
        """
        seq_len = x.size(1)
        
        if seq_len > self.max_len:
            raise RuntimeError(
                f"Sequence length ({seq_len}) exceeds maximum length ({self.max_len})"
            )
        
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(0, seq_len, device=x.device)
        
        # Get position embeddings and add to input
        position_emb = self.position_embeddings(positions)  # (seq_len, d_model)
        x = x + position_emb.unsqueeze(0)  # Broadcast over batch
        
        return self.dropout(x)


def visualize_positional_encoding(
    d_model: int = 128, 
    max_len: int = 100
) -> None:
    """
    Visualize sinusoidal positional encodings.
    
    Creates a heatmap showing the positional encoding values
    for different positions and dimensions.
    
    Args:
        d_model: Embedding dimension
        max_len: Number of positions to show
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib and numpy required for visualization")
        return
    
    pe = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
    encoding = pe.get_encoding(max_len).numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap of full encoding
    im = axes[0].imshow(encoding.T, aspect='auto', cmap='RdBu')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Dimension')
    axes[0].set_title('Positional Encoding Heatmap')
    plt.colorbar(im, ax=axes[0])
    
    # Line plot of specific dimensions
    dims_to_show = [0, 1, 4, 5, 20, 21, 50, 51]
    for dim in dims_to_show:
        if dim < d_model:
            label = f'dim {dim} ({"sin" if dim % 2 == 0 else "cos"})'
            axes[1].plot(encoding[:, dim], label=label, alpha=0.7)
    
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Encoding Value')
    axes[1].set_title('Positional Encoding by Dimension')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('positional_encoding_visualization.png', dpi=150)
    plt.show()


# Example Usage
if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 10, 512
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 1. Sinusoidal Positional Encoding (Original Transformer)
    pe = PositionalEncoding(d_model=d_model, max_len=1024, dropout=0.0)
    output = pe(x)
    print(f"Input shape: {x.shape}")
    print(f"Sinusoidal PE Output: {output.shape}")
    print(f"PE buffer shape: {pe.pe.shape}")
    
    # Get raw encoding for visualization
    encoding = pe.get_encoding(seq_len)
    print(f"Position 0 encoding (first 4 dims): {encoding[0, :4].tolist()}")
    print(f"Position 1 encoding (first 4 dims): {encoding[1, :4].tolist()}")
    
    # Verify values are bounded [-1, 1]
    print(f"Encoding range: [{encoding.min():.3f}, {encoding.max():.3f}]")
    
    # 2. Learned Positional Encoding (GPT-2 style)
    learned_pe = LearnedPositionalEncoding(d_model=d_model, max_len=1024, dropout=0.0)
    output = learned_pe(x)
    print(f"\nLearned PE Output: {output.shape}")
    print(f"Learnable parameters: {sum(p.numel() for p in learned_pe.parameters()):,}")
    
    # 3. Verify position encoding is added correctly
    x_zeros = torch.zeros(1, 5, d_model)
    pe_only = pe(x_zeros)  # Should just be the positional encoding
    print(f"\nPE-only output matches buffer: {torch.allclose(pe_only[0], pe.pe[0, :5], atol=1e-6)}")


