"""
Multi-Head Attention Module for Transformer Architecture

This module implements the Multi-Head Attention mechanism as described in
"Attention is All You Need" (Vaswani et al., 2017) and used in GPT-2.

Mathematical Foundation:

    Scaled Dot-Product Attention:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Multi-Head Attention:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
        where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)

Key Concepts:
    - Query (Q): "What am I looking for?"
    - Key (K): "What do I contain?"  
    - Value (V): "What information do I provide?"
    - Scaling by sqrt(d_k): Prevents softmax saturation for large dimensions
    - Multiple heads: Capture different types of relationships in parallel

Usage:
    >>> mha = MultiHeadAttention(d_model=512, num_heads=8)
    >>> x = torch.randn(batch_size, seq_len, d_model)
    >>> output = mha(x, x, x)  # Self-attention
    
    >>> # With causal mask for autoregressive generation
    >>> mask = create_causal_mask(seq_len)
    >>> output = mha(x, x, x, mask=mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal (lower triangular) attention mask.
    
    This mask ensures that position i can only attend to positions <= i,
    preventing information flow from future tokens (autoregressive property).
    
    Args:
        seq_len: Length of the sequence
        device: Device to create tensor on
        
    Returns:
        Tensor of shape (seq_len, seq_len) with 1s in lower triangle, 0s in upper
        
    Example:
        >>> mask = create_causal_mask(4)
        >>> print(mask)
        tensor([[1., 0., 0., 0.],
                [1., 1., 0., 0.],
                [1., 1., 1., 0.],
                [1., 1., 1., 1.]])
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.
    
    Formula:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        query: Queries of shape (..., seq_q, d_k)
        key: Keys of shape (..., seq_k, d_k)
        value: Values of shape (..., seq_k, d_v)
        mask: Optional mask of shape (..., seq_q, seq_k). 
              0 = masked position, 1 = unmasked position
        dropout: Optional dropout layer for attention weights
        
    Returns:
        output: Weighted sum of values, shape (..., seq_q, d_v)
        attention_weights: Attention probabilities, shape (..., seq_q, seq_k)
        
    Mathematical Breakdown:
        1. scores = Q @ K^T  →  How similar is each query to each key?
        2. scores / sqrt(d_k)  →  Prevent large values from saturating softmax
        3. mask + softmax  →  Convert to probabilities with future positions masked
        4. weights @ V  →  Weighted combination of values
    """
    d_k = query.size(-1)
    
    # Step 1: Compute raw attention scores
    # (..., seq_q, d_k) @ (..., d_k, seq_k) -> (..., seq_q, seq_k)
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Step 2: Scale by sqrt(d_k)
    # This keeps the variance of scores roughly unit, preventing
    # softmax from having extremely small gradients
    scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply mask (if provided)
    # Set masked positions to -inf so they become 0 after softmax
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 4: Convert to probabilities
    # Each row sums to 1
    attention_weights = F.softmax(scores, dim=-1)
    
    # Handle the case where a row is all -inf (becomes all nan after softmax)
    # This can happen if mask completely masks a query
    attention_weights = attention_weights.nan_to_num(0.0)
    
    # Optional: Apply dropout to attention weights
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # Step 5: Weighted sum of values
    # (..., seq_q, seq_k) @ (..., seq_k, d_v) -> (..., seq_q, d_v)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Runs h parallel attention operations ("heads") with different learned
    projections, then concatenates the results.
    
    Args:
        d_model (int): Total dimension of the model (must be divisible by num_heads)
        num_heads (int): Number of parallel attention heads
        dropout (float): Dropout probability for attention weights
        bias (bool): Whether to use bias in linear projections
        
    Attributes:
        d_k: Dimension per head = d_model // num_heads
        W_q, W_k, W_v: Linear layers for Q, K, V projections
        W_o: Output linear layer
        
    Input Shapes:
        query: (batch_size, seq_len_q, d_model)
        key: (batch_size, seq_len_k, d_model)
        value: (batch_size, seq_len_v, d_model)  [usually seq_len_k == seq_len_v]
        mask: (seq_len_q, seq_len_k) or (batch_size, 1, seq_len_q, seq_len_k)
        
    Output Shape:
        output: (batch_size, seq_len_q, d_model)
        
    Example:
        >>> mha = MultiHeadAttention(d_model=512, num_heads=8)
        >>> x = torch.randn(2, 10, 512)  # (batch, seq, d_model)
        >>> output = mha(x, x, x)  # Self-attention
        >>> print(output.shape)
        torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        # Validate dimensions
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        # Each projects from d_model to d_model (then reshaped to heads)
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization (optional)
        self._attention_weights = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection matrices with Xavier uniform."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Compute Multi-Head Attention.
        
        Args:
            query: Query tensor of shape (batch, seq_q, d_model)
            key: Key tensor of shape (batch, seq_k, d_model)
            value: Value tensor of shape (batch, seq_v, d_model)
            mask: Optional attention mask
            return_attention: If True, also return attention weights
            
        Returns:
            output: Attended values of shape (batch, seq_q, d_model)
            attention_weights: (optional) shape (batch, heads, seq_q, seq_k)
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # ================================================================
        # Step 1: Linear projections
        # ================================================================
        # (batch, seq, d_model) -> (batch, seq, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # ================================================================
        # Step 2: Reshape for multi-head attention
        # ================================================================
        # (batch, seq, d_model) -> (batch, seq, num_heads, d_k)
        # -> (batch, num_heads, seq, d_k)
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # ================================================================
        # Step 3: Prepare mask for multi-head
        # ================================================================
        if mask is not None:
            # Expand mask to (batch, 1, seq_q, seq_k) if needed
            if mask.dim() == 2:
                # (seq_q, seq_k) -> (1, 1, seq_q, seq_k)
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # (batch, seq_q, seq_k) -> (batch, 1, seq_q, seq_k)
                mask = mask.unsqueeze(1)
            # Broadcasting will handle the heads dimension
        
        # ================================================================
        # Step 4: Compute scaled dot-product attention
        # ================================================================
        # Q, K, V: (batch, heads, seq, d_k)
        # context: (batch, heads, seq_q, d_k)
        # attention_weights: (batch, heads, seq_q, seq_k)
        context, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        
        # Store for visualization if needed
        self._attention_weights = attention_weights.detach()
        
        # ================================================================
        # Step 5: Concatenate heads
        # ================================================================
        # (batch, heads, seq_q, d_k) -> (batch, seq_q, heads, d_k)
        context = context.transpose(1, 2)
        
        # (batch, seq_q, heads, d_k) -> (batch, seq_q, d_model)
        # .contiguous() ensures memory layout is contiguous for .view()
        context = context.contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # ================================================================
        # Step 6: Final output projection
        # ================================================================
        output = self.W_o(context)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get the attention weights from the last forward pass.
        
        Returns:
            Attention weights of shape (batch, num_heads, seq_q, seq_k)
            or None if forward hasn't been called
        """
        return self._attention_weights


class CausalSelfAttention(MultiHeadAttention):
    """
    Multi-Head Self-Attention with Causal Masking (GPT-2 style).
    
    This is a specialized version of MultiHeadAttention that:
    1. Uses the same input for Q, K, and V (self-attention)
    2. Automatically applies causal masking
    
    Used in decoder-only transformers like GPT-2 for autoregressive generation.
    
    Example:
        >>> attn = CausalSelfAttention(d_model=512, num_heads=8, max_seq_len=1024)
        >>> x = torch.randn(2, 100, 512)
        >>> output = attn(x)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__(d_model, num_heads, dropout, bias)
        
        # Pre-compute causal mask and register as buffer
        # Shape: (1, 1, max_seq_len, max_seq_len)
        mask = create_causal_mask(max_seq_len)
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
        self.register_buffer('causal_mask', mask)
        
        self.max_seq_len = max_seq_len
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Compute causal self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            return_attention: If True, also return attention weights
            
        Returns:
            output: Attended values of shape (batch, seq_len, d_model)
            attention_weights: (optional) shape (batch, heads, seq, seq)
        """
        seq_len = x.size(1)
        
        if seq_len > self.max_seq_len:
            raise RuntimeError(
                f"Sequence length ({seq_len}) exceeds maximum ({self.max_seq_len})"
            )
        
        # Extract the relevant portion of the pre-computed mask
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        # Self-attention: Q=K=V=x
        return super().forward(x, x, x, mask=mask, return_attention=return_attention)


def visualize_attention(
    attention_weights: torch.Tensor,
    tokens: Optional[list] = None,
    head_idx: int = 0
) -> None:
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Tensor of shape (batch, heads, seq_q, seq_k)
        tokens: Optional list of token strings for labels
        head_idx: Which attention head to visualize
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return
    
    # Get single batch, single head
    weights = attention_weights[0, head_idx].detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(weights, cmap='Blues')
    
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(f'Attention Weights (Head {head_idx})')
    
    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150)
    plt.show()


# Example Usage
if __name__ == "__main__":
    batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    print(f"Config: d_model={d_model}, num_heads={num_heads}, d_k={d_model // num_heads}")
    
    # 1. Multi-Head Self-Attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    output = mha(x, x, x)
    print(f"\nMHA Output: {output.shape}")
    print(f"MHA Parameters: {sum(p.numel() for p in mha.parameters()):,}")
    
    # 2. With Causal Mask
    mask = create_causal_mask(seq_len)
    output, attn_weights = mha(x, x, x, mask=mask, return_attention=True)
    print(f"\nCausal Mask shape: {mask.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Verify masking works (upper triangle should be zero)
    upper_triangle_max = attn_weights[0, 0].triu(diagonal=1).max().item()
    print(f"Upper triangle max (should be ~0): {upper_triangle_max:.6f}")
    
    # 3. Causal Self-Attention (GPT-2 style)
    csa = CausalSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=1024)
    output = csa(x)
    print(f"\nCausal SA Output: {output.shape}")
    
    # 4. Cross-Attention (encoder-decoder)
    encoder_out = torch.randn(batch_size, 20, d_model)
    decoder_in = torch.randn(batch_size, 15, d_model)
    cross_output = mha(decoder_in, encoder_out, encoder_out)
    print(f"\nCross-Attention: Q={decoder_in.shape}, K/V={encoder_out.shape} -> {cross_output.shape}")
    
    # 5. Verify attention weights sum to 1
    _, attn = mha(x, x, x, return_attention=True)
    row_sums = attn[0, 0].sum(dim=-1)
    print(f"\nAttention row sums - mean: {row_sums.mean():.4f}, std: {row_sums.std():.4f}")


