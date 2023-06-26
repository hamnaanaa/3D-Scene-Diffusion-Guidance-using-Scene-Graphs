import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfMultiheadAttention(nn.Module):
    """
    Multi-Head Attention Layer with Layer Normalization

    Args:
        input_dim (int): Dimension of the input matrix.
        embed_dim (int): Dimension of the embedding space.
        num_heads (int): Number of attention heads, must be a divisor of embed_dim.

    """
    def __init__(self, N, D, embed_dim, num_heads): # embed_dim = num_heads*D_weights!
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(D, 3*embed_dim, bias=False) # Linear layer
        self.o_proj = nn.Linear(embed_dim, D, bias=False)
        # Normalise over the last 2 dimensions
        self.layer_norm = nn.LayerNorm([N, D])

        self._reset_parameters()

    def _reset_parameters(self):
        # IS THIS NECESSARY???
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
    
    @staticmethod
    def scaled_dot_product(q, k, v):
        d_k = q.size()[-1]
        # Note: k[B, N, d_k], we transpose the matrix [N, d_k] for each item in B
        # Note: torch.matmul performs batchwise matrix multiplications
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) 
        attn_logits = attn_logits / np.sqrt(d_k)
        
        # Note: Probability Distribution over each q-vector
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values

        
    def forward(self, x):
        B, N, D = x.size() 
        
        device = x.device
        
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(B, N, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values = self.scaled_dot_product(q, k, v)  # Pass mask as a keyword argument
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(B, N, self.embed_dim)
        o = self.o_proj(values)
        
        # Move tensors to the GPU device
        o = o.to(device)
        self.layer_norm.to(device)
        
        out = self.layer_norm(o)

        return out
    
    
class CrossMultiheadAttention(nn.Module):
    """
    Multi-Head Attention Layer with Layer Normalization

    Args:
        input_dim (int): Dimension of the input matrix.
        embed_dim (int): Dimension of the embedding space.
        num_heads (int): Number of attention heads, must be a divisor of embed_dim.

    """
    def __init__(self, N, D, C, embed_dim, num_heads): # embed_dim = num_heads*D_weights!
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_proj = nn.Linear(D, embed_dim, bias=False) # Linear layer
        self.kv_proj = nn.Linear(C, 2*embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, D, bias=False)
        # Normalise over the last 2 dimensions
        self.layer_norm = nn.LayerNorm([N, D])

        self._reset_parameters()

    def _reset_parameters(self):
        # IS THIS NECESSARY???
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
    
    @staticmethod
    def scaled_dot_product(q, k, v):
        d_k = q.size()[-1]
        # Note: k[B, N, d_k], we transpose the matrix [N, d_k] for each item in B
        # Note: torch.matmul performs batchwise matrix multiplications
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) 
        attn_logits = attn_logits / np.sqrt(d_k)
        
        # Note: Probability Distribution over each q-vector
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values

        
    def forward(self, x, obj_cond):
        B, N, D = x.size()
        _, _, C = obj_cond.size()
        
        device = x.device
        
        q = self.q_proj(x)
        kv = self.kv_proj(obj_cond)
        
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        kv = kv.reshape(B, N, self.num_heads, 2*self.head_dim)
        
        q = q.permute(0, 2, 1, 3)
        kv = kv.permute(0, 2, 1, 3)
        
        k, v = kv.chunk(2, dim=-1)

        # Determine value outputs
        values = self.scaled_dot_product(q, k, v)  # Pass mask as a keyword argument
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(B, N, self.embed_dim)
        o = self.o_proj(values)
        
        # Move tensors to the GPU device
        o = o.to(device)
        self.layer_norm.to(device)
        
        out = self.layer_norm(o)

        return out