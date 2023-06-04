import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModifiedMultiheadAttention(nn.Module):
    """
    Multi-Head Attention Layer with Residual Connections and Layer Normalization followed by a linear projection.

    Args:
        input_dim (int): Dimension of the input matrix.
        embed_dim (int): Dimension of the embedding space.
        num_heads (int): Number of attention heads, must be a divisor of embed_dim.

    """
    def __init__(self, input_dim, embed_dim, num_heads): # embed_dim = num_heads*D_weights! input_dim=D!
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim, bias=False) # Linear layer
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        #self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        #self.o_proj.bias.data.fill_(0)
    
    @staticmethod
    def scaled_dot_product(q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / np.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

        
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size() # B, N
        
        device = x.device
        
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)  # Pass mask as a keyword argument
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)
        
        # Layer Normalization
        B, N, D = o.shape
        layer_norm = nn.LayerNorm([N, D])
        
        # Move tensors to the GPU device
        o = o.to(device)
        layer_norm.to(device)
        
        o = layer_norm(o)

        if return_attention:
            return o, attention
        else:
            return o