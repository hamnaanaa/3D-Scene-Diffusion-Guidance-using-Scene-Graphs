import math
import torch
from torch import nn

# class TimeEmbedding(nn.Module):
#     def __init__(self, dim):
#         super(TimeEmbedding, self).__init__()
#         self.time_embedding = nn.Linear(1, dim)

#     def forward(self, t):
#         t = t.unsqueeze(-1)  # Add an extra dimension for broadcasting
#         t = t.to(self.time_embedding.weight.dtype)  # Ensure input tensor matches weight tensor dtype
#         time_embedded = self.time_embedding(t)
#         return time_embedded


class TimeEmbedding(nn.Module):
    """
    TimeEmbedding module for generating time-based embeddings.

    Args:
        dim (int): The output dimension of the time embeddings.

    Example:
        >>> time_embedding = TimeEmbedding(dim=128)
        >>> inputs = torch.tensor([0, 1, 2, 3, 4])  # Assuming inputs represent time steps
        >>> embeddings = time_embedding(inputs)
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Enable odd dim
        if dim % 2 != 0:
            dim_1 = dim-1
        else:
            dim_1 = dim
            
        self.layers = nn.Sequential(
            nn.Linear(dim_1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        """
        Generate time-based embeddings.

        Args:
            x (torch.Tensor): Input tensor representing time steps. It should be of shape (B,).

        Returns:
            torch.Tensor: Time embeddings of shape (B, dim).

        Note:
            - The input tensor `x` needs to be of type torch.Tensor with length B.
            - The output tensor will have shape (B, dim) where dim is the specified output dimension.

        Example:
            >>> time_embedding = TimeEmbedding(dim=128)
            >>> inputs = torch.tensor([0, 1, 2, 3, 4])  # Assuming inputs represent time steps
            >>> embeddings = time_embedding(inputs)
        """
        device = x.device

        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = x[:, None] * emb[None, :]

        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)


        emb = self.layers(emb)
        return emb
